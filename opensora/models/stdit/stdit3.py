import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from transformers import PretrainedConfig, PreTrainedModel

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    SeqParallelAttention,
    SeqParallelMultiHeadCrossAttention,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
)
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint
from opensora.registry import MODELS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.datasets.aspect import get_num_frames
from opensora.datasets import save_sample


class STDiT3Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        temporal=False,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism

        if self.enable_sequence_parallelism and not temporal:
            # print("Nat: using seqparallelattention")
            # for spatial part and if sequence parallism is enabled, it will use this part
            # otherwise for spaital part, if sequence parallism is not enabled, it will not use this part
            attn_cls = SeqParallelAttention
            mha_cls = SeqParallelMultiHeadCrossAttention
        else:
            # print("Nat: use normal multiheadattention")
            attn_cls = Attention
            mha_cls = MultiHeadCrossAttention

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
        )
        self.cross_attn = mha_cls(hidden_size, num_heads)
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        # so the features where x_mask=1, use the corresponding location's feature of x
        # for the features where x_mask=0, use the corresponding location's feature of masked_x
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(
        self,
        x,
        y,
        t,
        mask=None,  # text mask
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
    ):
        # prepare modulate parameters
        B, N, C = x.shape
        # print(f"Nat: block input x {x.shape} y {y.shape} t {t.shape}")
        # Nat: block input x torch.Size([2, 28800, 1152]) y torch.Size([1, 42, 1152]) t torch.Size([2, 6912])
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)

        # modulate (attention)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # attention
        # the attention for temporal and spatial is just which dimension is merged and which dimension is treated as the query
        if self.temporal:
            x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
            x_m = self.attn(x_m)
            # here is the self attention, which is using class Attention in layers/blocks.py
            # only for temporal block, the q and k will be applied with rotary positional encoding
            x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
        else:
            x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
            x_m = self.attn(x_m)
            x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)

        # cross attention
        x = x + self.cross_attn(x, y, mask)

        # modulate (MLP)
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)

        return x


class STDiT3Config(PretrainedConfig):
    model_type = "STDiT3"

    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=512,
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        caption_channels=4096,
        model_max_length=300,
        qk_norm=True,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        only_train_temporal=False,
        freeze_y_embedder=False,
        skip_y_embedder=False,
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.only_train_temporal = only_train_temporal
        self.freeze_y_embedder = freeze_y_embedder
        self.skip_y_embedder = skip_y_embedder
        super().__init__(**kwargs)


class STDiT3(PreTrainedModel):
    config_class = STDiT3Config

    def __init__(self, config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels

        # model size related
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # computation related
        self.drop_path = config.drop_path
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel
        self.enable_sequence_parallelism = config.enable_sequence_parallelism

        # input size related
        self.patch_size = config.patch_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(config.hidden_size)
        # the rotary positional encoding is applied on the head dimension
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads) # 1152 // 16 = 72
        # TODO: replace the rope with positional embedding in different temporal blocks
        # ref: https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
        # TODO: deactivate the previous x% temporal blocks, x=0,25,50,75,100

        # embedding
        self.x_embedder = PatchEmbed3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.fps_embedder = SizeEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

        # spatial blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.spatial_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                )
                for i in range(config.depth)
            ]
        )

        # temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        # temporal_block_ratio = 0.5
        # print(f"Previous {int(temporal_block_ratio * config.depth)}/{config.depth} temporal blocks is not using RoPE")

        print("Using normal temporal blocks, all with RoPE")
        self.temporal_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                    # temporal
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys
                )
                for i in range(config.depth)
            ]
        )

        """
        temporal_block_index = 3
        print(f"Temporal block index={temporal_block_index} is not using RoPE")
        temporal_blocks = []
        for i in range(config.depth):
            if i == temporal_block_index:
                temporal_blocks.append(STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                    # temporal
                    temporal=True,
                    rope=None
                ))
            else:
                temporal_blocks.append(STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                    # temporal
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                ))

        self.temporal_blocks = nn.ModuleList(
            temporal_blocks
        )
        """
        temporal_block_rope_experiment = False # remove previous x% and last x% temporal block's RoPE
        if temporal_block_rope_experiment:
            print(f"Remove x% temporal blocks' RoPE, filename=/home/stud/ghuang/Open-Sora/tmp...")
            with open("/home/stud/ghuang/Open-Sora/tmp", "r") as f:
                content = f.read()
                content = content.split("-")
                temporal_block_ratio = float(content[1])
                
                if content[0] == "first":
                    print(f"Previous {int(temporal_block_ratio * config.depth)}/{config.depth} is not using RoPE")
                    self.temporal_blocks = nn.ModuleList(
                        [
                            STDiT3Block(
                                hidden_size=config.hidden_size,
                                num_heads=config.num_heads,
                                mlp_ratio=config.mlp_ratio,
                                drop_path=drop_path[i],
                                qk_norm=config.qk_norm,
                                enable_flash_attn=config.enable_flash_attn,
                                enable_layernorm_kernel=config.enable_layernorm_kernel,
                                enable_sequence_parallelism=config.enable_sequence_parallelism,
                                # temporal
                                temporal=True,
                                rope=None
                            )
                            for i in range(int(temporal_block_ratio * config.depth))
                        ] + 
                        [
                            STDiT3Block(
                                hidden_size=config.hidden_size,
                                num_heads=config.num_heads,
                                mlp_ratio=config.mlp_ratio,
                                drop_path=drop_path[i],
                                qk_norm=config.qk_norm,
                                enable_flash_attn=config.enable_flash_attn,
                                enable_layernorm_kernel=config.enable_layernorm_kernel,
                                enable_sequence_parallelism=config.enable_sequence_parallelism,
                                # temporal
                                temporal=True,
                                rope=self.rope.rotate_queries_or_keys,
                            )
                            for i in range(config.depth - int(temporal_block_ratio * config.depth))
                        ]
                    )
                else:
                    print(f"Last {int(temporal_block_ratio * config.depth)}/{config.depth} is not using RoPE")
                    self.temporal_blocks = nn.ModuleList(
                        [
                            STDiT3Block(
                                hidden_size=config.hidden_size,
                                num_heads=config.num_heads,
                                mlp_ratio=config.mlp_ratio,
                                drop_path=drop_path[i],
                                qk_norm=config.qk_norm,
                                enable_flash_attn=config.enable_flash_attn,
                                enable_layernorm_kernel=config.enable_layernorm_kernel,
                                enable_sequence_parallelism=config.enable_sequence_parallelism,
                                # temporal
                                temporal=True,
                                rope=self.rope.rotate_queries_or_keys
                            )
                            for i in range(config.depth - int(temporal_block_ratio * config.depth))
                        ] + 
                        [
                            STDiT3Block(
                                hidden_size=config.hidden_size,
                                num_heads=config.num_heads,
                                mlp_ratio=config.mlp_ratio,
                                drop_path=drop_path[i],
                                qk_norm=config.qk_norm,
                                enable_flash_attn=config.enable_flash_attn,
                                enable_layernorm_kernel=config.enable_layernorm_kernel,
                                enable_sequence_parallelism=config.enable_sequence_parallelism,
                                # temporal
                                temporal=True,
                                rope=None,
                            )
                            for i in range(int(temporal_block_ratio * config.depth))
                        ]
                    )

        """
        print("Remove one temporal block's RoPE")
        with open("/home/stud/ghuang/Open-Sora/tmp", "r") as f:
            content = f.read()
            content = content.split("-")
            replace_index = int(content[1])
            print(f"Replace index experiment: the temporal block {replace_index}/{config.depth} is not using RoPE")
            self.temporal_blocks = []
            for i in range(0, config.depth):
                if i == replace_index:
                    self.temporal_blocks.append(STDiT3Block(
                        hidden_size=config.hidden_size,
                        num_heads=config.num_heads,
                        mlp_ratio=config.mlp_ratio,
                        drop_path=drop_path[i],
                        qk_norm=config.qk_norm,
                        enable_flash_attn=config.enable_flash_attn,
                        enable_layernorm_kernel=config.enable_layernorm_kernel,
                        enable_sequence_parallelism=config.enable_sequence_parallelism,
                        # temporal
                        temporal=True,
                        rope=None
                    ))
                else:
                    self.temporal_blocks.append(STDiT3Block(
                        hidden_size=config.hidden_size,
                        num_heads=config.num_heads,
                        mlp_ratio=config.mlp_ratio,
                        drop_path=drop_path[i],
                        qk_norm=config.qk_norm,
                        enable_flash_attn=config.enable_flash_attn,
                        enable_layernorm_kernel=config.enable_layernorm_kernel,
                        enable_sequence_parallelism=config.enable_sequence_parallelism,
                        # temporal
                        temporal=True,
                        rope=self.rope.rotate_queries_or_keys,
                    ))
            self.temporal_blocks = nn.ModuleList(self.temporal_blocks)
        """

        # final layer
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.patch_size), self.out_channels)

        self.initialize_weights()
        if config.only_train_temporal:
            for param in self.parameters():
                param.requires_grad = False
            for block in self.temporal_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        if config.freeze_y_embedder:
            for param in self.y_embedder.parameters():
                param.requires_grad = False
        
        # define the vae used to decode the noised tensor
        device = "cuda"
        self.model_dtype = torch.bfloat16
        cfg = parse_configs(training=False)
        self.num_frames = get_num_frames(cfg.num_frames)
        self.intermediate_result_directory = "/home/stud/ghuang/Open-Sora/samples/samples/intermediate_results"
        fps = cfg.fps
        self.save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))

        self.vae = build_module(cfg.vae, MODELS).to(device, self.model_dtype).eval()

    def initialize_weights(self):
        # Nat: the weights of the STDiT3 is initialized here
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fps_embedder
        nn.init.normal_(self.fps_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.fps_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].bias, 0)

        # Initialize timporal blocks
        for block in self.temporal_blocks:
            nn.init.constant_(block.attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.mlp.fc2.weight, 0)

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def encode_text(self, y, mask=None):
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        # print(f"Nat: the output y after y embedder {y.shape}")
        # Nat: the output y after y embedder torch.Size([2, 1, 300, 1152])
        if mask is not None:
            # print(f"Nat: mask it not None, mask.shape {mask.shape}")
            # Nat: mask it not None, mask.shape torch.Size([1, 300])
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            # squeeze(dim) removes a dimension of size 1 along the specified dim.
            # unsqueeze(dim), adds a dimension of size 1 along the specified dim
            # print(f"Nat: before mask select, y.shape {y.shape} mask.shape {mask.shape}")
            # Nat: before mask select, y.shape torch.Size([2, 1, 300, 1152]) mask.shape torch.Size([2, 300])
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            # here, the -1 in view() means this dimension will be inferred automatically
            # print(f"Nat: after mask select, y.shape {y.shape} mask.shape {mask.shape}")
            # Nat: after mask select, y.shape torch.Size([1, 32, 1152]) mask.shape torch.Size([2, 300])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        # print(f"Output y {y.shape} y_lens {y_lens}")
        # Output y torch.Size([1, 32, 1152]) y_lens [16, 16]
        return y, y_lens

    def get_bool(self, content):
        return True if content == "True" else False

    def forward(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs):


        with open("/home/stud/ghuang/Open-Sora/experiment_configuration", "r") as f:
            content = f.readlines()
            conditions = content[1].split(',')
            temporal, spatial, experiment = self.get_bool(conditions[0]), self.get_bool(conditions[1]), self.get_bool(conditions[2])

        # control parameters
        temporal_block_experiment = temporal # temporal block level, use different embeddings
        spatial_block_experiment = spatial

        # print(f"Nat: input shape x:{x.shape} timestep:{timestep} y:{y.shape}")
        timestep_progress_name = str(timestep[0].item()).split('.')[0]
        # output sample: 
        # Nat: input shape x:torch.Size([2, 4, 30, 90, 160]) timestep:tensor([1000., 1000.], device='cuda:2') y:torch.Size([2, 1, 300, 4096]) 
        # here 4 is the out channels, 30 is because of num_frames=102, 204, etc. 90, 160 is because of frame resolution
        dtype = self.x_embedder.proj.weight.dtype
        B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)
        # x, the video frames tensor
        # timestep, the timestamp
        # y, the conditioning, ie, the text descriptions

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        # Tx=30, Hx=90. Wx=160
        # Tx, Hx, Wx, the number of frames, the height, the width, 
        # print(f"Nat: self.path_size {self.patch_size}")
        # Nat: self.path_size [1, 2, 2]
        T, H, W = self.get_dynamic_size(x)
        # print(f"Nat: the T={T} H={H} W={W}")
        # Nat: the T=30 H=45 W=80

        # adjust for sequence parallelism
        # we need to ensure H * W is divisible by sequence parallel size
        # for simplicity, we can adjust the height to make it divisible
        if self.enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            if H % sp_size != 0:
                h_pad_size = sp_size - H % sp_size
            else:
                h_pad_size = 0

            if h_pad_size > 0:
                hx_pad_size = h_pad_size * self.patch_size[1]

                # pad x along the H dimension
                H += h_pad_size
                x = F.pad(x, (0, 0, 0, hx_pad_size))
        # print(f"Nat: after parallism padding the H={H}")
        # Nat: after parallism padding the H=48
        # patched H=48, W=80, S=3840=H*W

        S = H * W # the total number of spatial locations
        base_size = round(S**0.5) # the square root of the total number of spatial locations
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        # positional embedding for every spatial location on the frame
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C], the embedding for the timestamp
        fps = self.fps_embedder(fps.unsqueeze(1), B) # the embedding for the frame per second
        t = t + fps # the timestamp embedding and the fps embedding are added together
        t_mlp = self.t_block(t) # SiLU + Linear Layer
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)
        # print(f"Nat: t_mlp.shape {t_mlp.shape}")
        # Nat: t_mlp.shape torch.Size([2, 6912])
        # === get y embed ===
        # print(f"Nat: before y.shape {y.shape}")
        # Nat: before y.shape torch.Size([2, 1, 300, 4096])
        if self.config.skip_y_embedder:
            # skip the embedding process of conditioning
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            # print(f"Nat: running here, use y embedder, mask.shape {mask.shape}")
            # Nat: running here, use y embedder, mask.shape torch.Size([1, 300])
            y, y_lens = self.encode_text(y, mask)

            if temporal_block_experiment or spatial_block_experiment:
                extension = "temporal" if temporal_block_experiment else "spatial"
                print(f"Using different prompt embeddings for different blocks, extension={extension}...")
                prompt_1_embedding = torch.load(f"/home/stud/ghuang/Open-Sora/model_args_1_y_{extension}.pt")
                prompt_2_embedding = torch.load(f"/home/stud/ghuang/Open-Sora/model_args_2_y_{extension}.pt")
                prompt_1_mask = torch.load(f"/home/stud/ghuang/Open-Sora/model_args_1_mask_{extension}.pt")
                prompt_2_mask = torch.load(f"/home/stud/ghuang/Open-Sora/model_args_2_mask_{extension}.pt")

                prompt_1_embedding = prompt_1_embedding.to(dtype)
                prompt_2_embedding = prompt_2_embedding.to(dtype)

                y_1, y_lens_1 = self.encode_text(prompt_1_embedding, prompt_1_mask)
                y_2, y_lens_2 = self.encode_text(prompt_2_embedding, prompt_2_mask)


        # print(f"Nat: after y.shape {y.shape}")
        # Nat: after y.shape torch.Size([1, 32, 1152])

        # === get x embed ===
        # print(f"Nat: before x_emebdder x.shape {x.shape}")
        # ([2, 4, 30, 96, 160])
        x = self.x_embedder(x)  # [B, N, C]
        # print(f"Nat: after embedder x.shape {x.shape}")
        # Nat: after embedder x.shape torch.Size([2, 108000, 1152])
        # ([2, 115200, 1152])
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S) 
        # print(f"Nat: after rearrange x.shape {x.shape} T={T} S={S}")
        # Nat: after rearrange x.shape torch.Size([2, 30, 3600, 1152]) T=30 S=3600
        # Nat: after rearrange x.shape torch.Size([2, 30, 3840, 1152]) T=30 S=3840
        # with one card, Nat: after rearrange x.shape torch.Size([2, 30, 3600, 1152]) T=30 S=3600, H=45, W=80
        # seperates the T and S, T is the number of frames, S is the number of spatial locations
        x = x + pos_emb # positional embedding is added
        # print(f"Nat: pos_emb.shape={pos_emb.shape}")
        # Nat: pos_emb.shape=torch.Size([1, 3600, 1152])

        # shard over the sequence dim if sp is enabled
        # the split dimension is the spatial dimension
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group()) 

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S) # this for each node, the video frames tensor
        # print(f"Nat: the x.shape {x.shape} T={T} S={S} H={H} W={W}")
        # Nat: the x.shape torch.Size([2, 28800, 1152]) T=30 S=960, because there are 4 cards, then S=origina_S//4. Origin_S=3840=28*80
        # Nat: with one H100, the output is:
        # Nat: the x.shape torch.Size([2, 108000, 1152]) T=30 S=3600 H=45 W=80

        # === blocks ===
        # the spatial block is 28 SDiTBlocks
        # the temporal block is 28 SDiTBlocks but with                     # temporal
        # temporal=True,
        # rope=self.rope.rotate_queries_or_keys,
        # here the 28 is the default depth
        # they use different attention mechanisms:
        # if self.enable_sequence_parallelism and not temporal:
        #     attn_cls = SeqParallelAttention
        #     mha_cls = SeqParallelMultiHeadCrossAttention
        # else:
        #     attn_cls = Attention
        #     mha_cls = MultiHeadCrossAttention

        if temporal_block_experiment:
            print(f"Going through temporal block experiment")
            block_index = 0
            with open("/home/stud/ghuang/Open-Sora/temporal_experiment_tmp", "r") as f:
                content = f.read()
                content = content.split("/")
                temporal_experiment_condition = content[0]
                temporal_experiment_percentage = float(content[1])
            print(f"Using different prompts for different temporal blocks, temporal percentage={temporal_experiment_percentage}, condition={temporal_experiment_condition}...")
            prompt_embedding = None
            prompt_embedding_lens = None

        if spatial_block_experiment:
            print(f"Going through spatial block experiment")
            block_index = 0
            with open("/home/stud/ghuang/Open-Sora/spatial_experiment_tmp", "r") as f:
                content = f.read()
                content = content.split("/")
                spatial_experiment_condition = content[0]
                spatial_experiment_percentage = float(content[1])
            print(f"Using different prompts for different spatial blocks, spatial percentage={spatial_experiment_percentage}, condition={spatial_experiment_condition}...")
            prompt_embedding = None
            prompt_embedding_lens = None

        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            if not temporal_block_experiment and not spatial_block_experiment:
                x = auto_grad_checkpoint(spatial_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
                x = auto_grad_checkpoint(temporal_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
            # auto_grad_checkpoint: This is a memory optimization technique used to reduce GPU 
            # memory consumption during training. It saves intermediate results selectively and 
            # recomputes them as needed during the backward pass.

            if temporal_block_experiment:
                if block_index < int(temporal_experiment_percentage * 28):
                    # print(f"spatial and temporal block {block_index}/28 is using prompt_1...")
                    prompt_embedding = y_1
                    prompt_embedding_lens = y_lens_1
                else:
                    # print(f"spatial and temporal block {block_index}/28 is using prompt_2...")
                    prompt_embedding = y_2
                    prompt_embedding_lens = y_lens_2

                # x = auto_grad_checkpoint(spatial_block, x, prompt_embedding, t_mlp, prompt_embedding_lens, x_mask, t0_mlp, T, S)
                if temporal_experiment_condition != "experiment2":
                # experiment 0, 1 use the event_0+event_1, the first prompt
                    # experiment 2 fix event_1, or fix event_0, the second prompt
                    x = auto_grad_checkpoint(spatial_block, x, y_1, t_mlp, y_lens_1, x_mask, t0_mlp, T, S)
                else:
                    x = auto_grad_checkpoint(spatial_block, x, y_2, t_mlp, y_lens_2, x_mask, t0_mlp, T, S)
                x = auto_grad_checkpoint(temporal_block, x, prompt_embedding, t_mlp, prompt_embedding_lens, x_mask, t0_mlp, T, S)

                # x = auto_grad_checkpoint(temporal_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
                block_index += 1

            if spatial_block_experiment:
                if block_index < int(spatial_experiment_percentage * 28):
                    # print(f"spatial and temporal block {block_index}/28 is using prompt_1...")
                    prompt_embedding = y_1
                    prompt_embedding_lens = y_lens_1
                else:
                    # print(f"spatial and temporal block {block_index}/28 is using prompt_2...")
                    prompt_embedding = y_2
                    prompt_embedding_lens = y_lens_2

                x = auto_grad_checkpoint(spatial_block, x, prompt_embedding, t_mlp, prompt_embedding_lens, x_mask, t0_mlp, T, S)
                if spatial_experiment_condition != "experiment2":
                # experiment 0, 1 use the event_0+event_1, the first prompt
                    # experiment 2 fix event_1, or fix event_0, the second prompt
                    x = auto_grad_checkpoint(temporal_block, x, y_1, t_mlp, y_lens_1, x_mask, t0_mlp, T, S)
                else:
                    x = auto_grad_checkpoint(temporal_block, x, y_2, t_mlp, y_lens_2, x_mask, t0_mlp, T, S)

                # x = auto_grad_checkpoint(temporal_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
                block_index += 1

            # the parameters of the STDiT block:
            # def forward(
            #     self,
            #     x,
            #     y,
            #     t,
            #     mask=None,  # text mask
            #     x_mask=None,  # temporal mask
            #     t0=None,  # t with timestamp=0
            #     T=None,  # number of frames
            #     S=None,  # number of pixel patches
            # ):
            # y_lens: the actual length of input text sequences
            # x_mask: This is a mask for temporal (time-based) operations on the input tensor; It allows selective processing of different frames or time steps
            # t0_mlp: The purpose of t0_mlp is to provide a baseline or reference embedding for the zero timestep. This is particularly useful in scenarios

        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group()) # restore the splitted spatial information
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === final layer ===
        # print(f"Nat: before final layer x.shape={x.shape} T={T} S={S}")
        # Nat: before final layer x.shape=torch.Size([2, 108000, 1152]) T=30 S=3600
        x = self.final_layer(x, t, x_mask, t0, T, S)
        # the C=1152 is the hidden size
        # print(f"Nat: after final_layer x.shape={x.shape}")
        # Nat: after final_layer x.shape=torch.Size([2, 108000, 32])
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)
        # print(f"Nat: after unpatchify x.shape={x.shape}")
        # Nat: after unpatchify x.shape=torch.Size([2, 8, 30, 90, 160])

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        # print(f"Nat: predicted noise output {x.shape}")
        # Nat: predicted noise output torch.Size([2, 8, 30, 90, 160])
        # 2 -> batch size, but with the same content
        # 8 -> 4*2, previous 4 is the denoised video, later 4 is the predicted noise
        first_denoised_video = x[0, :4]
        second_denoised_video = x[1, :4]
        # notice that since the output timestep tensor is exactly the same, this means first_denoised_video
        # and the second_denoised_video are the same
        # samples = vae.decode(samples.to(dtype), num_frames=num_frames)
        # Nat: the following code saved the intermediate denoising result to directory
        # first_sample = self.vae.decode(first_denoised_video.to(self.model_dtype), num_frames=self.num_frames)
        # second_sample = self.vae.decode(second_denoised_video.to(self.model_dtype), num_frames=self.num_frames)
        # first_save_path = save_sample(
        #     first_sample[0],
        #     fps=self.save_fps,
        #     save_path=os.path.join(self.intermediate_result_directory, "first_" + timestep_progress_name),
        #     verbose=False,
        # )
        # print(f"Saved the first sample to {first_save_path}")
        # second_save_path = save_sample(
        #     second_sample[0],
        #     fps=self.save_fps,
        #     save_path=os.path.join(self.intermediate_result_directory, "second_" + timestep_progress_name),
        #     verbose=False
        # )
        # print(f"Saved the second sample to {second_save_path}")
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size # 1, 2, 2
        x = rearrange(
            x,
            # N_t=30, N_h=45, N_w=80. T_p=1. H_p=2. W_p=2
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x


@MODELS.register_module("STDiT3-XL/2")
def STDiT3_XL_2(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = STDiT3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = STDiT3Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
        model = STDiT3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model


@MODELS.register_module("STDiT3-3B/2")
def STDiT3_3B_2(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = STDiT3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = STDiT3Config(depth=28, hidden_size=1872, patch_size=(1, 2, 2), num_heads=26, **kwargs)
        model = STDiT3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model
