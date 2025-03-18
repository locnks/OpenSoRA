import sys
import torch
from tqdm import tqdm

from opensora.registry import SCHEDULERS

from .rectified_flow import RFlowScheduler, timestep_transform


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )
    
    def get_bool(self, content):
        return True if content == "True" else False

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        print(f"Nat: prompts {prompts} n={n} num_sampling_steps={self.num_sampling_steps} num_timesteps={self.num_timesteps}")
        # Nat: prompts ['in the forest, a girl is riding a bike on the small path. the sunshine crosses the leaves. many beautiful flowers are growing on both sides. aesthetic score: 6.5.'] num_sampling_steps=30 num_timesteps=1000
        # Nat: prompts ['a girl is riding a bike. aesthetic score: 6.5.']

        with open("/home/stud/ghuang/Open-Sora/experiment_configuration", "r") as f:
            content = f.readlines()
            conditions = content[1].split(',')
            temporal, spatial, experiment = self.get_bool(conditions[0]), self.get_bool(conditions[1]), self.get_bool(conditions[2])

        denosing_step_different_embeddings_experiment = experiment
        split_event_embedding_experiment = True
        if denosing_step_different_embeddings_experiment:
            print("Using different text embeddings for previous x denosing steps...")
            #"""
            # experiment similar to ediff-i
            prompts = prompts[0].split('-=-') # special symbol to split two events
            print(f"Using multiple prompts, prompts={prompts}")
            model_args_1 = text_encoder.encode([prompts[0]])
            model_args_2 = text_encoder.encode([prompts[1]])
            embedding_1 = model_args_1["y"].squeeze(1)
            embedding_2 = model_args_2["y"].squeeze(1)
            mask_1 = model_args_1["mask"]
            mask_2 = model_args_2["mask"]

            # print(f"Testing embeddings, embedding_1 == embedding_2: {all(embedding_1 == embedding_2)}")

            y_null_1 = text_encoder.null(len([prompts[0]]))
            y_null_2 = text_encoder.null(len([prompts[1]]))

            model_args_1["y"] = torch.cat([model_args_1["y"], y_null_1], 0)
            model_args_2["y"] = torch.cat([model_args_2["y"], y_null_2], 0)
            if additional_args is not None:
                model_args_1.update(additional_args)
                model_args_2.update(additional_args)

            try:
                extension = ""
                if temporal:
                    extension = "temporal"
                elif spatial:
                    extension = "spatial"
                torch.save(model_args_1["y"], f"/home/stud/ghuang/Open-Sora/model_args_1_y_{extension}.pt")
                torch.save(model_args_2["y"], f"/home/stud/ghuang/Open-Sora/model_args_2_y_{extension}.pt")
                torch.save(model_args_1["mask"], f"/home/stud/ghuang/Open-Sora/model_args_1_mask_{extension}.pt")
                torch.save(model_args_2["mask"], f"/home/stud/ghuang/Open-Sora/model_args_2_mask_{extension}.pt")
            except:
                print("Failed to save prompt embeddings")
                sys.exit(0)
            with open("/home/stud/ghuang/Open-Sora/_experiment_tmp", "r") as f:
                content = f.read().split('/')
                percentage = float(content[1])
                print(f"Reading percentage denosing_step_different_embeddings_experiment={percentage}")


            #"""
        elif split_event_embedding_experiment:
            print("Concatenating different event embeddings...")
            #"""
            # experiment similar to ediff-i
            prompts = prompts[0].split('-=-') # special symbol to split two events
            print(f"Splitting events, event prompts={prompts}")
            model_args_1 = text_encoder.encode([prompts[0]])
            model_args_2 = text_encoder.encode([prompts[1]])
            embedding_1 = model_args_1["y"].squeeze(1)
            embedding_2 = model_args_2["y"].squeeze(1)
            mask_1 = model_args_1["mask"]
            mask_2 = model_args_2["mask"]

            # print(f"Testing embeddings, embedding_1 == embedding_2: {all(embedding_1 == embedding_2)}")

            y_null_1 = text_encoder.null(len([prompts[0]]))
            y_null_2 = text_encoder.null(len([prompts[1]]))

            model_args_1["y"] = torch.cat([model_args_1["y"], y_null_1], 0)
            model_args_2["y"] = torch.cat([model_args_2["y"], y_null_2], 0)
            if additional_args is not None:
                model_args_1.update(additional_args)
                model_args_2.update(additional_args)
            

            model_args = {}
            last_index_1 = (model_args_1["mask"] == 1).nonzero(as_tuple=True)[1].max().item()
            last_index_2 = (model_args_2["mask"] == 1).nonzero(as_tuple=True)[1].max().item()
            model_args["y"] = torch.cat((model_args_1["y"][:, :, :last_index_1 + 1, :],
                                         model_args_2["y"][:, :, :last_index_2 + 1, :],
                                         model_args_2["y"][:, :, last_index_2 + last_index_1 + 2:, :]), dim=2)
            model_args["mask"] = torch.cat((model_args_1["mask"][:, :last_index_1 + 1],
                                          model_args_2["mask"][:, :last_index_2 + 1],
                                          model_args_2["mask"][:, last_index_1 + last_index_2 + 2:]), dim=1)
            print("Additional arguments", additional_args.keys())
            if additional_args is not None:
                model_args.update(additional_args)
            # print(model_args["mask"], model_args["mask"].shape)
            # sys.exit(0)


            original_model_args = text_encoder.encode([prompts[0] + prompts[1]])
            print("Getting original model args information...")
            # dict_keys(['y', 'mask'])
            # torch.Size([1, 1, 300, 4096])
            # torch.Size([1, 300])

            original_text_encoder_embs = original_model_args["y"]
            original_attention_mask = original_model_args["mask"]

            original_text_encoder_embs = original_text_encoder_embs.squeeze(1)

            original_y_null = text_encoder.null(n)
            original_model_args["y"] = torch.cat([original_model_args["y"], original_y_null], 0)
            if additional_args is not None:
                original_model_args.update(additional_args)

        else:
            model_args = text_encoder.encode(prompts)
            print("Getting model args information...")
            # dict_keys(['y', 'mask'])
            # torch.Size([1, 1, 300, 4096])
            # torch.Size([1, 300])

            text_encoder_embs = model_args["y"]
            attention_mask = model_args["mask"]

            text_encoder_embs = text_encoder_embs.squeeze(1)
            # attention_mask = attention_mask.unsqueeze(-1)
            # print(text_encoder_embs.shape, attention_mask.shape) # torch.Size([1, 300, 4096]) torch.Size([1, 300])  
            # print(f"Nat: unpadded embeddings shape {text_encoder_embs[0, attention_mask[0].bool()].shape}")
            # Nat: unpadded embeddings shape torch.Size([16, 4096])

            y_null = text_encoder.null(n)
            model_args["y"] = torch.cat([model_args["y"], y_null], 0)
            if additional_args is not None:
                model_args.update(additional_args)


        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        # print(f"Nat: the shape of the mask {mask.shape}")
        # Nat: the shape of the mask torch.Size([1, 30])

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        
        for i, t in progress_wrap(enumerate(timesteps)):
            with open("/home/stud/ghuang/Open-Sora/denosing_step_information", "w") as f:
                f.write(str(i))
            
            if denosing_step_different_embeddings_experiment:
                #-=- replace embeddings
                #"""
                if i < int(percentage * len(timesteps)):
                    # print(f"Denoising step {i}/{len(timesteps)} is using {prompts[0]}")
                    model_args = model_args_1
                else:
                    # print(f"Denoising step {i}/{len(timesteps)} is using {prompts[1]}")
                    model_args = model_args_2
                #"""
            if split_event_embedding_experiment:
                print("Using text embeddings split...")
                with open("/home/stud/ghuang/Open-Sora/causal_mask_ratio", "r") as f:
                    content = f.read()
                    content = content.split('-=-')
                    prompt_key = content[0]
                    token_boundary = int(content[1])
                    denoising_step_ratio = float(content[3])
                    denoising_step_condition = content[2]
                    if denoising_step_condition == "_causal_mask_experiment_2_":
                        if i >= denoising_step_ratio * 30:
                            model_args = original_model_args
                    else:
                        if i < denoising_step_ratio * 30:
                            model_args = original_model_args

            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            # this means the batch size is using 2 for each prompt
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)
