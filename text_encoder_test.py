# from opensora.models.text_encoder.t5 import text_preprocessing
# from opensora.registry import MODELS, SCHEDULERS, build_module
# from opensora.utils.config_utils import parse_configs
# import torch

# cfg = parse_configs(training=False)

#     # == device and dtype ==
# device = "cuda" if torch.cuda.is_available() else "cpu"
# text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
# prompt = "A girl is riding a bike."
# result = text_encoder.encode(prompt)
# print(result["y"].shape, result["mask"].shape)
# embeddings = result["y"]
# mask = result["mask"]
# embeddings = embeddings.squeeze(1)
# valid_embeddings = embeddings[mask.bool()]
# print(valid_embeddings.shape)

# t5 is using the tokenizer of the huggingface
# TODO: find the configuraiton of the tokenizer that is being 

# y, y_lens = encode_text(y, mask)

# hidden_size = 1152

# def encode_text(self, y, mask=None):
#     y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
#     if mask is not None:
#         if mask.shape[0] != y.shape[0]:
#             mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
#         mask = mask.squeeze(1).squeeze(1)
#         y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, hidden_size)
#         y_lens = mask.sum(dim=1).tolist()
#     else:
#         y_lens = [y.shape[2]] * y.shape[0]
#         y = y.squeeze(1).view(1, -1, hidden_size)
#     return y, y_lens


# class CaptionEmbedder(nn.Module):
#     """
#     Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
#     """

#     def __init__(
#         self,
#         in_channels,
#         hidden_size,
#         uncond_prob,
#         act_layer=nn.GELU(approximate="tanh"),
#         token_num=120,
#     ):
#         super().__init__()
#         self.y_proj = Mlp(
#             in_features=in_channels,
#             hidden_features=hidden_size,
#             out_features=hidden_size,
#             act_layer=act_layer,
#             drop=0,
#         )
#         self.register_buffer(
#             "y_embedding",
#             torch.randn(token_num, in_channels) / in_channels**0.5,
#         )
#         self.uncond_prob = uncond_prob

#     def token_drop(self, caption, force_drop_ids=None):
#         """
#         Drops labels to enable classifier-free guidance.
#         """
#         if force_drop_ids is None:
#             drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
#         else:
#             drop_ids = force_drop_ids == 1
#         caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
#         return caption

#     def forward(self, caption, train, force_drop_ids=None):
#         if train:
#             assert caption.shape[2:] == self.y_embedding.shape
#         use_dropout = self.uncond_prob > 0
#         if (train and use_dropout) or (force_drop_ids is not None):
#             caption = self.token_drop(caption, force_drop_ids)
#         caption = self.y_proj(caption)
#         return caption

from transformers import AutoTokenizer, T5EncoderModel
import torch

pretrained_model_name = "DeepFloyd/t5-v1_1-xxl"
model_path = "THUDM/CogVideoX-5b"
tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
            cache_dir=None,
            local_files_only=False,
)
prompt = 'a girl is riding a bike. aesthetic score: 6.5.'
text_tokens_and_mask = tokenizer(
    prompt,
    max_length=300,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt",
)

input_ids = text_tokens_and_mask["input_ids"]
attention_mask = text_tokens_and_mask["attention_mask"]
print(input_ids, attention_mask)
# Create a dictionary of token_id to word mapping
token_words = {}
for token_id in input_ids[0]:
    token_words[token_id.item()] = tokenizer.decode(token_id)
print(token_words)

# This will show you how the text was tokenized
tokens_splits = tokenizer.convert_ids_to_tokens(input_ids[0])
print(tokens_splits)
# {71: 'A', 3202: 'girl', 19: 'is', 7494: 'riding', 3: '', 9: 'a', 3724: 'bike', 5: '.', 1: '</s>', 0: '<pad>'}

model = T5EncoderModel.from_pretrained(
    model_path,
    subfolder="text_encoder",
    cache_dir=None,
    local_files_only=False
).eval()

with torch.no_grad():
    text_encoder_embs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )["last_hidden_state"]
print(text_encoder_embs.shape)
# torch.Size([1, 300, 4096])
unpadded_embedding = text_encoder_embs[0, attention_mask[0].bool()]
print(unpadded_embedding.shape)
# torch.Size([9, 4096])
