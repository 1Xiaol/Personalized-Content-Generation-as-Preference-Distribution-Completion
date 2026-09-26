"""Frozen embedding-conditioned caption decoder components."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

PREFIX_TOKENS = 16
MAPPER_WIDTH = 1024
MAPPER_LAYERS = 2
MAPPER_HEADS = 8
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "out_proj",
)
CONDITION_WEIGHT = 0.1
CONDITION_MARGIN = 0.1
CONDITION_EVERY = 4
PERTURB_FRACTION = 0.2


class CompactPrefixMapper(nn.Module):
    def __init__(self, latent_dimension: int, lm_dimension: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(latent_dimension),
            nn.Linear(latent_dimension, MAPPER_WIDTH),
            nn.GELU(),
            nn.Linear(MAPPER_WIDTH, MAPPER_WIDTH),
        )
        self.queries = nn.Parameter(torch.empty(PREFIX_TOKENS, MAPPER_WIDTH))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=MAPPER_WIDTH,
            nhead=MAPPER_HEADS,
            dim_feedforward=4 * MAPPER_WIDTH,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=MAPPER_LAYERS)
        self.output = nn.Sequential(nn.LayerNorm(MAPPER_WIDTH), nn.Linear(MAPPER_WIDTH, lm_dimension))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.input(latent.to(self.queries.dtype))
        tokens = self.queries.unsqueeze(0) + hidden.unsqueeze(1)
        return self.output(self.transformer(tokens))


class EmbeddingConditionedCaptionDecoder(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        mapper: CompactPrefixMapper,
        before_soft_prefix_ids: Sequence[int],
        after_soft_prefix_ids: Sequence[int],
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.mapper = mapper
        self.register_buffer(
            "before_soft_prefix_ids", torch.tensor(list(before_soft_prefix_ids), dtype=torch.long)
        )
        self.register_buffer(
            "after_soft_prefix_ids", torch.tensor(list(after_soft_prefix_ids), dtype=torch.long)
        )

    def forward(
        self, latent: torch.Tensor, caption_ids: torch.Tensor, caption_mask: torch.Tensor
    ) -> torch.Tensor:
        embedding_layer = self.language_model.get_input_embeddings()
        token_embeddings = embedding_layer(caption_ids)
        prefix = self.mapper(latent).to(token_embeddings.dtype)
        before = embedding_layer(self.before_soft_prefix_ids).unsqueeze(0).expand(len(caption_ids), -1, -1)
        after = embedding_layer(self.after_soft_prefix_ids).unsqueeze(0).expand(len(caption_ids), -1, -1)
        inputs = torch.cat((before, prefix, after, token_embeddings), dim=1)
        framing_length = before.shape[1] + PREFIX_TOKENS + after.shape[1]
        framing_mask = torch.ones(
            (len(caption_ids), framing_length), dtype=caption_mask.dtype, device=caption_mask.device
        )
        attention_mask = torch.cat((framing_mask, caption_mask), dim=1)
        ignore = torch.full(
            (len(caption_ids), framing_length), -100, dtype=torch.long, device=caption_ids.device
        )
        labels = torch.cat((ignore, caption_ids.masked_fill(caption_mask == 0, -100)), dim=1)
        logits = self.language_model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits
        targets = labels[:, 1:]
        token_loss = F.cross_entropy(
            logits[:, :-1].float().transpose(1, 2), targets, ignore_index=-100, reduction="none"
        )
        valid = targets != -100
        return ((token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)).mean()


def spherical_perturb(
    latent: torch.Tensor, radius: float, fraction: float, *, seed: int
) -> tuple[torch.Tensor, int]:
    if radius <= 0 or fraction <= 0:
        return latent, 0
    generator = torch.Generator(device=latent.device)
    generator.manual_seed(seed)
    mask = torch.rand(len(latent), generator=generator, device=latent.device) < fraction
    count = int(mask.sum())
    if count == 0:
        return latent, 0
    selected = latent[mask].float()
    noise = torch.randn(selected.shape, generator=generator, device=latent.device)
    tangent = noise - (noise * selected).sum(1, keepdim=True) * selected
    tangent /= tangent.norm(dim=1, keepdim=True).clamp_min(1e-12)
    radii = radius * torch.rand((count, 1), generator=generator, device=latent.device)
    perturbed = torch.cos(radii) * selected + torch.sinc(radii / math.pi) * tangent * radii
    result = latent.clone()
    result[mask] = F.normalize(perturbed, dim=1).to(result.dtype)
    return result, count


def shuffled_latent(latent: torch.Tensor, max_cosine: float = 0.8) -> torch.Tensor:
    """Find an in-batch conditioning negative for every caption."""
    if len(latent) < 2:
        raise ValueError("conditioning contrast needs batch size >= 2")
    result = torch.empty_like(latent)
    assigned = torch.zeros(len(latent), dtype=torch.bool, device=latent.device)
    index = torch.arange(len(latent), device=latent.device)
    for shift in range(1, len(latent)):
        candidate = latent.index_select(0, (index - shift) % len(latent))
        choose = ~assigned & ((latent.float() * candidate.float()).sum(1) < max_cosine)
        result[choose] = candidate[choose]
        assigned |= choose
        if bool(assigned.all()):
            return result
    raise RuntimeError("no valid shuffled conditioning negative for every row")


def caption_decoder_training_loss(
    decoder: EmbeddingConditionedCaptionDecoder,
    latent: torch.Tensor,
    caption_ids: torch.Tensor,
    caption_mask: torch.Tensor,
    *,
    update: int,
    shuffle_max_cosine: float = 0.8,
) -> dict[str, torch.Tensor]:
    cross_entropy = decoder(latent, caption_ids, caption_mask)
    condition = torch.zeros((), dtype=cross_entropy.dtype, device=cross_entropy.device)
    if update % CONDITION_EVERY == 0:
        wrong = shuffled_latent(latent, shuffle_max_cosine)
        shuffled_loss = decoder(wrong, caption_ids, caption_mask)
        condition = F.softplus(CONDITION_MARGIN + cross_entropy - shuffled_loss)
    return {
        "loss": cross_entropy + CONDITION_WEIGHT * condition,
        "cross_entropy": cross_entropy,
        "condition": condition,
    }


def attach_qwen35_lora(model_name_or_path: str) -> nn.Module:
    """Load a user-supplied Qwen3.5 model and attach the formal LoRA modules."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("install the optional 'caption-decoder' dependencies") from error
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.requires_grad_(False)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
    )
    return get_peft_model(model, config)
