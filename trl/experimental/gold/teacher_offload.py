"""
Teacher model CPU offload for Gemma4.

CPU에 pinned memory로 teacher를 유지하고, forward 시 layer 단위로
GPU 복사본을 만들어 연산 후 버리는 방식. GPU→CPU 전송이 없어 빠름.

사용법:
    from teacher_offload import prepare_teacher_offload, teacher_forward_offload

    # 초기화 (1회)
    prepare_teacher_offload(teacher_model)

    # forward (매 step)
    logits = teacher_forward_offload(teacher_model, input_ids, attention_mask, device)
"""

import torch
import torch.nn.functional as F
from transformers.models.gemma4.modeling_gemma4 import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)


def prepare_teacher_offload(model):
    """Teacher 모델의 파라미터를 CPU pinned memory로 변환."""
    model.to("cpu")
    model.eval()
    model.requires_grad_(False)
    for param in model.parameters():
        param.data = param.data.pin_memory()
    for buf in model.buffers():
        try:
            buf.data = buf.data.pin_memory()
        except Exception:
            pass


def _swap_to_gpu(module, device):
    """모듈의 파라미터/버퍼를 GPU로 복사하고, CPU 원본 참조를 반환."""
    originals = {}
    for name, param in module.named_parameters():
        originals[("p", name)] = param.data
        param.data = param.data.to(device, non_blocking=True)
    for name, buf in module.named_buffers():
        originals[("b", name)] = buf.data
        buf.data = buf.data.to(device, non_blocking=True)
    return originals


def _restore_cpu(module, originals):
    """GPU 복사본을 버리고 CPU 원본을 복원."""
    for name, param in module.named_parameters():
        param.data = originals[("p", name)]
    for name, buf in module.named_buffers():
        buf.data = originals[("b", name)]


def teacher_forward_offload(model, input_ids, attention_mask=None, device="cuda"):
    """
    Layer-by-layer CPU offload forward.

    Args:
        model: CPU pinned memory에 있는 Gemma4ForCausalLM.
        input_ids: (batch, seq_len) 텐서. 어느 device든 가능.
        attention_mask: (batch, seq_len) 텐서 또는 None.
        device: forward를 실행할 GPU device.

    Returns:
        logits: (batch, seq_len, vocab_size) GPU 텐서.
    """
    # multimodal: model.model.language_model, text-only: model.model
    if hasattr(model.model, 'language_model'):
        inner = model.model.language_model
    else:
        inner = model.model
    config = inner.config

    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # --- Embedding ---
    w_embed = inner.embed_tokens.weight.data.to(device, non_blocking=True)
    embed_scale = None
    for _, b in inner.embed_tokens.named_buffers():
        embed_scale = b.data.to(device, non_blocking=True)
        break
    torch.cuda.synchronize()

    hidden = F.embedding(input_ids, w_embed, padding_idx=inner.embed_tokens.padding_idx)
    if embed_scale is not None:
        hidden = hidden * embed_scale.to(hidden.dtype)
    del w_embed, embed_scale

    # --- Rotary embeddings ---
    orig = _swap_to_gpu(inner.rotary_emb, device)
    torch.cuda.synchronize()

    position_ids = torch.arange(hidden.shape[1], device=device).unsqueeze(0)
    position_embeddings = {}
    for layer_type in inner.unique_layer_types:
        position_embeddings[layer_type] = inner.rotary_emb(hidden, position_ids, layer_type)

    _restore_cpu(inner.rotary_emb, orig)
    del orig

    # --- Causal masks ---
    mask_kwargs = {
        "config": config,
        "inputs_embeds": hidden,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }

    # --- Decoder layers ---
    num_layers = config.num_hidden_layers
    for i, layer in enumerate(inner.layers[:num_layers]):
        layer_type = config.layer_types[i]

        orig = _swap_to_gpu(layer, device)
        torch.cuda.synchronize()

        hidden = layer(
            hidden,
            position_embeddings=position_embeddings[layer_type],
            attention_mask=causal_mask_mapping[layer_type],
            position_ids=position_ids,
        )

        _restore_cpu(layer, orig)
        del orig

    # --- Final norm ---
    orig = _swap_to_gpu(inner.norm, device)
    torch.cuda.synchronize()
    hidden = inner.norm(hidden)
    _restore_cpu(inner.norm, orig)
    del orig

    # --- LM head ---
    orig = _swap_to_gpu(model.lm_head, device)
    torch.cuda.synchronize()
    logits = model.lm_head(hidden)
    _restore_cpu(model.lm_head, orig)
    del orig, hidden

    # --- Softcapping ---
    tc = getattr(model.config, "text_config", model.config)
    softcap = getattr(tc, "final_logit_softcapping", None)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return logits
