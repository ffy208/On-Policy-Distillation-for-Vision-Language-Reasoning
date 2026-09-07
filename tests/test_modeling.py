"""Tests for LoRA targeting and freezing on a tiny randomly initialised Qwen3-VL."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from vlm_opd.modeling import (
    LORA_TARGET_REGEX,
    apply_lora,
    freeze_all,
    trainable_summary,
)


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    text_cfg = {
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": 1024, "max_position_embeddings": 512, "head_dim": 16,
        "rope_theta": 10000.0,
    }
    vision_cfg = {
        "depth": 2, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4, "out_hidden_size": 64,
        "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3, "deepstack_visual_indexes": [0, 1],
    }
    cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg, image_token_id=1000, video_token_id=1001,
                        vision_start_token_id=1002, vision_end_token_id=1003)
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(cfg)


def test_module_names_match_expectations(tiny_model):
    names = [n for n, _ in tiny_model.named_modules()]
    assert any("language_model" in n and n.endswith("q_proj") for n in names)
    assert any("visual" in n for n in names)


def test_lora_targets_language_model_only(tiny_model):
    freeze_all(tiny_model)
    peft_model = apply_lora(tiny_model, r=4, alpha=8, dropout=0.0)
    lora_params = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert lora_params, "LoRA added no trainable parameters"
    assert all("lora_" in n for n in lora_params)
    assert all("language_model" in n for n in lora_params)
    assert not any("visual" in n for n in lora_params)
    summary = trainable_summary(peft_model)
    assert summary["vision_trainable"] == []
    assert 0 < summary["trainable"] < summary["total"]
    # every decoder projection got an adapter: 2 layers x 7 projections x (A, B)
    assert len(lora_params) == 2 * 7 * 2


def test_regex_excludes_vision_projections():
    import re

    pat = re.compile(LORA_TARGET_REGEX)
    assert pat.match("model.language_model.layers.0.self_attn.q_proj")
    assert pat.match("base_model.model.model.language_model.layers.3.mlp.down_proj")
    assert not pat.match("model.visual.blocks.0.attn.qkv")
    assert not pat.match("model.visual.blocks.0.attn.proj")
    assert not pat.match("model.visual.merger.linear_fc1")
    assert not pat.match("lm_head")


def test_sft_batch_forward_backward_on_tiny_model():
    """Real processor batch -> tiny Qwen3-VL with LoRA: loss is finite and only LoRA gets gradients."""
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from vlm_opd.collate import SFTCollator
    from vlm_opd.common import STUDENT_MODEL, image_pixel_bounds

    try:
        processor = AutoProcessor.from_pretrained(STUDENT_MODEL, **image_pixel_bounds())
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"processor unavailable: {type(e).__name__}")

    tok = processor.tokenizer
    text_cfg = {
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "vocab_size": len(tok), "max_position_embeddings": 4096, "head_dim": 16,
        "rope_theta": 10000.0,
    }
    vision_cfg = {
        "depth": 2, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4, "out_hidden_size": 64,
        "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3,
        "deepstack_visual_indexes": [0, 1],
    }
    cfg = Qwen3VLConfig(
        text_config=text_cfg, vision_config=vision_cfg,
        image_token_id=processor.image_token_id, video_token_id=tok.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tok.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tok.convert_tokens_to_ids("<|vision_end|>"),
    )
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(cfg)
    freeze_all(model)
    model = apply_lora(model, r=4, alpha=8, dropout=0.0)

    rows = [
        {"image": Image.new("RGB", (768, 512)), "question": "Q1?", "response": "Answer: 1"},
        {"image": Image.new("RGB", (512, 640)), "question": "Longer Q2?", "response": "Reasoning.\nAnswer: 2"},
    ]
    batch = SFTCollator(processor)(rows)
    out = model(**batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"
        else:
            assert p.grad is None
