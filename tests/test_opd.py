"""Tests for the OPD utilities and one full OPD step on a tiny Qwen3-VL (CPU)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from vlm_opd.opd_utils import (
    full_attention_mask,
    generation_mask,
    logits_positions,
    masked_mean,
    micro_batches,
    per_token_kl,
    slice_vision,
)

EOS, PAD = 9, 0


def test_generation_mask_stops_at_first_eos_and_ignores_padding():
    P = 2
    seqs = torch.tensor([
        [5, 5,  7, 8, EOS, PAD, PAD],   # ended at position 2 of the generation
        [5, 5,  7, 8, 6, 6, 6],         # never ended (hit max_new_tokens)
        [5, 5,  EOS, PAD, PAD, PAD, PAD],  # ended immediately
        [5, 5,  7, EOS, EOS, 8, PAD],   # second EOS and later tokens are excluded
    ])
    m = generation_mask(seqs, P, EOS, PAD)
    assert m.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
        [True, False, False, False, False],
        [True, True, False, False, False],
    ]
    attn = full_attention_mask(torch.tensor([[0, 1]] * 4), m)
    assert attn.shape == (4, 7) and attn[0].tolist() == [0, 1, 1, 1, 1, 0, 0]


def test_logits_positions_are_shifted_by_one():
    assert logits_positions(prompt_len=10, gen_len=3).tolist() == [9, 10, 11]


def test_per_token_kl_properties():
    torch.manual_seed(0)
    s = torch.randn(2, 5, 50)
    t = torch.randn(2, 5, 50)
    assert torch.allclose(per_token_kl(s, s.clone()), torch.zeros(2, 5), atol=1e-5)
    rev, fwd = per_token_kl(s, t, "reverse", chunk=2), per_token_kl(s, t, "forward", chunk=2)
    assert (rev >= 0).all() and (fwd >= 0).all()
    assert rev.shape == (2, 5) and not torch.allclose(rev, fwd)
    # gradient flows to the student only
    s_req = s.clone().requires_grad_(True)
    t_req = t.clone().requires_grad_(True)
    per_token_kl(s_req, t_req).sum().backward()
    assert s_req.grad is not None and s_req.grad.abs().sum() > 0
    assert t_req.grad is None
    with pytest.raises(ValueError):
        per_token_kl(s, t, "sideways")


def test_masked_mean_and_micro_batches():
    v = torch.tensor([[1.0, 2.0, 3.0]])
    assert masked_mean(v, torch.tensor([[True, False, True]])).item() == 2.0
    assert masked_mean(v, torch.zeros(1, 3, dtype=torch.bool)).item() == 0.0
    assert micro_batches(8, 4) == [(0, 4), (4, 8)]
    assert micro_batches(5, 2) == [(0, 2), (2, 4), (4, 5)]


def test_slice_vision_selects_patch_rows_per_image():
    grid = torch.tensor([[1, 2, 2], [1, 4, 2], [1, 2, 4]])  # 4, 8, 8 patches
    pv = torch.arange(20).unsqueeze(1).float()
    rows, g = slice_vision(pv, grid, 1, 3)
    assert g.tolist() == [[1, 4, 2], [1, 2, 4]]
    assert rows[:, 0].tolist() == list(range(4, 20))


@pytest.fixture(scope="module")
def tiny_setup():
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from vlm_opd.common import STUDENT_MODEL, image_pixel_bounds
    from vlm_opd.modeling import apply_lora, freeze_all

    try:
        processor = AutoProcessor.from_pretrained(STUDENT_MODEL, **image_pixel_bounds())
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"processor unavailable: {type(e).__name__}")
    tok = processor.tokenizer
    text_cfg = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "vocab_size": len(tok), "max_position_embeddings": 4096, "head_dim": 16,
                "rope_theta": 10000.0}
    vision_cfg = {"depth": 2, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4, "out_hidden_size": 64,
                  "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3,
                  "deepstack_visual_indexes": [0, 1]}
    cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg, image_token_id=processor.image_token_id,
                        video_token_id=tok.convert_tokens_to_ids("<|video_pad|>"),
                        vision_start_token_id=tok.convert_tokens_to_ids("<|vision_start|>"),
                        vision_end_token_id=tok.convert_tokens_to_ids("<|vision_end|>"))
    torch.manual_seed(0)
    teacher = Qwen3VLForConditionalGeneration(cfg).eval()
    freeze_all(teacher)
    torch.manual_seed(1)
    student = Qwen3VLForConditionalGeneration(cfg)
    freeze_all(student)
    student = apply_lora(student, r=4, alpha=8, dropout=0.0)
    rows = [
        {"image": Image.new("RGB", (768, 512)), "question": "What is the value for 2020?", "answer": "45"},
        {"image": Image.new("RGB", (512, 640)), "question": "Which country is highest?", "answer": "Germany"},
        {"image": Image.new("RGB", (640, 640)), "question": "Sum of the two bars?", "answer": "12"},
    ]
    return processor, teacher, student, rows


def test_opd_step_runs_and_updates_only_lora(tiny_setup):
    from vlm_opd.opd_trainer import OPDConfig, opd_step

    processor, teacher, student, rows = tiny_setup
    cfg = OPDConfig(data_repo="x", out_dir="y", batch_size=3, micro_batch=2, max_new_tokens=6, kl_direction="reverse")
    metrics = opd_step(student, teacher, processor, rows, cfg, step_seed=7)

    assert metrics["kl"] >= 0 and metrics["n_gen_tokens"] > 0
    assert 0 <= metrics["gen_len_mean"] <= 6 and metrics["gen_len_max"] <= 6
    assert 0.0 <= metrics["eos_rate"] <= 1.0 and 0.0 <= metrics["format_rate"] <= 1.0
    assert metrics["rollout_acc"] is not None
    for name, p in student.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name
    assert all(p.grad is None for p in teacher.parameters())
    student.zero_grad(set_to_none=True)


def test_opd_step_forward_kl_direction(tiny_setup):
    from vlm_opd.opd_trainer import OPDConfig, opd_step

    processor, teacher, student, rows = tiny_setup
    cfg = OPDConfig(data_repo="x", out_dir="y", batch_size=3, micro_batch=3, max_new_tokens=4, kl_direction="forward")
    metrics = opd_step(student, teacher, processor, rows[:2] + rows[:1], cfg, step_seed=3)
    assert metrics["kl"] >= 0
    student.zero_grad(set_to_none=True)
