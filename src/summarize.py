"""Draft an executive summary and publish it only if the fact-check passes."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

from src.factcheck import drop_unverified_sentences, fact_check


def canonical_summary(facts: dict) -> str:
    """Summary whose numbers are formatted straight from the computed tables."""
    sample = facts["sample"]
    retention = facts["retention"]
    field = facts["field_play"]
    propensity = facts["propensity"]
    forecast = facts["forecast"]
    segment = facts["segmentation"]
    return (
        f"This simulated panel covers {int(sample['n_accounts']):,} accounts "
        f"across {int(sample['n_months'])} months. "
        f"Twelve-month net revenue retention is {retention['nrr_12m']:.3f}. "
        f"Mean feature adoption is {retention['mean_adoption_pct']:.1f} percent, "
        f"and mean seat penetration is {retention['mean_penetration_pct']:.1f} percent. "
        f"The staggered field play estimate is {field['estimate']:.3f}, "
        f"which is {field['relative_error_pct']:.1f} percent away from the simulated "
        f"truth of {field['truth']:.3f}. "
        f"A naive before/after comparison is {field['naive_relative_error_pct']:.1f} "
        f"percent away from that truth. "
        f"LightGBM scores expansion propensity at {propensity['lgbm_auroc']:.3f} AUROC "
        f"on held-out months, compared with a logistic baseline of "
        f"{propensity['logistic_auroc']:.3f}. "
        f"The three-month revenue forecast has MAPE of {forecast['gbm_mape_pct']:.1f} "
        f"percent, against {forecast['seasonal_naive_mape_pct']:.1f} percent for the "
        f"seasonal naive forecast. "
        f"Segmentation silhouette is {segment['silhouette']:.3f}. "
        "Prioritize accounts the field has not yet worked when product usage is already "
        "high, and keep the never-treated accounts as the control group for the next readout."
    )


def _prompt(canonical: str) -> str:
    return (
        "Rewrite the executive summary below for a go-to-market leadership review. "
        "Keep every number exactly as written. Do not add any new number. "
        "Keep the recommendation.\n\n"
        f"{canonical}"
    )


def _repair_prompt(canonical: str) -> str:
    return (
        "Your previous draft added or changed a number. "
        "Copy the summary below and change only the wording around the numbers. "
        "Every numeral must stay identical.\n\n"
        f"{canonical}"
    )


def publish_summary(
    facts: dict,
    generate: Callable[[str], str],
    attempts: int = 3,
) -> dict:
    """Ask the generator for a draft. Publish it only when every number checks out.

    If a draft fails, sentences with unverified numbers are removed. If that
    still fails, the table-backed summary is published and the rejection is
    recorded. The published text is checked again from the string that will
    be written to disk.
    """
    canonical = canonical_summary(facts)
    canonical_check = fact_check(canonical, facts)
    if not canonical_check["passed"]:
        raise RuntimeError(
            "table-backed summary failed the fact check: "
            + ", ".join(canonical_check["unmatched"])
        )
    drafts: list[str] = []
    published = canonical
    source = "gated_fallback"
    for attempt in range(attempts):
        prompt = _prompt(canonical) if attempt == 0 else _repair_prompt(canonical)
        draft = generate(prompt).strip()
        drafts.append(draft)
        checked = fact_check(draft, facts)
        if checked["passed"]:
            published = draft
            source = "llm"
            break
        filtered = drop_unverified_sentences(draft, facts)
        filtered_check = fact_check(filtered, facts)
        if filtered_check["passed"]:
            published = filtered
            source = "llm_filtered"
            break
    final = fact_check(published, facts)
    if not final["passed"]:
        raise RuntimeError("published summary failed the fact check")
    rejected = 0 if source == "llm" else len(drafts)
    if source == "llm_filtered":
        rejected = len(drafts)
    return {
        "text": published,
        "source": source,
        "drafts": drafts,
        "match_rate": final["match_rate"],
        "n_numbers": final["n_numbers"],
        "drafts_rejected": rejected if source != "llm" else 0,
        "canonical": canonical,
    }


def generate_text(
    prompt: str,
    model_dir: Path,
    device: str,
    max_new_tokens: int,
    seed: int,
) -> str:
    """Run SmolLM2 locally. Weights must already be in ``model_dir``.

    CPU inference tries bfloat16 first to stay inside the memory budget, then
    float32 if that kernel is unavailable.
    """
    import torch

    os_offline()
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    dtypes = [torch.bfloat16, torch.float32] if device == "cpu" else [torch.float32]
    last_error: Exception | None = None
    for dtype in dtypes:
        try:
            return _generate_once(prompt, model_dir, device, dtype, max_new_tokens, seed)
        except Exception as exc:
            last_error = exc
            gc.collect()
    raise RuntimeError("local model generation failed") from last_error


def os_offline() -> None:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _generate_once(prompt, model_dir, device, dtype, max_new_tokens, seed) -> str:
    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(model_dir, dtype)
    model.to(device)
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
    except TypeError:
        encoded = {
            "input_ids": tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        }
    if not hasattr(encoded, "items"):
        encoded = {"input_ids": encoded}
    encoded = {key: value.to(device) for key, value in encoded.items() if hasattr(value, "to")}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = int(encoded["input_ids"].shape[1])
    text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    del output, encoded, model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return text


def _load_model(model_dir: Path, dtype):
    from transformers import AutoModelForCausalLM

    kwargs = {"local_files_only": True, "low_cpu_mem_usage": True}
    try:
        return AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, **kwargs)
