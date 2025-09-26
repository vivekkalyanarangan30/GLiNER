#!/usr/bin/env python
"""Compare GLiNER NER predictions with and without inference packing."""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
import time
from typing import List, Sequence

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gliner import GLiNER, InferencePackingConfig


DEFAULT_TEXTS: List[str] = [
    "OpenAI launched GPT-4o in San Francisco, while Sam Altman discussed future plans on CNBC.",
    "NASA announced that the Artemis II mission will send astronauts around the Moon in 2025.",
    "Amazon acquired Whole Foods for $13.7 billion and expanded grocery delivery across the United States.",
    "Taylor Swift kicked off her Eras Tour in Glendale before headlining shows across Europe in 2024.",
]

DEFAULT_LABELS: List[str] = [
    "Person",
    "Organization",
    "Location",
    "Event",
    "Date",
    "Money",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single GLiNER NER batch with and without inference packing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="urchade/gliner_small-v2.1",
        help="Model name or local path to load with GLiNER.from_pretrained",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (e.g. 'cpu', 'cuda', 'cuda:0')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size to use for the GLiNER dataloader",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold passed to GLiNER.run",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override maximum packed sequence length (defaults to model.config.max_len)",
    )
    parser.add_argument(
        "--streams-per-batch",
        type=int,
        default=1,
        help="Number of packed streams to generate per batch",
    )
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _format_prediction(example_id: int, text: str, entities: Sequence[dict]) -> str:
    header = f"Example {example_id}"
    body_lines = [textwrap.fill(text, width=100, initial_indent="  ", subsequent_indent="  ")]
    if not entities:
        body_lines.append("  (no entities above threshold)")
    else:
        for ent in entities:
            snippet = ent["text"].replace("\n", " ")
            label = ent["label"]
            score = ent.get("score", 0.0)
            start = ent.get("start")
            end = ent.get("end")
            body_lines.append(
                f"  - '{snippet}' ({label}, score={score:.2f}, span={start}:{end})"
            )
    return "\n".join([header] + body_lines)


def _display_predictions(texts: Sequence[str], predictions: Sequence[Sequence[dict]]) -> None:
    print("NER predictions:")
    for idx, (text, ents) in enumerate(zip(texts, predictions), start=1):
        print(_format_prediction(idx, text, ents))
        print()


def _run_once(
    model: GLiNER,
    *,
    texts: Sequence[str],
    labels: Sequence[str],
    batch_size: int,
    threshold: float,
    device: torch.device,
    packing_config: InferencePackingConfig | None,
) -> tuple[List[List[dict]], float]:
    _sync_if_cuda(device)
    start = time.perf_counter()
    predictions = model.run(
        list(texts),
        list(labels),
        batch_size=batch_size,
        threshold=threshold,
        packing_config=packing_config,
    )
    _sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    # The model returns a tuple when num_gen_sequences > 1; enforce list typing for consistency.
    return [list(example) for example in predictions], elapsed


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.streams_per_batch < 1:
        raise ValueError("--streams-per-batch must be >= 1")

    texts: List[str] = list(DEFAULT_TEXTS)
    labels: List[str] = list(DEFAULT_LABELS)

    device = torch.device(args.device)
    try:
        model = GLiNER.from_pretrained(args.model, map_location=str(device))
    except Exception as exc:  # pragma: no cover - network / I/O failures
        raise SystemExit(
            "Failed to load GLiNER model. Use --model with a local path or ensure network access."
        ) from exc
    model.to(device)
    model.eval()

    tokenizer = model.data_processor.transformer_tokenizer
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    if sep_token_id is None:
        sep_token_id = getattr(tokenizer, "eos_token_id", None)
    if sep_token_id is None:
        sep_token_id = getattr(tokenizer, "pad_token_id", None)

    max_length = args.max_length or getattr(model.config, "max_len", None)
    if max_length is None:
        raise ValueError("Unable to infer max sequence length for packing; please pass --max-length")

    packing_config = InferencePackingConfig(
        max_length=int(max_length),
        sep_token_id=sep_token_id,
        streams_per_batch=args.streams_per_batch,
    )

    print("Running baseline (no packing)...")
    baseline_preds, baseline_time = _run_once(
        model,
        texts=texts,
        labels=labels,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=device,
        packing_config=None,
    )
    _display_predictions(texts, baseline_preds)

    print("Running with inference packing...")
    packed_preds, packed_time = _run_once(
        model,
        texts=texts,
        labels=labels,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=device,
        packing_config=packing_config,
    )

    identical = baseline_preds == packed_preds

    print("Timing summary:")
    print(f"  Without packing: {baseline_time:.3f} s")
    print(f"  With packing   : {packed_time:.3f} s")
    if packed_time > 0:
        print(f"  Speedup        : {baseline_time / packed_time:.2f}x")
    print(f"Predictions identical: {identical}")


if __name__ == "__main__":
    main()
