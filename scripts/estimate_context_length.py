#!/usr/bin/env python3
"""Estimate context-length distribution for ShareGPT-style jsonl datasets.

Streams all ``*.jsonl`` under a directory (or a single file), measures each
sample's conversation character length, and optionally estimates token length
via a calibrated chars/token ratio or an exact HF tokenizer + chat template.

Examples:
    # Fast char + calibrated token estimate (default)
    python scripts/estimate_context_length.py --data data/data_jsonl

    # Exact tokens with GLM chat template (slower)
    python scripts/estimate_context_length.py \\
        --data data/data_jsonl \\
        --tokenizer /path/to/GLM-5.2 \\
        --exact --max-samples 20000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


def _iter_jsonl_files(data_path: Path) -> list[Path]:
    if data_path.is_file():
        return [data_path]
    files = sorted(data_path.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files under {data_path}")
    return files


def _flatten_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "value" in item:
                    parts.append(str(item["value"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _conversation_text(row: dict) -> str:
    """Concatenate conversation turns into a single text blob (for char estimate)."""
    conv = row.get("conversations") or row.get("messages") or []
    if not isinstance(conv, list):
        return ""
    chunks: list[str] = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from") or turn.get("role") or ""
        content = _flatten_content(turn.get("value", turn.get("content")))
        reasoning = turn.get("reasoning_content") or turn.get("thinking") or ""
        tool_calls = turn.get("tool_calls")
        pieces = [f"[{role}]"]
        if reasoning:
            pieces.append(str(reasoning))
        if content:
            pieces.append(content)
        if tool_calls:
            pieces.append(json.dumps(tool_calls, ensure_ascii=False))
        chunks.append("\n".join(pieces))
    return "\n".join(chunks)


def _normalize_messages(row: dict) -> list[dict]:
    """Normalize to role/content messages for chat-template tokenization."""
    conv = row.get("conversations") or row.get("messages") or []
    out: list[dict] = []
    if not isinstance(conv, list):
        return out
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from") or turn.get("role") or ""
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        elif role not in ("system", "tool"):
            continue
        msg: dict = {
            "role": role,
            "content": _flatten_content(turn.get("value", turn.get("content"))),
        }
        reasoning = turn.get("reasoning_content") or turn.get("thinking")
        if reasoning:
            msg["reasoning_content"] = reasoning
        if turn.get("tool_calls"):
            msg["tool_calls"] = turn["tool_calls"]
        if turn.get("tool_call_id"):
            msg["tool_call_id"] = turn["tool_call_id"]
        out.append(msg)
    return out


def _percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _histogram(lengths: list[int], edges: list[int]) -> list[tuple[str, int, float]]:
    """Return (bucket_label, count, fraction) for half-open edges."""
    counts = [0] * (len(edges) + 1)
    for x in lengths:
        placed = False
        for i, e in enumerate(edges):
            if x < e:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    n = max(len(lengths), 1)
    labels = []
    prev = 0
    for e in edges:
        labels.append(f"[{prev}, {e})")
        prev = e
    labels.append(f"[{edges[-1]}, +inf)")
    return [(lab, c, c / n) for lab, c in zip(labels, counts)]


def _print_distribution(title: str, lengths: list[int], edges: list[int]) -> None:
    if not lengths:
        print(f"\n=== {title}: empty ===")
        return
    s = sorted(lengths)
    n = len(s)
    mean = sum(s) / n
    print(f"\n=== {title} ===")
    print(f"n={n:,}")
    print(
        f"min={s[0]:,}  p10={_percentile(s, 10):.0f}  p25={_percentile(s, 25):.0f}  "
        f"p50={_percentile(s, 50):.0f}  p75={_percentile(s, 75):.0f}  "
        f"p90={_percentile(s, 90):.0f}  p95={_percentile(s, 95):.0f}  "
        f"p99={_percentile(s, 99):.0f}  max={s[-1]:,}"
    )
    print(f"mean={mean:.1f}")
    print("histogram:")
    for lab, c, frac in _histogram(s, edges):
        bar = "#" * int(frac * 40)
        print(f"  {lab:>16}  {c:>10,}  {frac:6.2%}  {bar}")


class _RawTokenizer:
    """Thin wrapper around ``tokenizers.Tokenizer`` (no chat template)."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(path)

    def encode_text(self, text: str) -> int:
        return len(self._tok.encode(text).ids)


def _load_tokenizer(model_path: str, trust_remote_code: bool, prefer_raw: bool = True):
    """Load tokenizer for length estimation.

    Prefer raw ``tokenizer.json`` via the ``tokenizers`` library: it is enough for
    length estimates and avoids heavy / fragile AutoTokenizer loads on some FS.
    Pass ``prefer_raw=False`` (``--hf-tokenizer``) to force HuggingFace loading.
    """
    model_dir = Path(model_path)
    tok_json = (
        model_dir
        if model_dir.is_file() and model_dir.suffix == ".json"
        else model_dir / "tokenizer.json"
    )

    if prefer_raw and tok_json.exists():
        print(f"Using raw tokenizer.json: {tok_json}", flush=True)
        return _RawTokenizer(str(tok_json))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    jinja = model_dir / "chat_template.jinja"
    if not getattr(tok, "chat_template", None) and jinja.exists():
        tok.chat_template = jinja.read_text(encoding="utf-8")
    return tok


def _token_len(tokenizer, messages: list[dict], fallback_text: str = "") -> int | None:
    if isinstance(tokenizer, _RawTokenizer):
        text = fallback_text or "\n".join(
            f"{m.get('role', '')}: {_flatten_content(m.get('content'))}"
            for m in messages
        )
        try:
            return tokenizer.encode_text(text)
        except Exception:
            return None

    if not messages:
        return 0
    try:
        if getattr(tokenizer, "chat_template", None):
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            return len(ids)
        text = "\n".join(
            f"{m.get('role', '')}: {_flatten_content(m.get('content'))}"
            for m in messages
        )
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=str,
        default="data/data_jsonl",
        help="jsonl file or directory (default: data/data_jsonl)",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF tokenizer / model path for token estimates",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code when loading HF tokenizer",
    )
    p.add_argument(
        "--hf-tokenizer",
        action="store_true",
        help="Force HuggingFace AutoTokenizer instead of raw tokenizer.json",
    )
    p.add_argument(
        "--exact",
        action="store_true",
        help="Tokenize every sample (slow). Default: calibrate chars/token then estimate.",
    )
    p.add_argument(
        "--calibrate-samples",
        type=int,
        default=2000,
        help="Samples used to calibrate chars/token ratio when not --exact (default: 2000)",
    )
    p.add_argument(
        "--chars-per-token",
        type=float,
        default=None,
        help="Fixed chars/token ratio; skips calibration if set",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after N samples (for quick smoke tests)",
    )
    p.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="Histogram upper edges",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=50000,
        help="Print progress every N samples",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    files = _iter_jsonl_files(data_path)
    print(f"Found {len(files)} jsonl file(s) under {data_path}")

    tokenizer = None
    if args.tokenizer or args.exact:
        model_path = args.tokenizer
        if not model_path:
            print("--exact requires --tokenizer", file=sys.stderr)
            return 2
        print(f"Loading tokenizer from {model_path} ...")
        tokenizer = _load_tokenizer(
            model_path,
            args.trust_remote_code,
            prefer_raw=not args.hf_tokenizer,
        )
        print("Tokenizer ready.")

    char_lens: list[int] = []
    token_lens: list[int] = []
    n_turns: list[int] = []
    sources: Counter[str] = Counter()
    n_bad = 0
    n_seen = 0

    # Calibration buffers (chars, tokens) for ratio estimate
    calib_chars: list[int] = []
    calib_tokens: list[int] = []
    need_calib = (
        tokenizer is not None
        and not args.exact
        and args.chars_per_token is None
    )

    for fi, path in enumerate(files):
        print(f"[{fi + 1}/{len(files)}] scanning {path.name} ...", flush=True)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if args.max_samples is not None and n_seen >= args.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue

                n_seen += 1
                text = _conversation_text(row)
                clen = len(text)
                char_lens.append(clen)

                conv = row.get("conversations") or row.get("messages") or []
                n_turns.append(len(conv) if isinstance(conv, list) else 0)
                src = row.get("source")
                if isinstance(src, str):
                    sources[src] += 1

                if tokenizer is not None:
                    msgs = _normalize_messages(row)
                    if args.exact:
                        tlen = _token_len(tokenizer, msgs, fallback_text=text)
                        if tlen is not None:
                            token_lens.append(tlen)
                        else:
                            n_bad += 1
                    elif need_calib and len(calib_tokens) < args.calibrate_samples:
                        tlen = _token_len(tokenizer, msgs, fallback_text=text)
                        if tlen is not None and tlen > 0 and clen > 0:
                            calib_chars.append(clen)
                            calib_tokens.append(tlen)

                if args.progress_every and n_seen % args.progress_every == 0:
                    print(f"  ... {n_seen:,} samples", flush=True)

        if args.max_samples is not None and n_seen >= args.max_samples:
            break

    print(f"\nDone scanning. samples={n_seen:,} bad/skipped_lines={n_bad:,}")
    if sources:
        print("top sources:")
        for name, cnt in sources.most_common(10):
            print(f"  {name}: {cnt:,}")

    edges = list(args.buckets)
    _print_distribution("character length (conversation text)", char_lens, edges)
    _print_distribution("num conversation turns", n_turns, [2, 4, 6, 8, 12, 20, 40])

    # Token estimates
    if args.exact and token_lens:
        _print_distribution("exact token length (chat template)", token_lens, edges)
    else:
        ratio = args.chars_per_token
        if ratio is None and calib_chars:
            ratio = sum(calib_chars) / sum(calib_tokens)
            print(
                f"\nCalibrated chars/token = {ratio:.3f} "
                f"on {len(calib_tokens):,} samples "
                f"(mean chars={sum(calib_chars)/len(calib_chars):.1f}, "
                f"mean tokens={sum(calib_tokens)/len(calib_tokens):.1f})"
            )
            _print_distribution(
                "calibration exact token length",
                calib_tokens,
                edges,
            )
        if ratio is None:
            ratio = 3.0
            print(
                f"\nNo tokenizer calibration; using default chars/token={ratio:.1f}"
            )
        est = [max(1, int(round(c / ratio))) for c in char_lens]
        _print_distribution(
            f"estimated token length (chars / {ratio:.3f})",
            est,
            edges,
        )

        # Fractions relative to common train seq lengths
        print("\n=== fraction exceeding common seq_length budgets (estimated tokens) ===")
        for lim in (1024, 2048, 4096, 8192, 16384):
            frac = sum(1 for x in est if x > lim) / max(len(est), 1)
            print(f"  > {lim:>5}: {frac:6.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
