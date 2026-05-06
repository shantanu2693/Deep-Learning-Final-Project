from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def list_midi_files(root: Path) -> list[Path]:
    exts = {".mid", ".midi"}
    paths: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)


def _flatten_token_ids(input_ids) -> list[int]:
    """Coerce HF / Aria outputs to a flat list of Python ints (handles nested lists + tensors)."""
    if input_ids is None:
        return []
    if hasattr(input_ids, "detach"):
        input_ids = input_ids.detach().cpu().reshape(-1).tolist()
    if hasattr(input_ids, "tolist") and not isinstance(input_ids, (list, tuple)):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, int):
        return [input_ids]
    if isinstance(input_ids, (list, tuple)):
        out: list[int] = []
        for x in input_ids:
            out.extend(_flatten_token_ids(x))
        return out
    return [int(input_ids)]


def encode_midi(tokenizer, midi_path: Path) -> list[int]:
    # Aria tokenizer exposes encode_from_file via trust_remote_code.
    enc = tokenizer.encode_from_file(str(midi_path), return_tensors=None)
    input_ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    return _flatten_token_ids(input_ids)


def pack_blocks(seqs: list[list[int]], block_size: int) -> np.ndarray:
    flat: list[int] = []
    for s in seqs:
        if len(s) == 0:
            continue
        flat.extend(s)
    if len(flat) < block_size:
        return np.zeros((0, block_size), dtype=np.int32)
    n_blocks = len(flat) // block_size
    flat = flat[: n_blocks * block_size]
    arr = np.asarray(flat, dtype=np.int32).reshape(n_blocks, block_size)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_root", type=str, required=True, help="Directory containing MAESTRO .mid/.midi files")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--model_name_or_path",
        type=str,
        default="loubb/aria-medium-base",
        help="Hub id or local snapshot directory containing tokenizer files.",
    )
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max_files", type=int, default=0, help="Optional cap for quick tests (0 = no cap)")
    args = ap.parse_args()

    midi_root = Path(args.midi_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    midi_files = list_midi_files(midi_root)
    if args.max_files and args.max_files > 0:
        midi_files = midi_files[: args.max_files]
    if not midi_files:
        raise SystemExit(f"No MIDI files found under: {midi_root}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    random.shuffle(midi_files)
    n_val = max(1, int(math.floor(len(midi_files) * args.val_ratio)))
    val_files = set(midi_files[:n_val])
    train_files = [p for p in midi_files if p not in val_files]

    def encode_many(paths: list[Path]) -> list[list[int]]:
        seqs: list[list[int]] = []
        for p in tqdm(paths, desc="Encoding MIDI"):
            try:
                seqs.append(encode_midi(tokenizer, p))
            except Exception as e:  # noqa: BLE001
                print(f"[warn] failed to encode {p}: {e}")
        return seqs

    train_seqs = encode_many(train_files)
    val_seqs = encode_many(list(val_files))

    train_arr = pack_blocks(train_seqs, args.block_size)
    val_arr = pack_blocks(val_seqs, args.block_size)

    np.save(out_dir / "train.npy", train_arr)
    np.save(out_dir / "val.npy", val_arr)

    stats = {
        "model_name_or_path": args.model_name_or_path,
        "midi_root": str(midi_root),
        "block_size": args.block_size,
        "num_midi_files_total": len(midi_files),
        "num_midi_files_train": len(train_files),
        "num_midi_files_val": len(val_files),
        "num_blocks_train": int(train_arr.shape[0]),
        "num_blocks_val": int(val_arr.shape[0]),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"Wrote {out_dir/'train.npy'}: {train_arr.shape}")
    print(f"Wrote {out_dir/'val.npy'}: {val_arr.shape}")
    print(f"Wrote {out_dir/'stats.json'}")


if __name__ == "__main__":
    main()

