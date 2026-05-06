"""
Autoregressive next-event-token training on MAESTRO MIDI.

Token vocabulary: see midi_event_tokenizer.py (WAIT / ON / OFF + SOS/EOS/PAD).

Model: fixed-context **MLP** — embed the last K tokens, flatten to a vector,
run a feedforward stack, and predict **one** next token (neural n-gram LM).
This is not an RNN: there is no hidden state carried across time inside the net;
context is exactly those K token ids.

Usage:
  python train_autoregressive_midi.py --maestro-root maestro-v3.0.0 --epochs 50 \\
      --max-train-files 80 --context-len 32 --batch-size 32 \\
      --loss-plot out/mlp_loss.png --loss-csv out/mlp_loss.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from loss_curve import perplexity_from_mean_ce, save_train_val_figure
from midi_event_tokenizer import PAD, VOCAB_SIZE, tokenize_midi


def load_train_midi_paths(maestro_root: Path) -> list[str]:
    csv_path = maestro_root / "maestro-v3.0.0.csv"
    paths: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                paths.append(row["midi_filename"])
    return paths


def load_val_midi_paths(maestro_root: Path, max_files: int) -> list[str]:
    csv_path = maestro_root / "maestro-v3.0.0.csv"
    paths: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "validation":
                paths.append(row["midi_filename"])
    return paths[:max_files]


class MidiWindowDataset(Dataset):
    """Random (context -> next token) pairs from tokenized performances."""

    def __init__(
        self,
        maestro_root: Path,
        midi_relpaths: list[str],
        context_len: int,
        max_files: int | None,
        seed: int,
    ) -> None:
        super().__init__()
        rng = random.Random(seed)
        if max_files is not None and max_files < len(midi_relpaths):
            midi_relpaths = rng.sample(midi_relpaths, max_files)

        self.context_len = context_len
        self.corpus: list[np.ndarray] = []
        self.maestro_root = maestro_root

        for rel in midi_relpaths:
            toks = tokenize_midi(maestro_root / rel, append_eos=True)
            if toks is None or len(toks) < context_len + 1:
                continue
            self.corpus.append(np.asarray(toks, dtype=np.int64))

        if not self.corpus:
            raise RuntimeError("No tokenized MIDI sequences long enough; check paths.")

        self._lengths = np.array([len(a) for a in self.corpus], dtype=np.int64)
        self._weights = self._lengths.astype(np.float64) - context_len
        self._weights = np.maximum(self._weights, 1.0)
        self._epoch_steps = min(8000, int(self._weights.sum()))

    def __len__(self) -> int:
        return self._epoch_steps

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        _ = idx
        j = int(np.random.choice(len(self.corpus), p=self._weights / self._weights.sum()))
        arr = self.corpus[j]
        max_start = len(arr) - self.context_len - 1
        start = int(np.random.randint(0, max_start + 1))
        chunk = arr[start : start + self.context_len + 1]
        context = chunk[:-1]
        target = chunk[-1]
        return torch.from_numpy(context.copy()), torch.tensor(target, dtype=torch.long)


class TokenMLP(nn.Module):
    """Embedding -> flatten -> MLP -> logits over vocabulary (single next token)."""

    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int,
        hidden: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        in_dim = context_len * d_model
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(1, num_hidden_layers)):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.ReLU())
            dim = hidden
        layers.append(nn.Linear(dim, vocab_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, K) -> logits (B, vocab_size)
        e = self.emb(context)
        flat = e.reshape(context.size(0), -1)
        return self.mlp(flat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro-root", type=Path, default=Path("maestro-v3.0.0"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--context-len",
        type=int,
        default=32,
        help="How many previous tokens the MLP sees (fixed window).",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Deprecated alias for --context-len (if set, overrides --context-len).",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument(
        "--mlp-layers",
        type=int,
        default=2,
        help="Number of hidden Linear+ReLU blocks before the final projection.",
    )
    ap.add_argument("--max-train-files", type=int, default=120)
    ap.add_argument("--max-val-files", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="If set, save model + hyperparameters here after training (for generate_midi_mlp.py).",
    )
    ap.add_argument(
        "--loss-csv",
        type=Path,
        default=None,
        help="If set, write epoch, train_loss, val_loss, train_ppl, val_ppl after training.",
    )
    ap.add_argument(
        "--loss-plot",
        type=Path,
        default=None,
        help="If set, save loss + perplexity side-by-side (PNG) after training.",
    )
    args = ap.parse_args()

    context_len = args.seq_len if args.seq_len is not None else args.context_len

    maestro_root = args.maestro_root.resolve()
    if not maestro_root.is_dir():
        print(f"Missing directory: {maestro_root}", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_paths = load_train_midi_paths(maestro_root)
    val_paths = load_val_midi_paths(maestro_root, args.max_val_files)

    train_ds = MidiWindowDataset(
        maestro_root,
        train_paths,
        context_len=context_len,
        max_files=args.max_train_files,
        seed=args.seed,
    )
    val_ds = MidiWindowDataset(
        maestro_root,
        val_paths,
        context_len=context_len,
        max_files=None,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenMLP(
        vocab_size=VOCAB_SIZE,
        context_len=context_len,
        d_model=args.d_model,
        hidden=args.hidden,
        num_hidden_layers=args.mlp_layers,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    print(
        f"vocab_size={VOCAB_SIZE}  context_len={context_len}  "
        f"train_sequences={len(train_ds.corpus)}  val={len(val_ds.corpus)}"
    )

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        nb = 0
        for ctx, tgt in train_loader:
            ctx, tgt = ctx.to(device), tgt.to(device)
            opt.zero_grad()
            logits = model(ctx)
            loss = crit(logits, tgt)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        tr = running / max(nb, 1)

        model.eval()
        vrunning = 0.0
        vnb = 0
        with torch.no_grad():
            for ctx, tgt in val_loader:
                ctx, tgt = ctx.to(device), tgt.to(device)
                logits = model(ctx)
                loss = crit(logits, tgt)
                vrunning += loss.item()
                vnb += 1
        va = vrunning / max(vnb, 1)
        train_losses.append(tr)
        val_losses.append(va)
        tr_ppl = perplexity_from_mean_ce(tr)
        va_ppl = perplexity_from_mean_ce(va)
        print(
            f"epoch {epoch:02d}  train_loss={tr:.4f}  val_loss={va:.4f}  "
            f"train_ppl={tr_ppl:.2f}  val_ppl={va_ppl:.2f}"
        )

    print("Done. (Loss is next-token cross-entropy; perplexity = exp(loss) in nats.)")

    if args.loss_csv is not None:
        with open(args.loss_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "train_ppl", "val_ppl"])
            for ep, tr_l, va_l in zip(range(1, len(train_losses) + 1), train_losses, val_losses):
                w.writerow(
                    [
                        ep,
                        f"{tr_l:.6f}",
                        f"{va_l:.6f}",
                        f"{perplexity_from_mean_ce(tr_l):.6f}",
                        f"{perplexity_from_mean_ce(va_l):.6f}",
                    ]
                )
        print(f"Wrote loss CSV to {args.loss_csv.resolve()}")

    if args.loss_plot is not None:
        save_train_val_figure(
            args.loss_plot,
            train_losses,
            val_losses,
            title="Token MLP — next-token cross-entropy",
        )
        print(f"Wrote loss plot to {args.loss_plot.resolve()}")

    if args.checkpoint is not None:
        payload = {
            "model_kind": "mlp",
            "model_state": model.state_dict(),
            "vocab_size": VOCAB_SIZE,
            "context_len": context_len,
            "d_model": args.d_model,
            "hidden": args.hidden,
            "mlp_layers": args.mlp_layers,
        }
        torch.save(payload, args.checkpoint)
        print(f"Wrote checkpoint to {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
