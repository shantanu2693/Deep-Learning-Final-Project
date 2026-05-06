"""
Autoregressive next-event-token training on MAESTRO MIDI (vanilla RNN only).

Same data setup as train_autoregressive_midi.py (random K-token context → predict
the next token), but the core is **torch.nn.RNN** only (no GRU, no LSTM).

Model: Embedding -> nn.RNN -> last timestep hidden -> Linear(vocab).

Usage:
  python train_autoregressive_rnn_midi.py --maestro-root maestro-v3.0.0 \\
      --checkpoint out/token_rnn.pt --epochs 50 --context-len 32 \\
      --loss-plot out/rnn_loss.png --loss-csv out/rnn_loss.csv
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
from torch.utils.data import DataLoader

from loss_curve import perplexity_from_mean_ce, save_train_val_figure
from midi_event_tokenizer import PAD, VOCAB_SIZE
from train_autoregressive_midi import (
    MidiWindowDataset,
    load_train_midi_paths,
    load_val_midi_paths,
)


class TokenVanillaRNN(nn.Module):
    """Embedding -> nn.RNN (tanh) -> last timestep -> logits. Many-to-one next token."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        hidden: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, K) -> logits (B, vocab_size)
        e = self.emb(context)
        out, _ = self.rnn(e)
        last = out[:, -1, :]
        return self.head(last)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro-root", type=Path, default=Path("maestro-v3.0.0"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--context-len", type=int, default=32)
    ap.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Deprecated alias for --context-len (if set, overrides --context-len).",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--rnn-layers", type=int, default=1, help="Stacked nn.RNN layers.")
    ap.add_argument("--max-train-files", type=int, default=120)
    ap.add_argument("--max-val-files", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="If set, save model + hyperparameters (for generate_midi_mlp.py).",
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
    model = TokenVanillaRNN(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        hidden=args.hidden,
        num_layers=args.rnn_layers,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    print(
        f"nn.RNN  vocab_size={VOCAB_SIZE}  context_len={context_len}  "
        f"rnn_layers={args.rnn_layers}  train_sequences={len(train_ds.corpus)}  "
        f"val={len(val_ds.corpus)}"
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
            title="Vanilla RNN — next-token cross-entropy",
        )
        print(f"Wrote loss plot to {args.loss_plot.resolve()}")

    if args.checkpoint is not None:
        payload = {
            "model_kind": "vanilla_rnn",
            "model_state": model.state_dict(),
            "vocab_size": VOCAB_SIZE,
            "context_len": context_len,
            "d_model": args.d_model,
            "hidden": args.hidden,
            "rnn_layers": args.rnn_layers,
        }
        torch.save(payload, args.checkpoint)
        print(f"Wrote checkpoint to {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
