"""Train vs. validation metrics: loss and perplexity (exp of mean CE in nats)."""

from __future__ import annotations

import math
from pathlib import Path


def perplexity_from_mean_ce(mean_ce_nats: float) -> float:
    """PyTorch CrossEntropyLoss uses natural log; perplexity = exp(mean NLL)."""
    return math.exp(mean_ce_nats)


def save_train_val_figure(
    out_path: Path,
    train_losses: list[float],
    val_losses: list[float],
    *,
    title: str = "Next-token cross-entropy",
) -> None:
    import matplotlib.pyplot as plt

    if len(train_losses) != len(val_losses) or not train_losses:
        raise ValueError("train_losses and val_losses must be non-empty and the same length.")

    epochs = list(range(1, len(train_losses) + 1))
    train_ppl = [perplexity_from_mean_ce(x) for x in train_losses]
    val_ppl = [perplexity_from_mean_ce(x) for x in val_losses]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    fig.suptitle(title)

    ax0.plot(epochs, train_losses, label="train", marker="o")
    ax0.plot(epochs, val_losses, label="val", marker="o")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss (nats)")
    ax0.set_title("Loss")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1.plot(epochs, train_ppl, label="train", marker="o")
    ax1.plot(epochs, val_ppl, label="val", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Perplexity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
