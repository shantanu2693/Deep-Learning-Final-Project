from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from aria_maestro_finetune.packed import PackedBlocksDataset, simple_collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed_dir", type=str, required=True, help="Directory containing train.npy / val.npy")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument(
        "--model_name_or_path",
        type=str,
        default="loubb/aria-medium-base",
        help="Hub id or local directory with config + tokenizer (e.g. after `hf download ... --local-dir ...`).",
    )
    ap.add_argument("--block_size", type=int, default=2048)

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_train_epochs", type=float, default=1)
    ap.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If set to a positive value, caps training at this many steps (overrides num_train_epochs).",
    )
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--disable_checkpoints",
        action="store_true",
        help="Disable periodic checkpoint saving (recommended for Aria tokenizer).",
    )

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Use 0 on macOS / restricted environments (avoids torch multiprocessing shared-memory issues).",
    )
    args = ap.parse_args()

    packed_dir = Path(args.packed_dir).expanduser().resolve()
    train_path = packed_dir / "train.npy"
    val_path = packed_dir / "val.npy"
    if not train_path.exists():
        raise SystemExit(f"Missing {train_path}")
    if not val_path.exists():
        raise SystemExit(f"Missing {val_path}")

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_ds = PackedBlocksDataset(train_path)
    val_ds = PackedBlocksDataset(val_path)
    if train_ds.block_size != args.block_size:
        raise SystemExit(f"--block_size={args.block_size} but train.npy has block_size={train_ds.block_size}")
    if val_ds.block_size != args.block_size:
        raise SystemExit(f"--block_size={args.block_size} but val.npy has block_size={val_ds.block_size}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs: dict = dict(
        output_dir=str(out_dir),
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    if args.disable_checkpoints:
        train_kwargs["save_strategy"] = "no"
    if args.max_steps > 0:
        train_kwargs["max_steps"] = args.max_steps

    train_args = TrainingArguments(**train_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=simple_collate,
    )

    trainer.train()
    trainer.save_model(str(out_dir))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

