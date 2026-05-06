Offline / air-gapped use of loubb/aria-medium-base
----------------------------------------------------

This training pipeline needs the model config, tokenizer, and weight files on disk
if your network cannot reach https://huggingface.co (common on locked-down networks).

On a machine with normal internet access, run one of:

  hf download loubb/aria-medium-base --local-dir ./loubb--aria-medium-base

Copy the resulting folder next to this file, e.g.:

  aria-maestro-finetune/models/loubb--aria-medium-base/

Then use that path everywhere the README mentions --model_name_or_path, for example:

  python scripts/prepare_tokens.py \
    --model_name_or_path models/loubb--aria-medium-base \
    --midi_root data/maestro-v3.0.0 \
    --out_dir data/packed

  python scripts/train.py \
    --model_name_or_path models/loubb--aria-medium-base \
    --packed_dir data/packed \
    --output_dir runs/aria-maestro

Large weight files use Git LFS on the Hub; the `hf download` command pulls them for you.
