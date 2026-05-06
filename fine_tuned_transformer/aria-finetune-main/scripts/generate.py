from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to finetuned checkpoint dir")
    ap.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="",
        help="Optional: tokenizer path (use base model dir if checkpoint didn't save a tokenizer).",
    )
    ap.add_argument("--prompt_midi", type=str, required=True)
    ap.add_argument("--out_midi", type=str, required=True)
    ap.add_argument("--prompt_tokens", type=int, default=512)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.97)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).expanduser().resolve()
    prompt_midi = Path(args.prompt_midi).expanduser().resolve()
    out_midi = Path(args.out_midi).expanduser().resolve()
    out_midi.parent.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(str(ckpt), trust_remote_code=True)
    tok_src = args.tokenizer_name_or_path.strip() or str(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    prompt = tokenizer.encode_from_file(str(prompt_midi), return_tensors="pt")
    input_ids = prompt.input_ids[..., : args.prompt_tokens].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_length=args.max_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            use_cache=True,
        )

    midi_dict = tokenizer.decode(out[0].tolist())
    midi_dict.to_midi().save(str(out_midi))
    print(f"Wrote {out_midi}")


if __name__ == "__main__":
    main()

