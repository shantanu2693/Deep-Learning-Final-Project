# Final project — MIDI language models


## Prerequisites

- Python 3.10+ recommended.
- Install dependencies (from the repository root):

  ```bash
  pip install -r requirements.txt
  ```

- Download **[MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro)** and place it so the layout matches:

  ```text
  maestro-v3.0.0/
    maestro-v3.0.0.csv
    2004/
    ... (year folders with .midi files)
  ```

  Training scripts expect `--maestro-root` to point at that folder (see examples below).

## Neural n-gram (MLP) training

Run from the **repository root** so paths to MAESTRO resolve correctly. The script lives in `ngram_and_rnn/`

```bash
python ngram_and_rnn/train_autoregressive_midi.py \
  --maestro-root maestro-v3.0.0 \
  --epochs 50 \
  --loss-plot out/mlp_loss.png \
  --loss-csv out/mlp_loss.csv \
  --checkpoint out/token_mlp.pt
```


## Vanilla RNN training

```bash
python ngram_and_rnn/train_autoregressive_rnn_midi.py \
  --maestro-root maestro-v3.0.0 \
  --epochs 50 \
  --loss-plot out/rnn_loss.png \
  --loss-csv out/rnn_loss.csv \
  --checkpoint out/token_rnn.pt
```

## Generate MIDI from a checkpoint

`generate_midi_mlp.py` lives at the repository root but imports model code from `ngram_and_rnn/`. Set **`PYTHONPATH`** so those imports resolve:

```bash
PYTHONPATH=ngram_and_rnn python generate_midi_mlp.py \
  --checkpoint out/token_mlp.pt \
  --out generated.mid \
  --steps 500 \
  --temperature 1.0 \
  --maestro-root maestro-v3.0.0 \
  --prime-midi maestro-v3.0.0/2004/some_file.midi
```

