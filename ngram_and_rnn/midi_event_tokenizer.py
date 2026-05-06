"""
Discrete event tokens for polyphonic piano MIDI (MAESTRO-friendly).

Stream format (one token per step, autoregressive target = next token):
  - WAIT_k: log-quantized time since the *previous event* (64 bins, ~0.1 ms .. 30 s).
  - ON_p:  note-on,  pitch p in 0..127.
  - OFF_p: note-off, pitch p in 0..127.

Special ids (see VOCAB):
  PAD, SOS, EOS — used for batching / sequence start / end of piece.

This is a small teaching vocabulary (not REMI/Compound Word). Velocity and
pedal are omitted to keep the vocab compact.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pretty_midi

N_WAIT = 64

# Special tokens
PAD = 0
SOS = 1
EOS = 2
# WAIT: 3 .. 2+N_WAIT
# ON:   3+N_WAIT .. 2+N_WAIT+128
# OFF:  after ON block
_BASE_WAIT = 3
_BASE_ON = _BASE_WAIT + N_WAIT
_BASE_OFF = _BASE_ON + 128
VOCAB_SIZE = _BASE_OFF + 128


def token_id_kind(tid: int) -> Literal["pad", "sos", "eos", "wait", "on", "off", "unknown"]:
    if tid == PAD:
        return "pad"
    if tid == SOS:
        return "sos"
    if tid == EOS:
        return "eos"
    if _BASE_WAIT <= tid < _BASE_ON:
        return "wait"
    if _BASE_ON <= tid < _BASE_OFF:
        return "on"
    if _BASE_OFF <= tid < VOCAB_SIZE:
        return "off"
    return "unknown"


def wait_bin_to_dt_sec(bin_idx: int) -> float:
    """Representative delta (seconds) for a WAIT bin (inverse of quantization, bin center in log space)."""
    b = int(np.clip(bin_idx, 0, N_WAIT - 1))
    log_lo = math.log(1e-4) + (b / N_WAIT) * (math.log(30.0) - math.log(1e-4))
    log_hi = math.log(1e-4) + ((b + 1) / N_WAIT) * (math.log(30.0) - math.log(1e-4))
    return float(math.exp(0.5 * (log_lo + log_hi)))


def tokens_to_pretty_midi(
    tokens: list[int],
    default_velocity: int = 80,
) -> pretty_midi.PrettyMIDI:
    """
    Decode a token id stream (WAIT/ON/OFF/EOS) into a single piano PrettyMIDI.
    Unmatched note-ons are closed at the end of the timeline; stray OFFs are ignored.
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=False, name="Piano")
    pm.instruments.append(inst)

    t = 0.0
    active: dict[int, float] = {}  # pitch -> start time

    for tid in tokens:
        kind = token_id_kind(tid)
        if kind == "wait":
            b = tid - _BASE_WAIT
            t += wait_bin_to_dt_sec(b)
        elif kind == "on":
            p = tid - _BASE_ON
            if p not in active:
                active[p] = t
        elif kind == "off":
            p = tid - _BASE_OFF
            if p in active:
                st = active.pop(p)
                if t >= st:
                    inst.notes.append(
                        pretty_midi.Note(
                            velocity=int(np.clip(default_velocity, 1, 127)),
                            pitch=int(p),
                            start=float(st),
                            end=float(t),
                        )
                    )
        elif kind == "eos":
            break
        # pad, sos, unknown: skip time

    end_t = t
    for p, st in list(active.items()):
        inst.notes.append(
            pretty_midi.Note(
                velocity=int(np.clip(default_velocity, 1, 127)),
                pitch=int(p),
                start=float(st),
                end=float(max(st + 0.05, end_t + 0.2)),
            )
        )

    inst.notes.sort(key=lambda n: n.start)
    return pm


def _dt_to_wait_bin(dt_sec: float) -> int:
    dt = float(np.clip(dt_sec, 1e-4, 30.0))
    z = (math.log(dt) - math.log(1e-4)) / (math.log(30.0) - math.log(1e-4))
    b = int(z * N_WAIT)
    return min(max(b, 0), N_WAIT - 1)


def _wait_token(bin_idx: int) -> int:
    return _BASE_WAIT + int(bin_idx)


def _on_token(pitch: int) -> int:
    return _BASE_ON + int(np.clip(pitch, 0, 127))


def _off_token(pitch: int) -> int:
    return _BASE_OFF + int(np.clip(pitch, 0, 127))


def tokenize_midi(
    midi_path: Path,
    append_eos: bool = True,
) -> list[int] | None:
    """
    Convert one MIDI file to a list of token ids (no SOS; EOS optional).
    Returns None if unreadable or empty.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return None

    events: list[tuple[float, Literal["off", "on"], int]] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            p = int(n.pitch)
            events.append((float(n.end), "off", p))
            events.append((float(n.start), "on", p))

    if not events:
        return None

    # Same time: note-offs before note-ons (common convention).
    events.sort(key=lambda x: (x[0], 0 if x[1] == "off" else 1, x[2]))

    out: list[int] = []
    prev_t = 0.0
    for t, kind, pitch in events:
        dt = max(t - prev_t, 0.0)
        out.append(_wait_token(_dt_to_wait_bin(dt)))
        out.append(_on_token(pitch) if kind == "on" else _off_token(pitch))
        prev_t = t

    if append_eos:
        out.append(EOS)
    return out
