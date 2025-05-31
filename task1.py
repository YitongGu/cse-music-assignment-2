#!/usr/bin/env python3
"""
Task 1: Symbolic, un-conditioned music generation (MAESTRO v3.0.0)

This code implements symbolic music generation using a sequence of tokens.

It preprocesses MIDI files from the MAESTRO dataset by:
1. Quantizing note timings to a 16th-note grid (GRID = 1/4)
2. Converting each MIDI file into a sequence of tokens representing:
   - Time shifts (T tokens): Represent gaps between notes (up to MAX_TS=64 steps)
   - Pitch values (P tokens): Represent MIDI pitch values (0-127)
   - Duration values (D tokens): Represent note durations in grid steps

The encoding process:
- Quantizes all note timings and durations to the grid
- Sorts notes by time and pitch
- Creates a sequence of tokens in the form: [T{time_shift}, P{pitch}, D{duration}, ...]
- Handles large time shifts by breaking them into multiple tokens

This tokenization approach allows representing MIDI music as a sequence
that can be processed by neural language models for music generation.

"""
import argparse, json, math, os, random, sys, pickle
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pretty_midi as pm
import muspy             
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ───────────────────────────────────────── tokenisation ──────────────────────────────────────────
GRID = 1/4          # 16-note resolution
MAX_TS = 64         # max time shift token

def ticks_per_step(resolution: int) -> int:
    return int(round(resolution * GRID))


# encode midi to tokens
def encode_midi(midi: pm.PrettyMIDI):
    midi = muspy.from_pretty_midi(midi)
    step = ticks_per_step(midi.resolution) 
    # quantization to snap all times to grid
    for track in midi.tracks:
        for note in track.notes:
            note.time = int(round(note.time     / step) * step)
            note.duration = int(max(step, round(note.duration / step) * step))
    
    notes = sorted(
        (n for tr in midi.tracks for n in tr.notes),
        key=lambda n: (n.time, n.pitch)
    )
    seq, t_prev = [], 0
    for n in notes:
        # time-shift
        ts = (n.time - t_prev) // step
        while ts > 0:
            shift = min(ts, MAX_TS)
            seq.append(f"T{shift}")
            ts -= shift
        # note & duration
        dur = max(1, n.duration // step)
        seq.extend([f"P{n.pitch}", f"D{min(dur, MAX_TS)}"])
        t_prev = n.time
    return seq

# build vocabulary from sequences
def build_vocab(seqs):
    uniq = sorted({tok for s in seqs for tok in s})
    vocab = {tok:i+4 for i,tok in enumerate(uniq)}
    vocab.update({"<PAD>":0, "<BOS>":1, "<EOS>":2, "<UNK>":3}) # add special tokens
    return vocab


# ────────────────────────────────────────── dataset ─────────────────────────────────────────────
# dataset class
class MaestroDataset(Dataset):
    def __init__(self, root: Path, seq_len: int):
        cache_file = root / "encoded_cache.pkl"
        # load from cache if it exists
        if cache_file.exists():
            print("Loading from cache...", file=sys.stderr)
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                self.vocab = cache['vocab']
                self.inv_vocab = cache['inv_vocab']
                self.samples = cache['samples']
            print(f"Loaded {len(self.samples)} sequences from cache "
                  f"(vocab={len(self.vocab)})", file=sys.stderr)
        else:
            # encode from scratch
            print("Indexing MAESTRO...", file=sys.stderr)
            midi_files = list(root.rglob("*.mid*"))
            all_seqs = []
            for f in tqdm(midi_files):
                try:
                    seq = encode_midi(pm.PrettyMIDI(str(f)))
                    all_seqs.append(seq)
                except Exception as e:
                    print(f"Failed to encode {f}: {e}", file=sys.stderr)
                    continue
            self.vocab = build_vocab(all_seqs)
            self.inv_vocab = {v:k for k,v in self.vocab.items()}
            
            # numericalise & chunk
            self.samples = []
            for seq in all_seqs:
                ids = [self.vocab["<BOS>"]] + [self.vocab.get(t,3) for t in seq] + [self.vocab["<EOS>"]]
                for i in range(0, len(ids), seq_len):
                    chunk = ids[i:i+seq_len]
                    if len(chunk) < 8:
                        continue
                    chunk += [0] * (seq_len - len(chunk))
                    self.samples.append(torch.tensor(chunk, dtype=torch.long))
            
            print(f"Found {len(self.samples)} sequences "
                  f"(vocab={len(self.vocab)})", file=sys.stderr)
            
            # save to cache
            print("Saving to cache...", file=sys.stderr)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'vocab': self.vocab,
                    'inv_vocab': self.inv_vocab,
                    'samples': self.samples
                }, f)

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ─────────────────────────────────────────── model ──────────────────────────────────────────────
# model class
class MusicLSTM(nn.Module):
    def __init__(self, ntok, d_model=896, nlayers=3, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(ntok, d_model, padding_idx=0)
        self.lstm  = nn.LSTM(d_model, d_model, nlayers,
                             dropout=dropout, batch_first=True)
        self.head  = nn.Linear(d_model, ntok)

    def forward(self, x, h=None):
        z = self.embed(x)
        z, h = self.lstm(z, h)
        return self.head(z), h


# ───────────────────────────────────── training / evaluation ────────────────────────────────────
# run epoch
def run_epoch(model, dl, crit, opt=None, device="cpu"):
    total, n_tokens = 0.0, 0
    for x in tqdm(dl, leave=False):
        x = x.to(device)
        y = x[:,1:].contiguous()
        inp = x[:,:-1]
        logits, _ = model(inp)
        loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
        if opt:
            opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * y.numel()
        n_tokens += y.numel()
    return math.exp(total / n_tokens)  # perplexity


# ───────────────────────────────────────── generation ───────────────────────────────────────────
# sample from model
def sample(model, vocab, inv_vocab, max_len=1024,
           top_p=0.95, temperature=0.8, device="cpu"):
    model.eval()
    idx = torch.tensor([[vocab["<BOS>"]]], device=device)
    h = None; out = []
    # generate tokens
    with torch.no_grad():
        for _ in range(max_len):
            logits, h = model(idx, h)
            logits = logits[0, -1] / temperature
            probs  = torch.softmax(logits, -1)

            # nucleus filter
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_p, dim=-1)
            keep = cdf <= top_p
            keep[0] = True
            probs_masked = torch.zeros_like(probs).scatter_(
                0, sorted_idx[keep], sorted_p[keep])
            probs_masked /= probs_masked.sum()   # renormalise
            nxt = torch.multinomial(probs_masked, 1).item()
            if nxt == vocab["<EOS>"]:
                break
            out.append(nxt)
            idx = torch.tensor([[nxt]], device=device)
    return out

# convert tokens to midi
def tokens_to_midi(ids, inv_vocab, out_path, bpm: int = 120):
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)
    sec_per_step = (60 / bpm) * GRID
    time = 0.0
    i = 0
    # generate notes
    while i < len(ids):
        # advance time only on explicit T-tokens
        tok = inv_vocab[ids[i]]
        if tok.startswith("T"):
            time += int(tok[1:]) * sec_per_step
            i += 1
            continue
        # handle pitch & duration but DON'T advance `time`
        if tok.startswith("P"):
            pitch = int(tok[1:])
            dur = sec_per_step
            if i + 1 < len(ids) and inv_vocab[ids[i+1]].startswith("D"):
                dur = int(inv_vocab[ids[i+1]][1:]) * sec_per_step
                i += 1
            note = pm.Note(velocity=80, pitch=pitch,
                           start=time, end=time + dur)
            inst.notes.append(note)
        i += 1 # move to next token

    midi.instruments.append(inst)
    midi.write(str(out_path))


# ───────────────────────────────────────────── main ─────────────────────────────────────────────
# python task1.py   --midi_dir  music_data/maestro-v3.0.0   --out_dir   ./exp/task1   --n_epochs  10   --batch_size 8   --seq_len  1024   --generate_len 512
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--midi_dir", required=True)
    p.add_argument("--out_dir",  required=True)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--generate_len", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--clear_cache", action="store_true", help="Clear the encoded cache")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # clear cache if requested
    if args.clear_cache:
        cache_file = Path(args.midi_dir) / "encoded_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()
            print("Cache cleared.", file=sys.stderr)
    # load dataset
    ds   = MaestroDataset(Path(args.midi_dir), args.seq_len+1)
    train_len = int(0.9 * len(ds))
    val_len   = len(ds) - train_len
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, val_len])
    # create data loaders
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   args.batch_size, shuffle=False, drop_last=False)
    # create model
    model = MusicLSTM(len(ds.vocab)).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_epochs)
    crit  = nn.CrossEntropyLoss(ignore_index=0)

    best_pp = float("inf")
    for epoch in range(1, args.n_epochs+1):
        print(f"\nEpoch {epoch}")
        _ = run_epoch(model, train_dl, crit, opt, args.device)
        pp = run_epoch(model, val_dl,   crit,     device=args.device)
        print(f"validation perplexity: {pp:6.2f}")
        torch.save(model.state_dict(), out/"last.pt")
        if pp < best_pp:
            best_pp = pp
            torch.save(model.state_dict(), out/"best.pt")
        sched.step()

    # sample -- use best checkpoint
    model.load_state_dict(torch.load(out/"best.pt", map_location=args.device))
    ids = sample(model, ds.vocab, ds.inv_vocab,
                 max_len=args.generate_len, device=args.device)
    tokens_to_midi(ids, ds.inv_vocab, out/"sample.mid", bpm=120)
    print(f"Saved sample to {out/'sample.mid'}")

if __name__ == "__main__":
    main()

