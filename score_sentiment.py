#!/usr/bin/env python3
"""
score_sentiment.py  -  Step 4 of the "LLMs for Sentiment Analysis in Trading"
pipeline: score each audio bar's transcript with FinBERT.

FinBERT (Araci, 2019; model "ProsusAI/finbert") is BERT fine-tuned on financial
text. For each bar it returns probabilities for positive, negative and neutral.
The sentiment score is

    score = P(positive) - P(negative)        in [-1, +1]

so +1 is clearly positive, -1 clearly negative and 0 neutral or mixed. Step 5
sets buy and sell thresholds from the quantiles of this score's distribution.

Input   bars/bar_transcripts.csv       (transcribe_audio.py)
Output  bars/bar_sentiment.csv         bar_transcripts.csv + p_positive,
                                       p_negative, p_neutral, label, score
        a summary per press conference on screen

Setup (PowerShell)
  python -m pip install torch transformers

The first run downloads FinBERT (about 440 MB).

Usage
  python score_sentiment.py
  python score_sentiment.py --model yiyanghkust/finbert-tone    # another FinBERT
  python score_sentiment.py --batch 8                           # if memory is low
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

DEFAULT_MODEL = "ProsusAI/finbert"


def load_classifier(model: str):
    try:
        from transformers import pipeline
    except ImportError:
        sys.exit("transformers is not installed: python -m pip install torch transformers")
    try:
        return pipeline("text-classification", model=model, tokenizer=model, device=-1)
    except Exception as e:
        sys.exit(f"Could not load {model}: {str(e)[:300]}")


def score_texts(clf, texts: list[str], batch: int) -> list[dict]:
    """Probability of each label for every text, plus score = P(pos) - P(neg)."""
    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        res = clf(chunk, batch_size=batch, truncation=True, max_length=512, top_k=None)
        for labels in res:
            p = {d["label"].lower(): float(d["score"]) for d in labels}
            pos, neg, neu = p.get("positive", 0.0), p.get("negative", 0.0), p.get("neutral", 0.0)
            out.append({"p_positive": round(pos, 4), "p_negative": round(neg, 4),
                        "p_neutral": round(neu, 4),
                        "label": max(p, key=p.get),
                        "score": round(pos - neg, 4)})
        print(f"\r  {min(i + batch, len(texts))}/{len(texts)} bars scored "
              f"({time.time() - t0:.0f} s)", end="", flush=True)
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="FinBERT sentiment score for every audio bar")
    ap.add_argument("--bars", default="bars")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    src = Path(args.bars) / "bar_transcripts.csv"
    if not src.exists():
        sys.exit(f"{src} not found. Run transcribe_audio.py first.")
    with src.open(newline="", encoding="utf-8") as f:
        bars = list(csv.DictReader(f))
    if not bars:
        sys.exit(f"{src} has no bars.")

    print(f"Loading {args.model} ...")
    clf = load_classifier(args.model)
    texts = [b["text"] if b["text"].strip() else "." for b in bars]
    scores = score_texts(clf, texts, args.batch)

    rows = []
    for b, s in zip(bars, scores):
        if not b["text"].strip():                       # no speech: no score
            s = {k: "" for k in s}
        rows.append({**b, **s, "model": args.model})
    dst = Path(args.bars) / "bar_sentiment.csv"
    with dst.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary per press conference
    by_meeting = defaultdict(list)
    for r in rows:
        if r["score"] != "":
            by_meeting[r["meeting_date"]].append(r)
    print(f"\n{'meeting':<12}{'bars':>6}{'mean':>8}{'min':>8}{'max':>8}"
          f"{'pos':>6}{'neg':>6}{'neu':>6}")
    for d in sorted(by_meeting):
        rs = by_meeting[d]
        sc = [float(r["score"]) for r in rs]
        n = {lab: sum(r["label"] == lab for r in rs) for lab in ("positive", "negative", "neutral")}
        print(f"{d:<12}{len(rs):>6}{statistics.mean(sc):>8.3f}{min(sc):>8.3f}{max(sc):>8.3f}"
              f"{n['positive']:>6}{n['negative']:>6}{n['neutral']:>6}")
    allsc = sorted(float(r["score"]) for rs in by_meeting.values() for r in rs)
    q = lambda p: allsc[min(len(allsc) - 1, int(p * len(allsc)))]
    print(f"\nAll bars: score quantiles  10% {q(0.10):.3f} | 50% {q(0.50):.3f} | 90% {q(0.90):.3f}")
    print(f"Written to {dst}")


if __name__ == "__main__":
    main()
