#!/usr/bin/env python3
"""
transcribe_audio.py  -  Step 3 of the "LLMs for Sentiment Analysis in Trading"
pipeline: speech-to-text with Whisper.

Faster variant of the course's step: instead of transcribing each 60-second bar
separately, it transcribes each full press conference once, with a timestamp for
every word, and then splits the words into the same bars as segment_audio.py.
The bars are identical; accuracy is better because Whisper hears whole sentences,
and it is about twice as fast because the overlapping bars aren't transcribed twice.

A word belongs to a bar if the middle of the word falls inside the bar's
60-second window. Neighbouring bars overlap by 30 seconds, so most words appear
in two bars, exactly as when the bars are transcribed separately.

Input   audio/metadata.csv and the MP3s (collect_fed_audio.py)
        bars/bars.csv                    (segment_audio.py)
Output  transcripts/<video_id>.json     full transcript: segments and words with times
        transcripts/<video_id>.txt      readable transcript
        bars/bar_transcripts.csv        bars.csv + text and word count for every bar

Setup (PowerShell)
  python -m pip install faster-whisper
  (fallback: python -m pip install openai-whisper ; needs ffmpeg on PATH)

The first run downloads the Whisper model (about 150 MB for "base").

Usage
  python transcribe_audio.py                       # Whisper base, as in the course
  python transcribe_audio.py --model small         # larger model, slower, more accurate
  python transcribe_audio.py --engine openai       # use the openai-whisper package
  python transcribe_audio.py --rebuild-bars        # only redo the bar split
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


# --------------------------------------------------------------------------
# Transcription engines - both return [{"start","end","text","words":[...]}]
# --------------------------------------------------------------------------
def load_engine(name: str, model: str, threads: int):
    if name in ("auto", "faster"):
        try:
            from faster_whisper import WhisperModel
            m = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
            return "faster-whisper", lambda path: _faster(m, path)
        except ImportError:
            if name == "faster":
                sys.exit("faster-whisper is not installed: python -m pip install faster-whisper")
    try:
        import whisper
    except ImportError:
        sys.exit("No Whisper package found. Install one of:\n"
                 "  python -m pip install faster-whisper      (recommended)\n"
                 "  python -m pip install openai-whisper")
    m = whisper.load_model(model, device="cpu")
    return "openai-whisper", lambda path: _openai(m, path)


def _faster(model, path: Path) -> list[dict]:
    segments, info = model.transcribe(str(path), language="en", word_timestamps=True,
                                      beam_size=5, condition_on_previous_text=True)
    out = []
    for s in segments:                       # generator: transcribes as we go
        out.append({"start": s.start, "end": s.end, "text": s.text.strip(),
                    "words": [{"word": w.word, "start": w.start, "end": w.end,
                               "probability": w.probability} for w in (s.words or [])]})
        print(f"\r    {s.end / 60:5.1f} min transcribed", end="", flush=True)
    print()
    return out


def _openai(model, path: Path) -> list[dict]:
    r = model.transcribe(str(path), language="en", word_timestamps=True,
                         fp16=False, verbose=False)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip(),
             "words": [{"word": w["word"], "start": w["start"], "end": w["end"],
                        "probability": w.get("probability")} for w in s.get("words", [])]}
            for s in r["segments"]]


# --------------------------------------------------------------------------
# Split words into bars
# --------------------------------------------------------------------------
def words_of(transcript: dict) -> list[dict]:
    return [w for s in transcript["segments"] for w in s["words"]]


def bar_text(words: list[dict], start: float, end: float) -> tuple[str, int]:
    """Words whose midpoint lies in [start, end), joined in order."""
    sel = [w for w in words if start <= (w["start"] + w["end"]) / 2 < end]
    text = "".join(w["word"] for w in sel).strip()
    return " ".join(text.split()), len(sel)


def build_bars(bars_csv: Path, tdir: Path, out_csv: Path) -> int:
    with bars_csv.open(newline="", encoding="utf-8") as f:
        bars = list(csv.DictReader(f))
    cache, rows, missing = {}, [], set()
    for b in bars:
        vid = b["video_id"]
        if vid not in cache:
            p = tdir / f"{vid}.json"
            cache[vid] = words_of(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None
        if cache[vid] is None:
            missing.add(vid)
            continue
        text, n = bar_text(cache[vid], float(b["start_sec"]), float(b["end_sec"]))
        rows.append({**b, "n_words": n, "text": text})
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if missing:
        print(f"Not yet transcribed (bars left out): {', '.join(sorted(missing))}")
    return len(rows)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe press conferences and split into bars")
    ap.add_argument("--audio", default="audio")
    ap.add_argument("--bars", default="bars")
    ap.add_argument("--out", default="transcripts")
    ap.add_argument("--model", default="base", help="Whisper model size (default: base, as in the course)")
    ap.add_argument("--engine", default="auto", choices=["auto", "faster", "openai"])
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (default: all)")
    ap.add_argument("--force", action="store_true", help="re-transcribe existing transcripts")
    ap.add_argument("--rebuild-bars", action="store_true",
                    help="skip transcription; only rebuild bar_transcripts.csv")
    args = ap.parse_args()

    audio_dir, bars_dir, tdir = Path(args.audio), Path(args.bars), Path(args.out)
    meta_path, bars_csv = audio_dir / "metadata.csv", bars_dir / "bars.csv"
    for p, step in ((meta_path, "collect_fed_audio.py"), (bars_csv, "segment_audio.py")):
        if not p.exists():
            sys.exit(f"{p} not found. Run {step} first.")
    tdir.mkdir(parents=True, exist_ok=True)

    if not args.rebuild_bars:
        with meta_path.open(newline="", encoding="utf-8") as f:
            meetings = sorted(csv.DictReader(f), key=lambda r: r["meeting_date"])
        todo = [m for m in meetings
                if args.force or not (tdir / f"{m['video_id']}.json").exists()]
        print(f"{len(meetings)} press conferences, {len(todo)} to transcribe.")
        if todo:
            engine_name, transcribe = load_engine(args.engine, args.model, args.threads)
            print(f"Engine: {engine_name}, model: {args.model}\n")
        for m in todo:
            src = audio_dir / m["file"]
            if not src.exists():
                print(f"Missing {src}, skipped")
                continue
            print(f"{m['meeting_date']}  {m['video_id']}")
            t = time.time()
            segs = transcribe(src)
            data = {"video_id": m["video_id"], "meeting_date": m["meeting_date"],
                    "model": args.model, "engine": engine_name, "segments": segs}
            tmp = tdir / f"{m['video_id']}.json.part"
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(tdir / f"{m['video_id']}.json")       # only complete files count
            (tdir / f"{m['video_id']}.txt").write_text(
                "\n".join(f"[{s['start'] / 60:5.1f} min] {s['text']}" for s in segs),
                encoding="utf-8")
            n_words = sum(len(s["words"]) for s in segs)
            print(f"    {n_words:,} words in {(time.time() - t) / 60:.1f} min")

    n = build_bars(bars_csv, tdir, bars_dir / "bar_transcripts.csv")
    print(f"\n{n} bars with text written to {bars_dir / 'bar_transcripts.csv'}")
    print(f"Full transcripts in {tdir.resolve()}")


if __name__ == "__main__":
    main()
