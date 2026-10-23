#!/usr/bin/env python3
"""
segment_audio.py  -  Step 2 of the "LLMs for Sentiment Analysis in Trading"
pipeline: cut each press conference into overlapping "audio bars".

As in the course:
  * each bar is 60 seconds of audio,
  * a new bar starts every 30 seconds (so consecutive bars overlap by 30 s),
  * each bar's timestamp is delayed by 15 seconds after the bar ends
    (embargo, so a signal is never timed before the audio could be processed).

Bar k covers [30k, 30k + 60) seconds of the recording and becomes usable at
30k + 60 + 15 seconds. Only complete 60-second bars are kept.

Input   audio/metadata.csv and the MP3s from collect_fed_audio.py
Output  bars/<video_id>/bar_0000.wav ...   16 kHz mono WAV, Whisper's input format
        bars/bars.csv                      one row per bar, with clock times (ET)

Clock times assume the recording starts at the scheduled 14:30 ET. Check the real
start of each video (e.g. when the Chair begins speaking) and set it in the
start_offset_sec column of audio/metadata.csv if it differs: a positive value
means the recording starts that many seconds after 14:30.

Usage
  python segment_audio.py
  python segment_audio.py --bar 60 --step 30 --embargo 15 --audio audio --out bars
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import wave
from datetime import datetime, timedelta
from pathlib import Path

FIELDS = ["video_id", "meeting_date", "bar", "start_sec", "end_sec", "available_sec",
          "start_et", "end_et", "available_et", "file"]


def ffmpeg_cut(src: Path, dst: Path, start: float, length: float) -> None:
    """Write to a temporary name and rename when complete, so an interrupted
    run never leaves a half-written bar that a rerun would skip."""
    tmp = dst.with_name(dst.stem + ".part.wav")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    tmp.replace(dst)


def complete(path: Path, length: float) -> bool:
    """True if a bar file exists with the full expected length."""
    try:
        with wave.open(str(path)) as w:
            return abs(w.getnframes() / w.getframerate() - length) < 0.05
    except Exception:
        return False


def duration(src: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(src)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Cut press-conference audio into overlapping bars")
    ap.add_argument("--audio", default="audio", help="folder with MP3s and metadata.csv")
    ap.add_argument("--out", default="bars", help="output folder")
    ap.add_argument("--bar", type=float, default=60.0, help="bar length, seconds (default 60)")
    ap.add_argument("--step", type=float, default=30.0, help="time between bars, seconds (default 30)")
    ap.add_argument("--embargo", type=float, default=15.0, help="delay after bar end, seconds (default 15)")
    ap.add_argument("--force", action="store_true", help="re-cut bars that already exist")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} was not found. Install ffmpeg (winget install --id Gyan.FFmpeg -e) "
                     "and open a new terminal.")
    audio_dir, out_dir = Path(args.audio), Path(args.out)
    meta_path = audio_dir / "metadata.csv"
    if not meta_path.exists():
        sys.exit(f"{meta_path} not found. Run collect_fed_audio.py first.")
    with meta_path.open(newline="", encoding="utf-8") as f:
        meetings = list(csv.DictReader(f))

    rows, n_new = [], 0
    for m in sorted(meetings, key=lambda r: r["meeting_date"]):
        src = audio_dir / m["file"]
        if not src.exists():
            print(f"Missing {src}, skipped")
            continue
        total = duration(src)
        offset = float(m.get("start_offset_sec") or 0)
        t0 = datetime.fromisoformat(f"{m['meeting_date']} {m['scheduled_start_et']}") \
            + timedelta(seconds=offset)
        vdir = out_dir / m["video_id"]
        vdir.mkdir(parents=True, exist_ok=True)
        n_bars = int((total - args.bar) // args.step) + 1 if total >= args.bar else 0
        print(f"{m['meeting_date']}  {m['video_id']}  {total / 60:5.1f} min -> {n_bars} bars", flush=True)
        for k in range(n_bars):
            start = k * args.step
            end = start + args.bar
            avail = end + args.embargo
            dst = vdir / f"bar_{k:04d}.wav"
            if args.force or not complete(dst, args.bar):
                ffmpeg_cut(src, dst, start, args.bar)
                n_new += 1
            clock = lambda s: (t0 + timedelta(seconds=s)).strftime("%Y-%m-%d %H:%M:%S")
            rows.append({"video_id": m["video_id"], "meeting_date": m["meeting_date"], "bar": k,
                         "start_sec": start, "end_sec": end, "available_sec": avail,
                         "start_et": clock(start), "end_et": clock(end),
                         "available_et": clock(avail),
                         "file": f"{m['video_id']}/bar_{k:04d}.wav"})

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "bars.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} bars ({n_new} newly cut) in {out_dir.resolve()}")
    print(f"Index: {out_dir / 'bars.csv'}")


if __name__ == "__main__":
    main()
