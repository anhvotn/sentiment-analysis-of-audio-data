#!/usr/bin/env python3
"""
collect_fed_audio.py  -  Step 1 of the "LLMs for Sentiment Analysis in Trading"
pipeline: collect audio of Federal Reserve (FOMC) press conferences from YouTube.

Pipeline (from the course):
  1. Collect audio from YouTube                    <- this script
  2. Segment audio into 60-second "audio bars" every 30 seconds (ffmpeg),
     timestamps delayed 15 seconds (embargo against look-ahead bias)
  3. Transcribe each bar with Whisper (base model)
  4. Score each transcript with FinBERT
  5. Buy/sell when the score crosses quantile thresholds; trade SPY on 30-second bars

What it does
  * Finds FOMC press conference videos on the Federal Reserve's YouTube channel
    (or takes URLs you give it), newest first.
  * Downloads the audio only and saves it as MP3 at 160 kbps, as in the course.
  * Writes audio/metadata.csv: meeting date, title, URL, duration, file and the
    scheduled start time (14:30 ET), which step 5 needs to line audio up with
    SPY prices.

The course uses pytube. pytube is no longer maintained and often fails with
current YouTube, so this script uses yt-dlp, which does the same job.

Setup (Anaconda Prompt, Windows)
  pip install yt-dlp
  conda install -c conda-forge ffmpeg        (or: winget install ffmpeg)

Usage
  python collect_fed_audio.py                          # 5 most recent press conferences
  python collect_fed_audio.py --count 10
  python collect_fed_audio.py --since 2022-01-01 --count 50
  python collect_fed_audio.py --list                   # show what would be downloaded
  python collect_fed_audio.py --title "press conference"   # looser title match
  python collect_fed_audio.py --urls https://www.youtube.com/watch?v=XXXX ...
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

FED_CHANNEL = "https://www.youtube.com/@federalreserve/videos"
TITLE_PATTERN = re.compile(r"FOMC\s+press\s+conference", re.IGNORECASE)
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
PRESS_CONF_START_ET = "14:30"      # scheduled start; check against the video
FIELDS = ["meeting_date", "title", "url", "video_id", "upload_date",
          "duration_sec", "file", "bitrate_kbps", "scheduled_start_et"]


def require_tools() -> None:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        sys.exit("yt-dlp is not installed. Run:  pip install yt-dlp")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found. Run:  conda install -c conda-forge ffmpeg  "
                 "(or: winget install ffmpeg), then open a new prompt.")


def meeting_date(title: str, upload_date: str | None) -> date | None:
    """Meeting date from a title like 'FOMC Press Conference, June 18, 2025';
    falls back to the upload date (YYYYMMDD)."""
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", title)
    if m and m.group(1).lower() in MONTHS:
        try:
            return date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
        except ValueError:
            pass
    if upload_date and len(upload_date) == 8:
        return datetime.strptime(upload_date, "%Y%m%d").date()
    return None


def find_press_conferences(channel: str, count: int, since: date | None,
                           scan: int, pattern: re.Pattern = TITLE_PATTERN) -> list[dict]:
    """Newest-first list of press conference videos on the channel."""
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist", "playlistend": scan}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(channel, download=False)
    except Exception as e:
        sys.exit(f"Could not read the channel {channel}:\n  {str(e)[:200]}\n"
                 "Check your internet connection, update yt-dlp (pip install -U yt-dlp), "
                 "or pass video links with --urls.")
    found = []
    for e in info.get("entries") or []:
        title = e.get("title") or ""
        if not pattern.search(title):
            continue
        d = meeting_date(title, e.get("upload_date"))
        if since and d and d < since:
            continue
        url = e.get("url") or f"https://www.youtube.com/watch?v={e.get('id')}"
        if not url.startswith("http"):
            url = f"https://www.youtube.com/watch?v={e.get('id')}"
        found.append({"title": title, "url": url, "video_id": e.get("id"),
                      "meeting_date": d})
        if len(found) >= count:
            break
    return found


def download_audio(url: str, out_dir: Path, bitrate: int) -> dict:
    """Download the audio track and convert it to MP3 at `bitrate` kbps."""
    import yt_dlp
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": False,
        "postprocessors": [{"key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3",
                            "preferredquality": str(bitrate)}],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return info


def load_metadata(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["video_id"]: r for r in csv.DictReader(f)}


def save_metadata(path: Path, rows: dict[str, dict]) -> None:
    ordered = sorted(rows.values(), key=lambda r: r["meeting_date"] or "", reverse=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(ordered)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect FOMC press conference audio from YouTube")
    ap.add_argument("--count", type=int, default=5, help="number of press conferences (default 5)")
    ap.add_argument("--since", help="only meetings on or after this date, YYYY-MM-DD")
    ap.add_argument("--urls", nargs="+", help="download these videos instead of searching")
    ap.add_argument("--out", default="audio", help="output folder (default: audio)")
    ap.add_argument("--bitrate", type=int, default=160, help="MP3 bitrate in kbps (default 160)")
    ap.add_argument("--channel", default=FED_CHANNEL)
    ap.add_argument("--scan", type=int, default=400,
                    help="how many recent channel videos to search through (default 400)")
    ap.add_argument("--title", default=TITLE_PATTERN.pattern,
                    help="regular expression a video title must match "
                         "(default: 'FOMC press conference', case-insensitive)")
    ap.add_argument("--list", action="store_true", help="only list the videos, don't download")
    args = ap.parse_args()

    require_tools()
    since = date.fromisoformat(args.since) if args.since else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.csv"
    meta = load_metadata(meta_path)

    if args.urls:
        videos = [{"title": "", "url": u, "video_id": None, "meeting_date": None}
                  for u in args.urls]
    else:
        print(f"Searching {args.channel} for FOMC press conferences ...")
        videos = find_press_conferences(args.channel, args.count, since, args.scan,
                                        re.compile(args.title, re.IGNORECASE))
        if not videos:
            sys.exit("No press conference videos found. Increase --scan, or pass --urls.")

    print(f"\n{len(videos)} video(s):")
    for v in videos:
        print(f"  {v['meeting_date'] or '?':<10}  {v['title'] or v['url']}")
    if args.list:
        return

    ok, failed = 0, []
    for v in videos:
        if v["video_id"] and v["video_id"] in meta and (out_dir / meta[v["video_id"]]["file"]).exists():
            print(f"\nSkipping {v['video_id']} (already downloaded)")
            ok += 1
            continue
        print(f"\nDownloading {v['title'] or v['url']} ...")
        try:
            info = download_audio(v["url"], out_dir, args.bitrate)
        except Exception as e:  # keep going with the rest
            print(f"  FAILED: {e}")
            failed.append(v["url"])
            continue
        vid = info.get("id")
        title = info.get("title") or v["title"]
        d = v["meeting_date"] or meeting_date(title, info.get("upload_date"))
        meta[vid] = {
            "meeting_date": d.isoformat() if d else "",
            "title": title,
            "url": info.get("webpage_url") or v["url"],
            "video_id": vid,
            "upload_date": info.get("upload_date") or "",
            "duration_sec": int(info.get("duration") or 0),
            "file": f"{vid}.mp3",
            "bitrate_kbps": args.bitrate,
            "scheduled_start_et": PRESS_CONF_START_ET,
        }
        save_metadata(meta_path, meta)
        mins = int(info.get("duration") or 0) // 60
        print(f"  saved {vid}.mp3  ({mins} min)")
        ok += 1

    total_min = sum(int(r["duration_sec"] or 0) for r in meta.values()) // 60
    print(f"\nDone: {ok} saved, {len(failed)} failed. Folder: {out_dir.resolve()}")
    print(f"metadata.csv lists {len(meta)} press conference(s), {total_min} minutes of audio.")
    if failed:
        print("Failed URLs (retry later or update yt-dlp: pip install -U yt-dlp):")
        for u in failed:
            print("  " + u)
    print("\nNote: scheduled_start_et is the Fed's usual 14:30 ET start. The video may "
          "begin a little earlier or later; check it before aligning audio with SPY prices.")


if __name__ == "__main__":
    main()
