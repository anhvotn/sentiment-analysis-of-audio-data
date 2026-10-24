# Fed Press Conference Sentiment Signals

A pipeline that turns the spoken words of Federal Reserve press conferences into trading signals for SPY. It downloads the audio, cuts it into overlapping one-minute "audio bars", transcribes them with Whisper, scores each bar with FinBERT, and generates buy/sell signals that can be backtested on intraday SPY prices.

It follows the pipeline taught in a Coursera course on using LLMs for sentiment analysis in trading, with a few practical changes described below. The repository contains my own implementation and results; no course material is included.

---

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. Collect audio | `collect_fed_audio.py` | Finds FOMC press conference videos on the Federal Reserve's YouTube channel and saves the audio as MP3 at 160 kbps |
| 2. Segment | `segment_audio.py` | Cuts each recording into 60-second bars, a new one every 30 seconds, each usable 15 seconds after it ends (embargo against look-ahead) |
| 3. Transcribe | `transcribe_audio.py` | Transcribes each full recording with Whisper (base model), with a timestamp for every word, then assigns the words to the bars |
| 4. Score | `score_sentiment.py` | Scores every bar with FinBERT: score = P(positive) − P(negative), from −1 to +1 |
| 5. Signals and backtest | `generate_signals.py` | Buys above the 80th percentile of scores, sells below the 20th, flat in between; backtests on 30-second SPY prices if you supply them |

Each script reads the previous step's output, skips work that is already done, and can be rerun safely.

```
audio/          MP3s + metadata.csv            (step 1)
bars/           bars.csv, WAV bars             (step 2)
transcripts/    full transcripts, JSON + text  (step 3)
bars/           bar_transcripts.csv            (step 3)
bars/           bar_sentiment.csv              (step 4)
signals/        signals.csv, backtest files    (step 5)
```

---

## Results so far

Five press conferences from March to September 2026: 3.7 hours of audio, 430 bars, about 35,000 transcribed words. Whisper's average word confidence was 0.91–0.93 for every meeting.

FinBERT scores by meeting:

| Meeting | Bars | Mean score | Positive | Negative | Neutral |
|---|---|---|---|---|---|
| 18 Mar 2026 | 95 | −0.109 | 5 | 18 | 72 |
| 29 Apr 2026 | 106 | +0.005 | 5 | 6 | 95 |
| 17 Jun 2026 | 84 | +0.067 | 2 | 0 | 82 |
| 29 Jul 2026 | 89 | +0.079 | 5 | 0 | 84 |
| 16 Sep 2026 | 56 | +0.109 | 7 | 2 | 47 |

Most bars are neutral, and the tone becomes steadily more positive from March to September.

Signals with thresholds from all five meetings (buy above +0.108, sell below −0.064):

| Meeting | Buy | Sell | Flat |
|---|---|---|---|
| 18 Mar 2026 | 14 | 42 | 39 |
| 29 Apr 2026 | 13 | 21 | 72 |
| 17 Jun 2026 | 19 | 5 | 60 |
| 29 Jul 2026 | 19 | 9 | 61 |
| 16 Sep 2026 | 20 | 9 | 27 |

**No backtest results yet.** The backtest needs 30-second SPY prices, which are not included (see [Data](#data)).

---

## Setup

Python 3.10+ and [ffmpeg](https://ffmpeg.org/) on your PATH (Windows: `winget install --id Gyan.FFmpeg -e`).

```bash
pip install -r requirements.txt
```

The first runs of steps 3 and 4 download the Whisper base model (about 150 MB) and FinBERT (about 440 MB).

## Usage

```bash
python collect_fed_audio.py --list      # show which press conferences would be downloaded
python collect_fed_audio.py             # 5 most recent; --count 10 or --since 2022-01-01 for more
python segment_audio.py
python transcribe_audio.py              # 15–60 min on a laptop CPU for 5 meetings
python score_sentiment.py
python generate_signals.py              # signals only
python generate_signals.py --spy spy_prices.csv            # signals + backtest
python generate_signals.py --mode expanding --out signals_expanding
```

Useful options:

| Script | Option | Meaning |
|---|---|---|
| `segment_audio.py` | `--bar 60 --step 30 --embargo 15` | Bar length, spacing and embargo in seconds |
| `transcribe_audio.py` | `--model small` | Larger Whisper model: slower, more accurate |
| `transcribe_audio.py` | `--engine openai` | Use `openai-whisper` instead of `faster-whisper` |
| `score_sentiment.py` | `--model <name>` | Another Hugging Face classifier |
| `generate_signals.py` | `--upper 0.9 --lower 0.1` | Signal thresholds as score quantiles |
| `generate_signals.py` | `--mode expanding` | Thresholds from earlier meetings only (no look-ahead) |
| `generate_signals.py` | `--cost-bps 0.5` | Trading cost per unit of position change |
| `generate_signals.py` | `--spy-tz UTC` | Time zone of the SPY timestamps, if not US Eastern |

---

## Data

**Included:** transcripts, bar-level sentiment scores and signals, and `audio/metadata.csv` with the links to the source videos. The press conferences are public statements by a US government agency.

**Not included, but recreated by the scripts:** the MP3s (about 250 MB) and the WAV audio bars (about 790 MB).

**Not included, and needed for the backtest:** intraday SPY prices for each press conference, roughly 14:30–15:30 New York time. Provide a CSV with a timestamp column and a close or price column, at 30-second intervals as in the course, or finer. Coarser data such as 1-minute bars works, but only approximately; the script warns about it. Free sources rarely cover more than the last month at this resolution; a broker's historical data or a pay-per-use data vendor are the practical options. Check the provider's terms before publishing any prices.

---

## Design choices

Where this implementation differs from, or adds to, the course:

- **yt-dlp instead of pytube.** pytube is no longer maintained and often fails with current YouTube.
- **Transcribe once, then split.** Each full recording is transcribed once, with word timestamps, and the words are assigned to bars by their midpoint. The bars are the same as when each bar is transcribed separately, but it is about twice as fast and more accurate, because Whisper hears whole sentences.
- **faster-whisper.** The same Whisper base model, running several times faster on a CPU. Audio is decoded with ffmpeg, which avoids a version conflict between faster-whisper and recent PyAV releases.
- **Signal rules not specified in the course** (all adjustable): 80th/20th percentile thresholds, flat between them, positions held for one 30-second step, trades at the first :00 or :30 after the embargo, 0.5 bp cost per unit traded.
- **Two threshold modes.** `insample` uses the whole sample, as in the course, which includes look-ahead. `expanding` uses only earlier meetings, which is what could actually have been traded.
- **Clock times** assume each recording starts at the scheduled 14:30 ET. In this sample, speech begins 5–10 seconds into every recording. A `start_offset_sec` column in `audio/metadata.csv` corrects a late start.

---

## Limitations

- **FinBERT measures general tone, not monetary policy stance.** For the stock market, what matters is whether the Fed sounds hawkish (rates higher for longer) or dovish. FinBERT reads "PCE prices rose 3.5%" as positive (the strongest buy signal in this sample), although higher inflation usually points to higher rates, which is negative for stocks.
- **Bars cut sentences.** A 60-second window often starts and ends mid-sentence, which makes classification harder.
- **Restated facts look like news.** Much of the opening statement repeats data already published, which the market has priced in.
- **Small sample.** Five meetings and 430 bars cannot establish whether the signals work. The in-sample thresholds also use information from later meetings.

## Possible improvements

1. Score hawkish/dovish stance with a model trained on Fed text, such as FOMC-RoBERTa from [Trillion Dollar Words](https://arxiv.org/abs/2305.07972) (Shah, Paturi & Chava, ACL 2023).
2. Score whole sentences and average them within each bar.
3. Separate the opening statement from the Q&A, and measure changes in tone rather than levels.
4. Add features from the voice itself: Gorodnichenko, Pham & Talavera, "The Voice of Monetary Policy" (*American Economic Review*, 2023), find that the Fed Chair's vocal tone moves markets separately from the words.

---

## References

- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models.
- Radford, A. et al. (2022). Robust speech recognition via large-scale weak supervision (Whisper).
- Shah, A., Paturi, S. & Chava, S. (2023). Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis. ACL 2023.
- Gorodnichenko, Y., Pham, T. & Talavera, O. (2023). The Voice of Monetary Policy. *American Economic Review*, 113(2), 548–584.

## Disclaimer

This is a research and learning project. It is not investment advice, and no result here shows that the signals are profitable.
