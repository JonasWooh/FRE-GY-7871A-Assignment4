# AI Risk Warnings, Media Tone and AI Equity Returns

Research code for a study of AI-related news and social-media sentiment around Evan Hubinger's
September 8, 2026 post estimating a >10% chance that AI kills all humans within a decade, and of its
relation to returns on an equal-weight basket of six AI equities (NVDA, MSFT, AMZN, GOOGL, META, AVGO),
September 8-30, 2026. The full write-up is in [`report/AI_Sentiment_Report.pdf`](report/AI_Sentiment_Report.pdf).

## What the code does

| Step | Script | Output (written to `data/` or `results/`) |
|---|---|---|
| Collect news, Hacker News comments, prices | `fetch_data.py all` | Guardian AI-section headlines, HN comments with AI vocabulary (Algolia API), Yahoo adjusted closes |
| Collect TechCrunch | `fetch_techcrunch.py` | AI-category headlines (publisher API) |
| Collect Reddit | `fetch_reddit.py` | AI posts in six subreddits (Arctic Shift archive), scored |
| Collect GDELT | `fetch_gdelt.py` | US vs non-US tone histograms and daily volumes |
| Lexicon | `fetch_lexicon.py` | Hu-Liu opinion lexicon |
| Sentiment | `contextual_sentiment.py` | Frozen Cardiff Twitter-RoBERTa (pinned INT8 ONNX revision), P(pos) − P(neg) |
| Core analysis | `analysis.py --model roberta` (and `--model vader`) | Daily tone, trading-window alignment, basket returns, primary and secondary Granger tests, source contrasts |
| Extensions | `extensions.py` | Risk-attention trend and event window, Reddit and GDELT comparisons, event-day returns, supplementary Granger tests, measurement validation, X sample |

The Granger specification is `r_t = α + φ r_{t−1} + β ΔS_{t−1} + ε_t`, tested with a nested-model F test;
texts are assigned to the first regular close at or after publication.

## Running

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt  (Python 3.11)
python fetch_data.py all
python fetch_techcrunch.py
python fetch_lexicon.py
python contextual_sentiment.py
python fetch_reddit.py --suffix snapshot
python fetch_gdelt.py
python analysis.py --model vader
python analysis.py --model roberta
python extensions.py
```

Collected data are not included in this repository. A new collection reflects the archives at the time it is run
(deleted or edited posts change), so it will not reproduce the original snapshot exactly. `fetch_reddit.py` and
`fetch_gdelt.py` write `*_<suffix>` files; rename them to `reddit.csv` / `reddit_scores.csv` / `gdelt.json` before
running `extensions.py`. The measurement-validation and X-sample steps in `extensions.py` additionally expect the
hand-coded files under `validation/` and `data/x_sample/`, which are kept with the original project.
