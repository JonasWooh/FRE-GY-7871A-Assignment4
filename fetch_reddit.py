"""Reddit posts (Arctic Shift archive API) for the study window, scored with the pinned RoBERTa model.

The retained snapshot (`data/reddit.csv`, `data/reddit_scores.csv`) was produced with exactly this query logic:
six subreddits, every post created 2026-09-08 00:00 to 2026-10-01 00:00 New York time, kept when
title + body matches the same AI vocabulary used for Hacker News. Text = title + first 2,000 body
characters ([removed]/[deleted] bodies dropped). Scores use the same model file (SHA-256 in
data/contextual_model.json), one text per inference call, 512-token truncation.

Running this script writes NEW files (data/reddit_<suffix>.csv and data/reddit_scores_<suffix>.csv)
so that the retained snapshot is never overwritten; archive deletions can change a later collection.
"""
import argparse, hashlib, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from fetch_data import ROOT, START, END, AI

SUBREDDITS = ['artificial', 'singularity', 'technology', 'stocks', 'investing', 'wallstreetbets']
API = 'https://arctic-shift.photon-reddit.com/api/posts/search'
FIELDS = 'id,created_utc,subreddit,title,selftext,score,num_comments,author,url'


def collect():
    rows = {}
    lo, hi = int(START.timestamp()), int(END.timestamp())
    for sub in SUBREDDITS:
        after = lo-1
        while True:
            r = requests.get(API, params={'subreddit': sub, 'after': after, 'before': hi, 'limit': 100, 'sort': 'asc', 'fields': FIELDS}, timeout=60)
            r.raise_for_status(); data = r.json()['data']
            for p in data: rows[p['id']] = p
            if len(data) < 100: break
            last = data[-1]['created_utc']; after = last-1 if last-1 > after else last
            time.sleep(.25)
    out = []
    for p in rows.values():
        clean = lambda s: re.sub(r'\s+', ' ', s or '').strip()
        body = clean(p.get('selftext'))
        body = '' if body in ('[removed]', '[deleted]') else body
        title = clean(p.get('title')); text = (title+' '+body[:2000]).strip() if body else title
        if not AI.search(title+' '+body): continue
        out.append({'id': p['id'], 'created_utc': pd.Timestamp(p['created_utc'], unit='s', tz='UTC').isoformat(), 'subreddit': p['subreddit'],
                    'score': p.get('score'), 'num_comments': p.get('num_comments'), 'title': title, 'text': text})
    return pd.DataFrame(out).sort_values('created_utc')


def score(texts):
    import onnxruntime as ort
    from scipy.special import softmax
    from transformers import AutoTokenizer
    from contextual_sentiment import MODEL, REV, preprocess
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(MODEL, revision=REV, allow_patterns=['*.json', '*.txt', 'onnx/model_quantized.onnx'], local_dir=ROOT/'models'/'twitter_roberta'))
    meta = json.loads((ROOT/'data'/'contextual_model.json').read_text())
    assert meta['model_sha256'] == hashlib.sha256((path/'onnx'/'model_quantized.onnx').read_bytes()).hexdigest()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    sess = ort.InferenceSession(str(path/'onnx'/'model_quantized.onnx'), providers=['CPUExecutionProvider'])
    res = []
    for t in texts:
        p = preprocess(t); n = len(tok(p, truncation=False)['input_ids'])
        enc = tok(p, truncation=True, max_length=512, return_tensors='np')
        pr = softmax(sess.run(None, {x.name: enc[x.name].astype(np.int64) for x in sess.get_inputs()})[0], axis=1)[0]
        res.append((float(pr[0]), float(pr[1]), float(pr[2]), n))
    return pd.DataFrame(res, columns=['p_negative', 'p_neutral', 'p_positive', 'tokens'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--suffix', default='refresh'); a = ap.parse_args()
    assert a.suffix, 'A suffix is required so the retained snapshot is never overwritten'
    d = collect().reset_index(drop=True)
    d['url'] = 'https://www.reddit.com/comments/'+d.id
    d.to_csv(ROOT/'data'/f'reddit_{a.suffix}.csv', index=False)
    s = score(d.text.tolist()); s.insert(0, 'id', d.id)
    s.to_csv(ROOT/'data'/f'reddit_scores_{a.suffix}.csv', index=False)
    print('saved', len(d), 'posts')
