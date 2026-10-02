"""Risk-attention trend, event windows, Reddit and GDELT comparisons, and measurement validation.

Runs after `analysis.py --model roberta`. It reads the scored corpus and trading panel,
adds the sources and diagnostics described in the report, and never alters primary results.
"""
import json, re, hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fetch_data import ROOT
from analysis import regression, RISK, EXIST, BASKET

OUT = ROOT/'results'; NY = 'America/New_York'
EVENT_WEEK = ('2026-09-09', '2026-09-13')       # first five calendar days after the 21:27 EDT post
LATE = ('2026-09-14', '2026-09-30')
SEMIS = ['NVDA', 'AVGO']; PLATFORMS = ['MSFT', 'AMZN', 'GOOGL', 'META']
AI_COMMUNITIES = ['artificial', 'singularity', 'technology']
INVESTOR_COMMUNITIES = ['stocks', 'investing', 'wallstreetbets']


def hac_mean(values):
    """Mean of a daily series with lag-one HAC uncertainty."""
    v = pd.Series(values).dropna().to_numpy(float)
    f = sm.OLS(v, np.ones((len(v), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': 1}, use_t=True)
    return float(f.params[0]), float(f.conf_int()[0, 0]), float(f.conf_int()[0, 1]), float(f.pvalues[0]), len(v)


def window_contrast(daily):
    """Event-week minus remainder difference in a daily series, lag-one HAC."""
    d = daily.dropna()
    x = d.index.to_series().between(*EVENT_WEEK).astype(float)
    late = d.index.to_series().between(*LATE)
    keep = (x == 1) | late
    f = sm.OLS(d[keep].to_numpy(float), sm.add_constant(x[keep].to_numpy())).fit(cov_type='HAC', cov_kwds={'maxlags': 1}, use_t=True)
    return {'event_week': float(d[x == 1].mean()), 'remainder': float(d[late].mean()), 'difference': float(f.params[1]),
            'ci_low': float(f.conf_int()[1, 0]), 'ci_high': float(f.conf_int()[1, 1]), 'p': float(f.pvalues[1]),
            'days': int(keep.sum()), 'peak_date': str(d.idxmax()), 'peak': float(d.max())}


def kappa(a, b):
    a = np.asarray(a); b = np.asarray(b); po = (a == b).mean()
    pe = sum((a == c).mean()*(b == c).mean() for c in np.union1d(a, b))
    return float((po-pe)/(1-pe))


def main():
    texts = pd.read_csv(OUT/'texts_scored.csv', dtype={'id': str}, low_memory=False)
    panel = pd.read_csv(OUT/'trading_panel.csv', index_col=0)
    summary = json.loads((OUT/'summary.json').read_text())

    # ---------------- Reddit (frozen RoBERTa probabilities, same pinned model file) ----------------
    red = pd.read_csv(ROOT/'data'/'reddit.csv', dtype={'id': str})
    rs = pd.read_csv(ROOT/'data'/'reddit_scores.csv', dtype={'id': str})
    red = red.merge(rs, on='id', how='left', validate='one_to_one')
    assert red.p_negative.notna().all(), 'Unscored Reddit post'
    red['risk_mention'] = red.text.str.contains(RISK).astype(int)
    red['existential_mention'] = red.text.str.contains(EXIST).astype(int)
    vader = SentimentIntensityAnalyzer()
    red['vader_tone'] = red.text.map(lambda s: vader.polarity_scores(s)['compound'])
    red['created'] = pd.to_datetime(red.created_utc, utc=True)
    start = pd.Timestamp('2026-09-08', tz=NY); end = pd.Timestamp('2026-10-01', tz=NY)
    assert red.created.ge(start).all() and red.created.lt(end).all()
    assert not red.id.duplicated().any()
    red['date'] = red.created.dt.tz_convert(NY).dt.strftime('%Y-%m-%d')
    red['tone'] = red.p_positive-red.p_negative
    red['negative'] = (red[['p_negative', 'p_neutral', 'p_positive']].to_numpy().argmax(axis=1) == 0).astype(int)
    red['community'] = np.where(red.subreddit.str.lower().isin(INVESTOR_COMMUNITIES), 'Investor subreddits', 'AI/tech subreddits')
    close_grid = pd.DatetimeIndex([pd.Timestamp(d+' 16:00', tz=NY).tz_convert('UTC') for d in panel.index])
    pos = close_grid.searchsorted(red.created)
    red['trading_date'] = [panel.index[i] if i < len(panel) else None for i in pos]
    red.to_csv(OUT/'reddit_scored.csv', index=False)

    # ---------------- Daily calendar series for all sources ----------------
    cal = texts[['date', 'source', 'tone', 'negative', 'risk_mention', 'existential_mention', 'id']].rename(columns={'risk_mention': 'risk', 'existential_mention': 'existential'})
    cal = pd.concat([cal, red[['date', 'tone', 'negative', 'risk_mention', 'existential_mention', 'id']].rename(columns={'risk_mention': 'risk', 'existential_mention': 'existential'}).assign(source='Reddit')])
    cal_news = cal[cal.source.isin(['Guardian', 'TechCrunch'])].assign(source='News (pooled)')
    daily = pd.concat([cal, cal_news]).groupby(['date', 'source']).agg(n=('id', 'size'), tone=('tone', 'mean'), negative=('negative', 'mean'), risk=('risk', 'mean'), existential=('existential', 'mean')).reset_index()
    daily.to_csv(OUT/'calendar_daily_all.csv', index=False)

    # ---------------- Risk attention: event week versus remainder ----------------
    rows = []
    for src in ['Hacker News', 'Reddit', 'News (pooled)']:
        g = daily[daily.source.eq(src)].set_index('date')
        for metric in ['existential', 'risk', 'tone', 'negative']:
            r = window_contrast(g[metric]); r.update(source=src, metric=metric); rows.append(r)
    ew = pd.DataFrame(rows)
    ew['p_holm'] = np.nan
    for metric, idx in ew.groupby('metric').groups.items():
        ew.loc[idx, 'p_holm'] = multipletests(ew.loc[idx, 'p'], method='holm')[1]
    ew.to_csv(OUT/'event_window.csv', index=False)
    # Existential-risk subset tone: does discussion of the warning itself turn more negative?
    sub = cal[(cal.existential == 1) & cal.source.isin(['Hacker News', 'Reddit'])]
    subset = sub.groupby(['source', sub.date.between(*EVENT_WEEK).map({True: 'Sep 9-13', False: 'other'})]).agg(n=('id', 'size'), tone=('tone', 'mean'), negative=('negative', 'mean')).reset_index()
    subset.to_csv(OUT/'existential_subset.csv', index=False)

    # ---------------- GDELT: English-language web news, US versus non-US sources ----------------
    g = json.loads((ROOT/'data'/'gdelt.json').read_text())
    gd = []
    for key, bins in g['tonechart'].items():
        b = pd.DataFrame(bins, columns=['bin', 'count'])
        n = int(b['count'].sum())
        gd.append({'query': key, 'articles': n, 'negative_share': float(b.loc[b.bin < 0, 'count'].sum()/n), 'positive_share': float(b.loc[b.bin > 0, 'count'].sum()/n),
                   'mean_bin_tone': float((b.bin*b['count']).sum()/n)})
    gd = pd.DataFrame(gd).set_index('query')
    vol = {}
    for key, series in g['timelinevolraw'].items():
        vol[key] = pd.DataFrame(series, columns=['date', 'value', 'norm']).assign(date=lambda d: d.date.str[:8])
    for side in ['us', 'nonus']:
        if f'risk_{side}' in gd.index and f'ai_{side}' in vol:
            gd.loc[f'risk_{side}', 'risk_share_of_ai_coverage'] = gd.loc[f'risk_{side}', 'articles']/vol[f'ai_{side}'].value.sum()
    gd.to_csv(OUT/'gdelt_tone.csv')
    # US versus non-US negative share of AI-risk coverage (articles as units; syndicated copies are not independent)
    n1, n2 = gd.loc['risk_us', 'articles'], gd.loc['risk_nonus', 'articles']
    p1, p2 = gd.loc['risk_us', 'negative_share'], gd.loc['risk_nonus', 'negative_share']
    pp = (p1*n1+p2*n2)/(n1+n2); z = (p1-p2)/np.sqrt(pp*(1-pp)*(1/n1+1/n2))
    se = np.sqrt(p1*(1-p1)/n1+p2*(1-p2)/n2)
    from scipy.stats import norm
    (OUT/'gdelt_us_nonus.json').write_text(json.dumps({'us_negative_share': p1, 'nonus_negative_share': p2, 'us_articles': int(n1), 'nonus_articles': int(n2),
        'difference': p1-p2, 'ci_low': p1-p2-1.96*se, 'ci_high': p1-p2+1.96*se, 'z': float(z), 'p': float(2*norm.sf(abs(z)))}, indent=2))
    if 'risk_all' in vol and 'ai_all' in vol:
        share = vol['risk_all'].merge(vol['ai_all'], on='date', suffixes=('_risk', '_ai'))
        share['risk_share_of_ai'] = share.value_risk/share.value_ai
        # GDELT labels each 24-hour UTC bin by its end; 'day' is the UTC day the bin covers.
        share['day'] = (pd.to_datetime(share.date, format='%Y%m%d')-pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
        share.to_csv(OUT/'gdelt_risk_share.csv', index=False)

    # ---------------- Event-day returns ----------------
    excess_sd = (panel.basket-panel.QQQ).std(); semis_sd = (panel[SEMIS].mean(axis=1)-panel.QQQ).std()
    ed = []
    for day, label in [('2026-09-09', 'First session after the Hubinger post'), ('2026-09-14', 'First session after the Amodei slowdown call'), ('2026-09-29', 'Session after the Anthropic prospectus')]:
        r = panel.loc[day]
        ed.append({'date': day, 'event': label, 'basket': r.basket, 'semis': r[SEMIS].mean(), 'platforms': r[PLATFORMS].mean(), 'QQQ': r.QQQ, 'SPY': r.SPY,
                   'basket_minus_QQQ': r.basket-r.QQQ, 'semis_minus_QQQ': r[SEMIS].mean()-r.QQQ,
                   'basket_excess_sd': excess_sd, 'semis_excess_sd': semis_sd, 'excess_z': (r.basket-r.QQQ)/excess_sd, 'semis_excess_z': (r[SEMIS].mean()-r.QQQ)/semis_sd})
    pd.DataFrame(ed).to_csv(OUT/'event_day_returns.csv', index=False)

    # ---------------- Supplementary Granger tests (separate Holm family) ----------------
    win = texts.dropna(subset=['trading_date'])
    def by_window(frame, col):
        return frame.groupby('trading_date')[col].mean().reindex(panel.index)
    hn = win[win.source.eq('Hacker News')]
    social_pool = pd.concat([hn[['trading_date', 'existential_mention']], red.dropna(subset=['trading_date'])[['trading_date', 'existential_mention']]])
    news = win[win.kind.eq('news')]
    preds = {
        'Reddit tone change': by_window(red.dropna(subset=['trading_date']), 'tone').diff(),
        'HN risk-attention change': by_window(hn, 'existential_mention').diff(),
        'Social risk-attention change (HN+Reddit)': by_window(social_pool, 'existential_mention').diff(),
        'News risk-attention change': by_window(news, 'existential_mention').diff(),
    }
    sup = []
    for name, x in preds.items():
        row, d, _ = regression(panel.basket, x); row.update(test=name, family='supplementary'); sup.append(row)
    sup = pd.DataFrame(sup); sup['p_holm'] = multipletests(sup.p, method='holm')[1]
    sup.to_csv(OUT/'supplementary_granger.csv', index=False)
    pd.DataFrame({k: v for k, v in preds.items()}).assign(basket=panel.basket).to_csv(OUT/'supplementary_panel.csv')

    # ---------------- Source framing, Reddit contrasts (separate exploratory family) ----------------
    comps = []
    piv = {m: daily.pivot(index='date', columns='source', values=m) for m in ['tone', 'negative', 'risk', 'existential']}
    for metric, p in piv.items():
        for a, b in [('Reddit', 'Hacker News'), ('Reddit', 'News (pooled)')]:
            m, lo, hi, pv, n = hac_mean(p[a]-p[b])
            comps.append({'metric': metric, 'a': a, 'b': b, 'days': n, 'difference': m, 'ci_low': lo, 'ci_high': hi, 'p': pv})
    rc = pd.DataFrame(comps); rc['p_holm'] = multipletests(rc.p, method='holm')[1]
    rc.to_csv(OUT/'reddit_gaps.csv', index=False)
    comm = red.groupby('community').agg(n=('id', 'size'), tone=('tone', 'mean'), negative=('negative', 'mean'), risk=('risk_mention', 'mean'), existential=('existential_mention', 'mean'))
    cd = red.groupby(['date', 'community']).negative.mean().unstack()
    m, lo, hi, pv, n = hac_mean(cd['Investor subreddits']-cd['AI/tech subreddits'])
    comm.attrs = {}
    comm.to_csv(OUT/'reddit_communities.csv')
    (OUT/'reddit_community_gap.json').write_text(json.dumps({'metric': 'negative share, investor minus AI/tech subreddits', 'difference': m, 'ci_low': lo, 'ci_high': hi, 'p': pv, 'days': n}, indent=2))

    # ---------------- Measurement validation ----------------
    val = measurement_validation()
    xs = x_sample()

    extra = {'reddit_n': int(len(red)), 'reddit_after_final_close': int(red.trading_date.isna().sum()), 'reddit_by_subreddit': red.subreddit.value_counts().to_dict(),
             'gdelt_missing_days': g.get('missing_days', []), 'validation': val, 'x_sample': xs}
    (OUT/'extensions_summary.json').write_text(json.dumps(extra, indent=2))
    print(json.dumps(extra, indent=2))


def measurement_validation():
    """Classifier and risk-indicator checks against the 150-text codebook sample and, when completed,
    the author's independent blind recoding of 50 texts (validation/blind_review_50.csv)."""
    V = ROOT/'validation'
    code = pd.read_csv(V/'codebook_labels_150.csv', dtype={'id': str})
    scores = pd.read_csv(ROOT/'data'/'contextual_scores.csv')
    code['text_hash'] = code.text.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    code = code.merge(scores, on='text_hash', how='left', validate='many_to_one')
    assert code.p_negative.notna().all()
    code['roberta'] = code[['p_negative', 'p_neutral', 'p_positive']].to_numpy().argmax(axis=1)-1
    vader = SentimentIntensityAnalyzer()
    code['vader'] = code.text.map(lambda s: vader.polarity_scores(s)['compound']).map(lambda c: 1 if c >= .05 else -1 if c <= -.05 else 0)
    code['regex_risk'] = code.text.str.contains(RISK).astype(int)
    def metrics(frame, ref):
        out = {'n': int(len(frame))}
        for m in ['roberta', 'vader']:
            out[m+'_agreement'] = float((frame[m] == frame[ref]).mean()); out[m+'_kappa'] = kappa(frame[m], frame[ref])
            neg = frame[ref] == -1; pred = frame[m] == -1
            out[m+'_negative_precision'] = float((neg & pred).sum()/max(pred.sum(), 1)); out[m+'_negative_recall'] = float((neg & pred).sum()/max(neg.sum(), 1))
        return out
    res = {'codebook': metrics(code, 'sentiment')}
    rr = code.regex_risk == 1; rk = code.ai_risk == 1
    res['codebook'].update(risk_regex_precision=float((rr & rk).sum()/rr.sum()), risk_regex_recall=float((rr & rk).sum()/rk.sum()))
    code.to_csv(OUT/'validation_codebook_scored.csv', index=False)
    blind = pd.read_csv(V/'blind_review_50.csv', encoding='utf-8-sig', encoding_errors='replace')
    key = pd.read_csv(V/'blind_key.csv')
    lab = pd.to_numeric(blind.sentiment, errors='coerce'); risk = pd.to_numeric(blind.ai_risk, errors='coerce')
    complete = len(blind) == 50 and lab.isin([-1, 0, 1]).all() and risk.isin([0, 1]).all()
    res['author_recode_complete'] = bool(complete)
    if complete:
        b = blind.assign(sentiment=lab.astype(int), ai_risk=risk.astype(int))[['blind_id', 'sentiment', 'ai_risk']].merge(key, on='blind_id', validate='one_to_one')
        b = b.merge(code[['item', 'sentiment', 'ai_risk', 'roberta', 'vader', 'regex_risk']], on='item', suffixes=('_author', '_codebook'), validate='one_to_one')
        a = metrics(b.rename(columns={'sentiment_author': 'ref'}), 'ref')
        a.update(intercoder_agreement=float((b.sentiment_author == b.sentiment_codebook).mean()), intercoder_kappa=kappa(b.sentiment_author, b.sentiment_codebook),
                 risk_intercoder_agreement=float((b.ai_risk_author == b.ai_risk_codebook).mean()), risk_intercoder_kappa=kappa(b.ai_risk_author, b.ai_risk_codebook))
        res['author'] = a
        b.to_csv(OUT/'validation_author_recode.csv', index=False)
    (OUT/'validation_coding.json').write_text(json.dumps(res, indent=2))
    return res


def x_sample():
    """Manually collected X sample: systematic reactions to the post and a daily search sample."""
    X = pd.read_csv(ROOT/'data'/'x_sample'/'x_sample.csv', dtype={'id': str})
    sc = pd.read_csv(ROOT/'data'/'x_sample'/'x_scores.csv', dtype={'id': str})
    X = X.merge(sc, on='id', how='left', validate='one_to_one')
    assert (X.text.map(lambda t: hashlib.sha256(t.encode()).hexdigest()) == X.text_sha256).all(), 'X text/score mismatch'
    X['tone'] = X.p_positive-X.p_negative
    X['negative'] = (X[['p_negative', 'p_neutral', 'p_positive']].to_numpy().argmax(axis=1) == 0).astype(int)
    X['risk_mention'] = X.text.str.contains(RISK).astype(int); X['existential_mention'] = X.text.str.contains(EXIST).astype(int)
    X['date'] = pd.to_datetime(X.created_utc, utc=True).dt.tz_convert(NY).dt.strftime('%Y-%m-%d')
    code = pd.read_csv(ROOT/'data'/'x_sample'/'x_reaction_coding.csv', dtype={'id': str}, encoding='utf-8-sig')
    X = X.merge(code[['id', 'stance']], on='id', how='left')
    X.to_csv(OUT/'x_scored.csv', index=False)
    daily = X[X.frame.eq('daily')]
    assert daily.date.eq(daily.search_date).all() and daily.date.between('2026-09-08', '2026-09-30').all()
    react = X[X.frame.isin(['quote', 'reply']) & X.exclude_reason.isna() & X.stance.ne('copy')]
    st = react.groupby('stance').agg(n=('id', 'size'), likes=('likes', 'sum'), negative=('negative', 'mean'), tone=('tone', 'mean'))
    st['share'] = st.n/st.n.sum(); st['like_share'] = st.likes/st.likes.sum()
    st.to_csv(OUT/'x_stance.csv')
    per = daily.assign(period=np.where(daily.date.le('2026-09-13'), 'Sep 8-13', 'Sep 14-30')).groupby('period').agg(
        n=('id', 'size'), negative=('negative', 'mean'), tone=('tone', 'mean'), risk=('risk_mention', 'mean'), existential=('existential_mention', 'mean'))
    out = {'reactions_n': int(len(react)), 'reactions_negative': float(react.negative.mean()), 'reactions_tone': float(react.tone.mean()),
           'reactions_like_weighted_negative': float((react.negative*react.likes).sum()/react.likes.sum()),
           'daily_n': int(len(daily)), 'daily_negative': float(daily.negative.mean()), 'daily_tone': float(daily.tone.mean()),
           'daily_periods': per.reset_index().to_dict('records'), 'excluded': X.loc[X.exclude_reason.notna(), 'url'].tolist()}
    (OUT/'x_summary.json').write_text(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    main()
