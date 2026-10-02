"""Reproduce tone, return, bias and Granger estimates from saved source observations."""
from pathlib import Path
import json, re, sys, importlib.metadata, hashlib, argparse
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.tsa.stattools import adfuller
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fetch_data import ROOT,START,END,AI

PARSER=argparse.ArgumentParser();PARSER.add_argument('--model',choices=['vader','roberta'],default='roberta')
ARGS=PARSER.parse_args() if __name__=='__main__' else argparse.Namespace(model='roberta')
OUT=ROOT/('results' if ARGS.model=='roberta' else 'results_vader'); OUT.mkdir(exist_ok=True)
STAMP=pd.Timestamp(json.loads((ROOT/'data'/'acquisition.json').read_text())['last_run_utc'])
CUTOFF=min(STAMP,END).tz_convert('UTC')
NY='America/New_York'
BASKET=['NVDA','MSFT','AMZN','GOOGL','META','AVGO']
# Literal content indicators, not classifiers of truth, risk magnitude or stance.
RISK=re.compile(r'\b(?:risks?|danger\w*|threat\w*|safety|unsafe|kill\w*|extinct\w*|catastroph\w*|doom\w*|harm\w*|alignment)\b',re.I)
EXIST=re.compile(r'\b(?:extinct\w*|existential|superintelligen\w*|kill (?:all |us |every)|doom\w*)',re.I)

def installed_versions(names):
    out={}
    for x in names:
        try:out[x]=importlib.metadata.version(x)
        except importlib.metadata.PackageNotFoundError:pass
    return out

def regression(y,x,p=1,control=None):
    # Shift before dropping missing rows: missing days must never be compressed.
    d=pd.DataFrame({'y':y})
    own=[]; xx=[]
    for k in range(1,p+1):
        d[f'y{k}']=y.shift(k); own.append(f'y{k}')
        d[f'x{k}']=x.shift(k); xx.append(f'x{k}')
    if control is not None:
        d['market1']=control.shift(1); own.append('market1')
    d=d.dropna(); X=sm.add_constant(d[own+xx]); X0=sm.add_constant(d[own])
    assert np.linalg.matrix_rank(X)==X.shape[1]
    f=sm.OLS(d.y,X).fit(); f0=sm.OLS(d.y,X0).fit()
    F,pvalue,df=f.compare_f_test(f0)
    hc=f.get_robustcov_results(cov_type='HC3')
    hc_test=hc.f_test(' , '.join(f'{v}=0' for v in xx))
    ci=f.conf_int().loc['x1']
    row={'n':len(d),'df_resid':f.df_resid,'lag':p,'beta':f.params.x1,'ci_low':ci.iloc[0],'ci_high':ci.iloc[1],
         'F':float(F),'p':float(pvalue),'p_HC3':float(hc_test.pvalue),'r2':f.rsquared,'bg_p':acorr_breusch_godfrey(f,nlags=1)[1]}
    return row,d,f

def main():
    texts=pd.concat([pd.read_csv(ROOT/'data'/f,dtype={'id':str}) for f in ['news.csv','techcrunch.csv','social.csv']],ignore_index=True)
    texts['created_utc']=pd.to_datetime(texts.created_utc,utc=True,format='mixed')
    assert texts.created_utc.ge(START).all() and texts.created_utc.lt(END).all()
    texts=texts[texts.created_utc<CUTOFF].copy()
    texts['date']=texts.created_utc.dt.tz_convert(NY).dt.strftime('%Y-%m-%d')
    analyzer=SentimentIntensityAnalyzer()
    negwords={w.strip() for w in (ROOT/'data'/'lexicon'/'negative.txt').read_text(encoding='latin-1').splitlines() if w and not w.startswith(';')}
    poswords={w.strip() for w in (ROOT/'data'/'lexicon'/'positive.txt').read_text(encoding='latin-1').splitlines() if w and not w.startswith(';')}
    def hul(i):
        words=re.findall(r"[a-z]+(?:'[a-z]+)?",i.lower())
        if not words:return np.nan,np.nan
        n=sum(w in negwords for w in words);p=sum(w in poswords for w in words)
        return (p-n)/len(words),n/len(words)
    texts['tone']=texts.text.map(lambda s:analyzer.polarity_scores(s)['compound'])
    texts['vader_tone']=texts.tone
    hl=texts.text.map(hul)
    texts['hu_liu']=hl.map(lambda x:x[0]);texts['hu_liu_negative']=hl.map(lambda x:x[1])
    texts['explicit_ai']=texts.text.str.contains(AI)
    texts['negative']=texts.tone.le(-.05).astype(int)
    if ARGS.model=='roberta':
        scores=pd.read_csv(ROOT/'data'/'contextual_scores.csv')
        texts['text_hash']=texts.text.map(lambda t:hashlib.sha256(t.encode()).hexdigest())
        texts=texts.merge(scores,on='text_hash',how='left',validate='many_to_one')
        assert texts[['p_negative','p_neutral','p_positive']].notna().all().all(), 'Unscored text'
        texts['tone']=texts.p_positive-texts.p_negative
        texts['negative']=(texts[['p_negative','p_neutral','p_positive']].to_numpy().argmax(axis=1)==0).astype(int)
    texts['risk_mention']=texts.text.str.contains(RISK).astype(int)
    texts['existential_mention']=texts.text.str.contains(EXIST).astype(int)
    texts['words']=texts.text.str.split().str.len()
    complete_date=(CUTOFF.tz_convert(NY).normalize()-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    if CUTOFF>=END:complete_date='2026-09-30'
    observed_date=min(CUTOFF-pd.Timedelta(seconds=1),END-pd.Timedelta(seconds=1)).tz_convert(NY).strftime('%Y-%m-%d')
    texts['complete_day']=texts.date.le(complete_date)
    # Descriptive analysis includes the observed final day; completeness is recorded separately.
    full=texts.copy()
    daily=texts.groupby(['date','source']).agg(n=('id','size'),tone=('tone','mean'),negative=('negative','mean'),risk=('risk_mention','mean'),existential=('existential_mention','mean')).reset_index()
    daily.to_csv(OUT/'calendar_daily.csv',index=False)
    source=full.groupby('source').agg(n=('id','size'),tone=('tone','mean'),negative=('negative','mean'),risk=('risk_mention','mean'),existential=('existential_mention','mean'))
    source.to_csv(OUT/'source_summary.csv')
    full['period']=np.select([full.date.le('2026-09-14'),full.date.le('2026-09-21')],['Sep 8-14','Sep 15-21'],default='Sep 22-'+observed_date[-2:])
    periods=full.groupby(['source','period']).agg(n=('id','size'),tone=('tone','mean'),negative=('negative','mean'),risk=('risk_mention','mean')).reset_index()
    periods.to_csv(OUT/'periods.csv',index=False)
    # Day-level differences respect common news cycles; individual comments are not independent observations.
    comparisons=[]
    for metric in ['tone','negative','risk','existential']:
        pivot=daily.pivot(index='date',columns='source',values=metric)
        for a,b in [('Guardian','TechCrunch'),('Hacker News','Guardian'),('Hacker News','TechCrunch')]:
            diff=(pivot[a]-pivot[b]).dropna()
            f=sm.OLS(diff.values,np.ones((len(diff),1))).fit(cov_type='HAC',cov_kwds={'maxlags':1},use_t=True)
            comparisons.append({'metric':metric,'a':a,'b':b,'days':len(diff),'difference':float(f.params[0]),'ci_low':float(f.conf_int()[0,0]),'ci_high':float(f.conf_int()[0,1]),'p':float(f.pvalues[0])})
    gaps=pd.DataFrame(comparisons)
    gaps['p_holm']=multipletests(gaps.p,method='holm')[1]
    gaps.to_csv(OUT/'source_gaps.csv',index=False)
    # Matched news URLs give a narrow within-story check; comments and headlines remain different genres.
    def urlkey(s):return re.sub(r'^https?://(?:www\.)?','',str(s)).split('?')[0].rstrip('/')
    news=full[full.kind.eq('news')].assign(urlkey=lambda d:d.url.map(urlkey))
    social=full[full.kind.eq('social')].assign(urlkey=lambda d:d.story_url.map(urlkey))
    thread=social.groupby('urlkey').agg(comment_tone=('tone','mean'),comment_negative=('negative','mean'),comments=('id','size'))
    paired=news.merge(thread,on='urlkey'); paired.to_csv(OUT/'matched_stories.csv',index=False)
    # Complete adjusted close observations only. The current day is not complete before 16:00 EDT.
    p=pd.read_csv(ROOT/'data'/'prices.csv').pivot(index='date',columns='symbol',values='adjusted_close').sort_index()
    complete_closes=[d for d in p.index if pd.Timestamp(d+' 16:00',tz=NY)<CUTOFF]
    p=p.loc[complete_closes]; assert p.notna().all().all()
    ret=p.pct_change(fill_method=None)*100
    ret['basket']=ret[BASKET].mean(axis=1)
    ret['basket_SPY']=ret.basket-ret.SPY; ret['basket_QQQ']=ret.basket-ret.QQQ
    ret=ret.loc['2026-09-08':'2026-09-30'].copy()
    close_grid=pd.DatetimeIndex([pd.Timestamp(d+' 16:00',tz=NY).tz_convert('UTC') for d in ret.index])
    # np.searchsorted assigns each post to the first close at or after publication.
    positions=close_grid.searchsorted(texts.created_utc)
    texts['trading_date']=[ret.index[i] if i<len(ret) else None for i in positions]
    texts.to_csv(OUT/'texts_scored.csv',index=False)
    win=texts.dropna(subset=['trading_date'])
    group=win.groupby(['trading_date','source']).tone.mean().unstack().reindex(ret.index)
    # Fixed 50/50 outlet weight prevents outlet-volume shifts from changing news composition.
    ret['news']=(group.Guardian+group.TechCrunch)/2
    ret['social']=group['Hacker News']
    ret['news_pooled']=win[win.kind.eq('news')].groupby('trading_date').tone.mean().reindex(ret.index)
    thread_day=win[win.kind.eq('social')].groupby(['trading_date','story_id']).tone.mean()
    ret['social_thread']=thread_day.groupby(level=0).mean().reindex(ret.index)
    hu=win.groupby(['trading_date','source']).hu_liu.mean().unstack().reindex(ret.index)
    strict=win[win.explicit_ai].groupby(['trading_date','source']).tone.mean().unstack().reindex(ret.index)
    ret['news_hu']=(hu.Guardian+hu.TechCrunch)/2;ret['social_hu']=hu['Hacker News']
    ret['news_strict']=(strict.Guardian+strict.TechCrunch)/2
    for c in ['news_hu','social_hu','news_strict']:ret['d_'+c]=ret[c].diff()
    ret['basket_no_META']=ret[[s for s in BASKET if s!='META']].mean(axis=1)
    exclusions=[]
    for symbol in BASKET:
        series=ret[[s for s in BASKET if s!=symbol]].mean(axis=1)
        exclusions.append({'excluded':symbol,'cumulative_return':float((1+series/100).prod()-1)})
    pd.DataFrame(exclusions).to_csv(OUT/'leave_one_out_returns.csv',index=False)
    for col in ['news','social','news_pooled','social_thread']:ret['d_'+col]=ret[col].diff()
    ret.to_csv(OUT/'trading_panel.csv')
    cum=(1+ret[BASKET+['SPY','QQQ','basket','basket_no_META']]/100).prod()-1
    cum.rename('cumulative_return').to_csv(OUT/'cumulative_returns.csv')
    tests=[]; design=[]
    for kind in ['news','social']:
        row,d,f=regression(ret.basket,ret['d_'+kind]); row.update(test=kind+' changes -> basket',family='primary')
        tests.append(row); d.assign(test=row['test'],date=d.index).to_csv(OUT/(kind+'_primary_design.csv'),index=False)
        for name,x,y,lag,control in [
            ('two lags',ret['d_'+kind],ret.basket,2,None),
            ('tone levels',ret[kind],ret.basket,1,None),
            ('SPY spread',ret['d_'+kind],ret.basket_SPY,1,None),
            ('QQQ spread',ret['d_'+kind],ret.basket_QQQ,1,None),
            ('lagged SPY control',ret['d_'+kind],ret.basket,1,ret.SPY),
            ('reverse',ret.basket,ret['d_'+kind],1,None),
            ('alternative weighting',ret['d_news_pooled' if kind=='news' else 'd_social_thread'],ret.basket,1,None)]:
            row,_,_=regression(y,x,lag,control); row.update(test=kind+': '+name,family='secondary'); tests.append(row)
        for name,x,y in [('Hu-Liu',ret['d_'+kind+'_hu'],ret.basket),('exclude META',ret['d_'+kind],ret.basket_no_META)]:
            row,_,_=regression(y,x);row.update(test=kind+': '+name,family='secondary');tests.append(row)
    row,_,_=regression(ret.basket,ret.d_news_strict);row.update(test='news: explicit AI headlines',family='secondary');tests.append(row)
    full.groupby('source').agg(n=('id','size'),explicit_ai_share=('explicit_ai','mean'),hu_liu=('hu_liu','mean'),hu_liu_negative=('hu_liu_negative','mean')).to_csv(OUT/'measurement_checks.csv')
    tests=pd.DataFrame(tests); mask=tests.family.eq('primary')
    for family,indices in tests.groupby('family').groups.items():
        tests.loc[indices,'p_holm']=multipletests(tests.loc[indices,'p'],method='holm')[1]
    tests.to_csv(OUT/'granger_tests.csv',index=False)
    diagnostic=[]
    for col in ['basket','news','social','d_news','d_social']:
        d=ret[col].dropna(); stat,pv,lag,n,*_=adfuller(d,maxlag=0,regression='c',autolag=None)
        diagnostic.append({'series':col,'ADF':stat,'p':pv,'n':n})
    pd.DataFrame(diagnostic).to_csv(OUT/'stationarity.csv',index=False)
    trend=[]
    for s,g in daily.groupby('source'):
        g=g.sort_values('date'); days=(pd.to_datetime(g.date)-pd.Timestamp('2026-09-08')).dt.days
        fit=sm.OLS(g.tone,sm.add_constant(days)).fit(cov_type='HAC',cov_kwds={'maxlags':1},use_t=True)
        trend.append({'source':s,'slope_per_day':fit.params.iloc[1],'p':fit.pvalues.iloc[1],'ci_low':fit.conf_int().iloc[1,0],'ci_high':fit.conf_int().iloc[1,1]})
    pd.DataFrame(trend).to_csv(OUT/'trends.csv',index=False)
    summary={'cutoff_utc':str(CUTOFF),'observed_calendar_end':observed_date,'full_window_elapsed':bool(CUTOFF>=END),'complete_calendar_end':complete_date,'complete_market_end':ret.index[-1],'trading_days':len(ret),'full_texts':len(full),'partial_texts':int((~texts.complete_day).sum()),'news_n':len(full[full.kind.eq('news')]),'social_n':len(social),'social_threads':int(social.story_id.nunique()),'social_authors':int(social.author.nunique()),'matched_stories':len(paired),'paired_tone_gap':float((paired.comment_tone-paired.tone).mean()),'after_final_close_texts':int(texts.trading_date.isna().sum()),'returns':cum.to_dict(),'primary':tests[mask].to_dict('records'),'sources':source.reset_index().to_dict('records')}
    summary['model']=ARGS.model
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'runtime.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,'packages':installed_versions(['pandas','numpy','scipy','statsmodels','vaderSentiment','requests','beautifulsoup4','reportlab','matplotlib','pymupdf','nbformat','nbclient'])},indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
