"""Acquire timestamped public observations and immutable raw-response snapshots."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json, re, time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data'/'raw'
RAW.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2026-09-08',tz='America/New_York')
END=pd.Timestamp('2026-10-01',tz='America/New_York')
AI=re.compile(r'\b(?:AI|AGI|LLMs?|ChatGPT|OpenAI|Anthropic|Claude|artificial intelligence|large language models?|superintelligen\w*)\b',re.I)
ACQUISITION=ROOT/'data'/'acquisition.json'

def get(url,params=None):
    req=requests.Request('GET',url,params=params).prepare()
    key=hashlib.sha256(req.url.encode()).hexdigest()
    p=RAW/(key+'.body')
    if not p.exists():
        r=requests.get(req.url,headers={'User-Agent':'Mozilla/5.0 (academic research)'},timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
        (RAW/(key+'.json')).write_text(json.dumps({'url':r.url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(r.content).hexdigest(),'status':r.status_code},indent=2),encoding='utf-8')
    return p.read_bytes()

def guardian():
    rows=[]; coverage=[]
    for page in range(1,101):
        soup=BeautifulSoup(get('https://www.theguardian.com/technology/artificialintelligenceai',{'page':page}),'html.parser')
        stamps=[]
        for h in soup.select('h3.card-headline'):
            li=h.find_parent('li'); link=li.find('a',href=True,attrs={'aria-label':True}); stamp=li.find('time',datetime=True)
            if link is None or stamp is None: continue
            ts=pd.Timestamp(stamp['datetime']); stamps.append(ts)
            if START<=ts<END:
                headline=h.select_one('.headline-text')
                if headline is None: raise ValueError('Missing Guardian headline element')
                url=urljoin('https://www.theguardian.com',link['href'])
                rows.append({'id':url,'url':url,'created_utc':ts.isoformat(),'text':headline.get_text(' ',strip=True),'source':'Guardian','kind':'news','story_id':None})
        if not stamps: raise ValueError(f'Guardian page {page} has no dated cards')
        coverage.append({'page':page,'first':str(min(stamps)),'last':str(max(stamps))})
        print('Guardian',page,len(rows),min(stamps),flush=True)
        if min(stamps)<START: break
    else: raise ValueError('Archive did not reach window start')
    pd.DataFrame(rows).drop_duplicates('id').to_csv(ROOT/'data'/'news.csv',index=False)
    (ROOT/'data'/'news_coverage.json').write_text(json.dumps(coverage,indent=2))

def hn_query(query,lo,hi):
    params={'query':query,'tags':'comment','numericFilters':f'created_at_i>={lo},created_at_i<{hi}','hitsPerPage':1000,'page':0}
    d=json.loads(get('https://hn.algolia.com/api/v1/search_by_date',params))
    if d['nbHits']>=1000:
        if hi-lo<=1: raise ValueError('HN result cap cannot be resolved')
        mid=(lo+hi)//2
        return hn_query(query,lo,mid)+hn_query(query,mid,hi)
    if len(d['hits'])!=d['nbHits']: raise ValueError('Incomplete HN page')
    return d['hits']

def social():
    queries=['AI','artificial intelligence','LLM','ChatGPT','OpenAI','Anthropic','Claude','AGI','superintelligence']
    hi=int(pd.Timestamp(json.loads(ACQUISITION.read_text())['text_cutoff_utc']).timestamp())
    def task(q):
        h=hn_query(q,int(START.timestamp()),hi)
        print('HN',q,len(h),flush=True)
        return h
    with ThreadPoolExecutor(max_workers=4) as ex: hits=[h for chunk in ex.map(task,queries) for h in chunk]
    rows=[]
    for h in hits:
        txt=BeautifulSoup(h.get('comment_text') or '', 'html.parser').get_text(' ',strip=True)
        if not AI.search(txt): continue
        rows.append({'id':str(h['objectID']),'url':'https://news.ycombinator.com/item?id='+str(h['objectID']),'created_utc':h['created_at'],'text':txt,'source':'Hacker News','kind':'social','story_id':h['story_id'],'story_title':h.get('story_title'),'story_url':h.get('story_url'),'author':h.get('author')})
    df=pd.DataFrame(rows).drop_duplicates('id')
    df.to_csv(ROOT/'data'/'social.csv',index=False)
    print('HN unique',len(df),flush=True)

def prices():
    rows=[]
    for symbol in ['NVDA','MSFT','AMZN','GOOGL','META','AVGO','SPY','QQQ']:
        d=json.loads(get(f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',{'period1':int(pd.Timestamp('2026-09-04',tz='UTC').timestamp()),'period2':int(pd.Timestamp('2026-10-01',tz='UTC').timestamp()),'interval':'1d'}))['chart']['result'][0]
        q=d['indicators']['quote'][0]; a=d['indicators']['adjclose'][0]['adjclose']
        for i,t in enumerate(d['timestamp']):
            ts=pd.Timestamp(t,unit='s',tz='UTC'); date=ts.tz_convert('America/New_York').strftime('%Y-%m-%d')
            rows.append({'date':date,'symbol':symbol,'adjusted_close':a[i],'close':q['close'][i],'timestamp_utc':ts.isoformat()})
        print(symbol,len(d['timestamp']),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'data'/'prices.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('part',choices=['news','social','prices','all']); args=parser.parse_args()
    if args.part=='all':
        cutoff=min(pd.Timestamp.now(tz='UTC').floor('s'),END).tz_convert('UTC')
        ACQUISITION.write_text(json.dumps({'last_run_utc':cutoff.isoformat(),'text_cutoff_utc':cutoff.isoformat(),'window_start':START.isoformat(),'window_end_exclusive':END.isoformat(),'calendar_day_policy':'Include the observed portion of September 30; distinguish observed from complete days.'},indent=2))
    for name,fn in [('news',guardian),('social',social),('prices',prices)]:
        if args.part in [name,'all']: fn()
    if args.part=='all':
        meta=json.loads(ACQUISITION.read_text());meta['collection_finished_utc']=datetime.now(timezone.utc).isoformat()
        ACQUISITION.write_text(json.dumps(meta,indent=2))
