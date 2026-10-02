"""Census of TechCrunch's AI category in the study interval, via publisher API."""
import json
import pandas as pd
from bs4 import BeautifulSoup
from fetch_data import ROOT,START,END,get

cats=json.loads(get('https://techcrunch.com/wp-json/wp/v2/categories',{'slug':'artificial-intelligence'}))
assert len(cats)==1,cats
rows=[]
for page in range(1,100):
    # API boundaries here are UTC, and exact New York boundaries are filtered below.
    posts=json.loads(get('https://techcrunch.com/wp-json/wp/v2/posts',{'categories':cats[0]['id'],'after':'2026-09-07T00:00:00','before':'2026-10-01T04:00:00','per_page':100,'page':page,'orderby':'date','order':'asc'}))
    for p in posts:
        ts=pd.Timestamp(p['date_gmt'],tz='UTC')
        if START<=ts<END:
            rows.append({'id':p['link'],'url':p['link'],'created_utc':ts.isoformat(),'text':BeautifulSoup(p['title']['rendered'],'html.parser').get_text(' ',strip=True),'source':'TechCrunch','kind':'news','story_id':None})
    print(page,len(posts),len(rows),flush=True)
    if len(posts)<100:break
else:raise ValueError('Pagination incomplete')
pd.DataFrame(rows).drop_duplicates('id').to_csv(ROOT/'data'/'techcrunch.csv',index=False)
