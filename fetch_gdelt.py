"""GDELT DOC 2.0 aggregates for English-language web news, used for the US versus non-US comparison.

Queries (2026-09-08 04:00 to 2026-10-01 04:00 UTC):
  ai   = "artificial intelligence" sourcelang:english
  risk = "artificial intelligence" (extinction OR existential OR superintelligence OR "kill all humans") sourcelang:english
each for all sources, sourcecountry:US and -sourcecountry:US, in modes tonechart (article tone histogram),
timelinevolraw (daily article counts) and timelinetone. GDELT asks for at most one request every 5 seconds;
the API also returns a rate-limit notice under load, so failed calls are retried with long pauses.
Writes data/gdelt_refresh.json; the retained snapshot is data/gdelt.json.
"""
import json, time
import requests
from fetch_data import ROOT

BASE = 'https://api.gdeltproject.org/api/v2/doc/doc'
R = '(extinction OR existential OR superintelligence OR "kill all humans")'
Q = {'ai_all': '"artificial intelligence" sourcelang:english', 'ai_us': '"artificial intelligence" sourcelang:english sourcecountry:US',
     'ai_nonus': '"artificial intelligence" sourcelang:english -sourcecountry:US', 'risk_all': f'"artificial intelligence" {R} sourcelang:english',
     'risk_us': f'"artificial intelligence" {R} sourcelang:english sourcecountry:US', 'risk_nonus': f'"artificial intelligence" {R} sourcelang:english -sourcecountry:US'}

if __name__ == '__main__':
    out = {'tonechart': {}, 'timelinevolraw': {}, 'timelinetone': {}}
    for key, q in Q.items():
        for mode in out:
            for attempt in range(5):
                time.sleep(30)
                r = requests.get(BASE, params={'query': q, 'mode': mode, 'format': 'json', 'startdatetime': '20260908040000', 'enddatetime': '20261001040000', 'timelinesmooth': 0}, timeout=120)
                try: j = r.json()
                except ValueError: continue
                if mode == 'tonechart': out[mode][key] = [[b['bin'], b['count']] for b in j['tonechart']]
                else: out[mode][key] = [[d['date'], d['value'], d.get('norm')] for d in j['timeline'][0]['data']]
                break
    (ROOT/'data'/'gdelt_refresh.json').write_text(json.dumps(out, indent=2)); print('saved')
