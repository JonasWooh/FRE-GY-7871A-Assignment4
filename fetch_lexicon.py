"""Download the published Hu-Liu opinion lexicon distributed by NLTK."""
from io import BytesIO
from zipfile import ZipFile
from fetch_data import ROOT,get
p=ROOT/'data'/'lexicon';p.mkdir(exist_ok=True)
z=ZipFile(BytesIO(get('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/opinion_lexicon.zip')))
for word in ['negative','positive']:
    (p/(word+'.txt')).write_bytes(z.read('opinion_lexicon/'+word+'-words.txt'))
