"""Published Cardiff Twitter-RoBERTa model, pinned ONNX export; no fine-tuning.
Use the model-card URL/username preprocessing and a documented 512-token limit.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from scipy.special import softmax
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from fetch_data import ROOT

MODEL='Xenova/twitter-roberta-base-sentiment-latest'
REV='f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445'

def preprocess(text):
    return ' '.join('@user' if t.startswith('@') and len(t)>1 else 'http' if t.startswith('http') else t for t in text.split(' '))

def main():
    path=Path(snapshot_download(MODEL,revision=REV,allow_patterns=['*.json','*.txt','onnx/model_quantized.onnx'],local_dir=ROOT/'models'/'twitter_roberta'))
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    options=ort.SessionOptions();options.intra_op_num_threads=6
    session=ort.InferenceSession(str(path/'onnx'/'model_quantized.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    config=json.loads((path/'config.json').read_text());print('labels',config['id2label'],flush=True)
    assert [config['id2label'][str(i)].lower() for i in range(3)]==['negative','neutral','positive']
    d=pd.concat([pd.read_csv(ROOT/'data'/f,dtype={'id':str}) for f in ['news.csv','techcrunch.csv','social.csv']],ignore_index=True)
    d['text_hash']=d.text.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    cache=ROOT/'data'/'contextual_scores.csv'
    if cache.exists():
        prior=json.loads((ROOT/'data'/'contextual_model.json').read_text())
        assert prior['model']==MODEL and prior['revision']==REV and prior['max_tokens']==512
        assert prior['model_sha256']==hashlib.sha256((path/'onnx'/'model_quantized.onnx').read_bytes()).hexdigest()
    old=pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=['text_hash','p_negative','p_neutral','p_positive','tokens','truncated'])
    todo=d[~d.text_hash.isin(old.text_hash)].drop_duplicates('text_hash').copy()
    todo['prepared']=todo.text.map(preprocess)
    encoded=tokenizer(todo.prepared.tolist(),truncation=False,padding=False,verbose=False)
    todo['encoded']=encoded['input_ids'];todo['tokens']=todo.encoded.map(len)
    todo=todo.sort_values('tokens')
    # Note: the INT8 export quantizes activations dynamically per call, so probabilities depend slightly on
    # batch composition (padding). Cached scores were produced with length-sorted batches of 16.
    rows=[];start=time.monotonic()
    for offset in range(0,len(todo),16):
        batch=todo.iloc[offset:offset+16]
        inputs=tokenizer(batch.prepared.tolist(),truncation=True,max_length=512,padding=True,return_tensors='np')
        probs=softmax(session.run(None,{x.name:inputs[x.name].astype(np.int64) for x in session.get_inputs()})[0],axis=1)
        for (_,r),p in zip(batch.iterrows(),probs):rows.append({'text_hash':r.text_hash,'p_negative':float(p[0]),'p_neutral':float(p[1]),'p_positive':float(p[2]),'tokens':r.tokens,'truncated':r.tokens>512})
        if offset%1600==0:print('scored',offset+len(batch),'/',len(todo),'seconds',round(time.monotonic()-start,1),flush=True)
    scores=pd.concat([old,pd.DataFrame(rows)],ignore_index=True).drop_duplicates('text_hash')
    scores.to_csv(cache,index=False)
    meta={'model':MODEL,'revision':REV,'original_model':'cardiffnlp/twitter-roberta-base-sentiment-latest','model_sha256':hashlib.sha256((path/'onnx'/'model_quantized.onnx').read_bytes()).hexdigest(),'format':'published INT8 quantized ONNX','max_tokens':512,'positive_score':'P(positive)-P(negative)','preprocessing':'official model-card username and URL placeholders','n_cached':len(scores)}
    (ROOT/'data'/'contextual_model.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':
    main()
