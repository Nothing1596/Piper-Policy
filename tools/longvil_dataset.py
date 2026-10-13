"""Pinned LongVIL candidate ordering and review material; never approves labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil
import urllib.request

REVISION='59f002209575ae1b3e619a2ffd5a831a15194c2a'
EXCLUDED={'level1_4_blockmanipulation_1','level2_6_blockmanipulation_1','level3_12_blockmanipulation_1','level1_1_tabletopcleanup_1'}
CATEGORIES=('blockmanipulation','tabletopcleanup','vegetablesorting')


def fetch(url):
    with urllib.request.urlopen(url,timeout=60) as response:return response.read()


def catalog(root):
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for level in ('level1','level2','level3'):
        url=f'https://huggingface.co/api/datasets/cq838/LongVILBench/tree/{REVISION}/data/{level}'
        rows.extend((level,r['path']) for r in json.loads(fetch(url)) if r['type']=='directory')
    def annotation(item):
        level,path=item;name=path.rsplit('/',1)[-1]
        url=f'https://huggingface.co/datasets/cq838/LongVILBench/resolve/{REVISION}/{path}/{name}.json'
        try:
            raw=fetch(url);data=json.loads(raw)
            folder=root/'annotations';folder.mkdir(exist_ok=True)
            file=folder/(name+'.json');file.write_bytes(raw)
            category=''.join(c for c in str(data.get('class','')).lower() if c.isalnum())
            return {'id':name,'source_path':path,'level':str(data.get('level')),'category':category,
                    'annotation':str(file),'sha256':hashlib.sha256(raw).hexdigest(),'url':url,
                    'eligible_metadata':name not in EXCLUDED and category in CATEGORIES and str(data.get('level')) in ('1','2','3')}
        except Exception as error:return {'id':name,'error_type':type(error).__name__,'eligible_metadata':False}
    results=list(ThreadPoolExecutor(6).map(annotation,rows))
    strata={}
    for level in ('1','2','3'):
        for category in CATEGORIES:
            key=level+'/'+category
            selected=[r for r in results if r.get('eligible_metadata') and r['level']==level and r['category']==category]
            selected.sort(key=lambda r:r['id']);random.Random('20261010/'+key).shuffle(selected)
            strata[key]=[r['id'] for r in selected]
    result={'dataset_revision':REVISION,'seed':20261010,'excluded':sorted(EXCLUDED),'strata':strata,'tasks':results}
    (root/'catalog.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({key:len(ids) for key,ids in strata.items()}),flush=True)


def prepare(root,count):
    catalog_data=json.loads((root/'catalog.json').read_text())
    indexed={r['id']:r for r in catalog_data['tasks']}
    ids=[ident for names in catalog_data['strata'].values() for ident in names[:count]]
    if shutil.disk_usage(root).free<8*1024**3:raise RuntimeError('At least 8 GiB free space required for review preparation')
    def download(ident):
        item=indexed[ident];folder=root/'data'/ident;folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(item['annotation'],folder/(ident+'.json'))
        files=[]
        for variant in ('clean','complex'):
            name=ident+'_'+variant+'.mp4';url=f'https://huggingface.co/datasets/cq838/LongVILBench/resolve/{REVISION}/{item["source_path"]}/{name}'
            dest=folder/name
            if not dest.exists():
                with urllib.request.urlopen(url,timeout=90) as response,dest.with_suffix('.part').open('wb') as out:
                    total=0
                    while chunk:=response.read(1024*1024):
                        total+=len(chunk)
                        if total>512*1024**2:raise RuntimeError('Review video exceeds 512 MiB')
                        out.write(chunk)
                dest.with_suffix('.part').replace(dest)
            files.append({'variant':variant,'path':str(dest),'url':url,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size,'review_status':'pending'})
        print('downloaded '+ident,flush=True)
        return {**item,'videos':files,'review_status':'pending'}
    selected=list(ThreadPoolExecutor(4).map(download,ids))
    (root/'review-manifest.json').write_text(json.dumps({'dataset_revision':REVISION,'seed':20261010,'tasks':selected,'approval':'pending','formal_submission_allowed':False},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['catalog','prepare']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--per-stratum',type=int,default=3)
    args=parser.parse_args()
    catalog(args.root) if args.operation=='catalog' else prepare(args.root,args.per_stratum)
