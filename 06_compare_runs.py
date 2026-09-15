"""Compare completed Ultralytics results.csv files."""
from pathlib import Path
import argparse,csv

def num(row,*keys):
    for k in keys:
        if k in row:
            try:return float(row[k])
            except (TypeError,ValueError):pass
    return None

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='runs'); p.add_argument('--output',default='runs/run_comparison.csv'); a=p.parse_args()
    files=list(Path(a.root).rglob('results.csv'))
    rows=[]
    for f in files:
        data=list(csv.DictReader(f.open(encoding='utf-8')))
        if not data: continue
        r=data[-1]; rows.append({'run':str(f.parent),'epoch':num(r,'epoch'),'precision':num(r,'metrics/precision(B)'), 'recall':num(r,'metrics/recall(B)'),'mAP50':num(r,'metrics/mAP50(B)'),'mAP50-95':num(r,'metrics/mAP50-95(B)')})
    rows.sort(key=lambda x:x['mAP50-95'] if x['mAP50-95'] is not None else -1,reverse=True)
    for i,r in enumerate(rows,1): print(f"{i}. {r['run']} | mAP50={r['mAP50']} | mAP50-95={r['mAP50-95']} | P={r['precision']} | R={r['recall']}")
    if rows:
        out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print(f'Comparison: {out}')
    else: print('No results.csv files found.')

if __name__=='__main__': main()
