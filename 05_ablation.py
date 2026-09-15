"""Run controlled YOLO ablation experiments."""
from pathlib import Path
import argparse,csv
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data.yaml'); p.add_argument('--epochs',type=int,default=50); p.add_argument('--device',default='0'); a=p.parse_args()
    experiments=[('baseline_n640','yolo11n.pt',640,16),('small_s640','yolo11s.pt',640,16),('baseline_n960','yolo11n.pt',960,8)]
    rows=[]
    for name,model_name,imgsz,batch in experiments:
        print(f'\n=== {name} ===')
        m=YOLO(model_name)
        m.train(data=a.data,epochs=a.epochs,imgsz=imgsz,batch=batch,device=a.device,project='runs/ablation',name=name,plots=True)
        r=m.val(data=a.data,split='val',imgsz=imgsz,batch=batch,device=a.device,verbose=False)
        rows.append({'experiment':name,'model':model_name,'imgsz':imgsz,'batch':batch,'mAP50':float(r.box.map50),'mAP50-95':float(r.box.map),'precision':float(r.box.mp),'recall':float(r.box.mr)})
    out=Path('runs/ablation/ablation_results.csv'); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f'Results: {out}')

if __name__=='__main__': main()
