"""Evaluate a trained YOLO detector."""
from pathlib import Path
import argparse
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',default='runs/detect/vehicle_baseline/weights/best.pt')
    p.add_argument('--data',default='data.yaml')
    p.add_argument('--split',default='test',choices=['train','val','test'])
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--device',default='0')
    p.add_argument('--project',default='runs/evaluate')
    p.add_argument('--name',default='vehicle_evaluation')
    a=p.parse_args()
    if not Path(a.weights).exists(): raise FileNotFoundError(a.weights)
    m=YOLO(a.weights)
    metrics=m.val(data=a.data,split=a.split,imgsz=a.imgsz,batch=a.batch,device=a.device,project=a.project,name=a.name,plots=True)
    print(f'mAP50={metrics.box.map50:.4f}')
    print(f'mAP50-95={metrics.box.map:.4f}')
    print(f'Precision={metrics.box.mp:.4f}')
    print(f'Recall={metrics.box.mr:.4f}')

if __name__=='__main__': main()
