"""Run visual vehicle detection on an image, folder, video, or other Ultralytics source."""
from pathlib import Path
import argparse
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser(); p.add_argument('--weights',default='runs/detect/vehicle_baseline/weights/best.pt'); p.add_argument('--source',required=True); p.add_argument('--imgsz',type=int,default=640); p.add_argument('--conf',type=float,default=0.25); p.add_argument('--iou',type=float,default=0.7); p.add_argument('--device',default='0'); p.add_argument('--project',default='runs/predict'); p.add_argument('--name',default='vehicle_visual'); p.add_argument('--save-txt',action='store_true'); p.add_argument('--save-conf',action='store_true'); a=p.parse_args()
    if not Path(a.weights).exists(): raise FileNotFoundError(a.weights)
    if not Path(a.source).exists(): raise FileNotFoundError(a.source)
    m=YOLO(a.weights)
    results=m.predict(source=a.source,imgsz=a.imgsz,conf=a.conf,iou=a.iou,device=a.device,project=a.project,name=a.name,save=True,save_txt=a.save_txt,save_conf=a.save_conf,exist_ok=True)
    total=sum(len(r.boxes) for r in results if r.boxes is not None)
    print(f'Results: {a.project}/{a.name}'); print(f'Detections: {total}')

if __name__=='__main__': main()
