"""Train the YOLO vehicle detector."""
from pathlib import Path
import argparse
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data.yaml')
    p.add_argument('--model',default='yolo11n.pt')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--device',default='0')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--project',default='runs/detect')
    p.add_argument('--name',default='vehicle_baseline')
    p.add_argument('--patience',type=int,default=30)
    a=p.parse_args()
    if not Path(a.data).exists(): raise FileNotFoundError(a.data)
    model=YOLO(a.model)
    model.train(data=a.data,epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,device=a.device,workers=a.workers,project=a.project,name=a.name,patience=a.patience,pretrained=True,plots=True,save=True)
    best=Path(a.project)/a.name/'weights'/'best.pt'
    print(f'Best model: {best}')

if __name__=='__main__': main()
