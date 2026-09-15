"""Benchmark inference and optionally export a trained YOLO model."""
from pathlib import Path
import argparse,time
import cv2
import numpy as np
from ultralytics import YOLO

def images(source):
    exts={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
    p=Path(source)
    if p.is_file(): return [p]
    return [x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in exts]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--weights',default='runs/detect/vehicle_baseline/weights/best.pt'); p.add_argument('--source',default='datasets/vehicle/images/test'); p.add_argument('--imgsz',type=int,default=640); p.add_argument('--device',default='0'); p.add_argument('--warmup',type=int,default=10); p.add_argument('--iterations',type=int,default=100); p.add_argument('--export',choices=['none','onnx','engine'],default='none'); a=p.parse_args()
    if not Path(a.weights).exists(): raise FileNotFoundError(a.weights)
    ims=images(a.source)
    if not ims: raise RuntimeError('No benchmark images found')
    m=YOLO(a.weights); frame=cv2.imread(str(ims[0]))
    for _ in range(a.warmup): m.predict(frame,imgsz=a.imgsz,device=a.device,verbose=False)
    times=[]
    for i in range(a.iterations):
        frame=cv2.imread(str(ims[i%len(ims)])); start=time.perf_counter(); m.predict(frame,imgsz=a.imgsz,device=a.device,verbose=False); times.append(time.perf_counter()-start)
    avg=float(np.mean(times)); print(f'Average latency: {avg*1000:.2f} ms'); print(f'Median latency: {np.median(times)*1000:.2f} ms'); print(f'FPS: {1/avg:.2f}')
    if a.export!='none': print('Exported:',m.export(format=a.export,imgsz=a.imgsz,device=a.device))

if __name__=='__main__': main()
