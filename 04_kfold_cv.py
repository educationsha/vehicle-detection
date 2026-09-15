"""K-fold image-level cross-validation starter for YOLO detection."""
from pathlib import Path
import argparse, shutil, yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO

EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def link_or_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    try: dst.hardlink_to(src)
    except (OSError,NotImplementedError): shutil.copy2(src,dst)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data.yaml'); p.add_argument('--folds',type=int,default=5)
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=16); p.add_argument('--model',default='yolo11n.pt')
    p.add_argument('--device',default='0'); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    cfg=yaml.safe_load(Path(a.data).read_text(encoding='utf-8'))
    root=Path(cfg.get('path','.')); root=(Path(a.data).resolve().parent/root).resolve() if not root.is_absolute() else root.resolve()
    samples=[]
    for im in sorted((root/'images'/'train').glob('*')):
        if im.suffix.lower() in EXTS:
            lab=root/'labels'/'train'/f'{im.stem}.txt'
            if lab.exists(): samples.append((im,lab))
    if len(samples)<a.folds: raise ValueError('Not enough labeled images for requested folds')
    kf=KFold(n_splits=a.folds,shuffle=True,random_state=a.seed)
    for fold,(tr,va) in enumerate(kf.split(samples),1):
        base=Path('runs/kfold_dataset')/f'fold_{fold}'
        if base.exists(): shutil.rmtree(base)
        for split,idxs in [('train',tr),('val',va)]:
            for i in idxs:
                im,lab=samples[i]; link_or_copy(im,base/'images'/split/im.name); link_or_copy(lab,base/'labels'/split/lab.name)
        fold_yaml=base/'data.yaml'
        fold_cfg={'path':str(base.resolve()),'train':'images/train','val':'images/val','nc':cfg.get('nc',1),'names':cfg.get('names',{0:'vehicle'})}
        fold_yaml.write_text(yaml.safe_dump(fold_cfg,sort_keys=False),encoding='utf-8')
        m=YOLO(a.model)
        m.train(data=str(fold_yaml),epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,device=a.device,project='runs/kfold',name=f'fold_{fold}',plots=True)
        r=m.val(data=str(fold_yaml),split='val',imgsz=a.imgsz,batch=a.batch,device=a.device,verbose=False)
        print(f'Fold {fold}: mAP50={r.box.map50:.4f}, mAP50-95={r.box.map:.4f}')

if __name__=='__main__': main()
