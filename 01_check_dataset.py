"""Validate a YOLO detection dataset."""
from pathlib import Path
import argparse
import yaml
from PIL import Image

EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data.yaml')
    args=p.parse_args()
    cfg_path=Path(args.data).resolve()
    cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    root=Path(cfg.get('path','.'))
    root=(cfg_path.parent/root).resolve() if not root.is_absolute() else root.resolve()
    nc=int(cfg.get('nc',len(cfg.get('names',[]))))
    problems=0
    print(f'Dataset: {root}')
    print(f'Classes: {cfg.get("names",{})}')
    for split in ('train','val','test'):
        img_dir=root/'images'/split; lbl_dir=root/'labels'/split
        images=[x for x in img_dir.rglob('*') if x.is_file() and x.suffix.lower() in EXTS] if img_dir.exists() else []
        image_stems={x.stem for x in images}
        objects=0
        for image in images:
            try:
                with Image.open(image) as im: im.verify()
            except Exception as e:
                print(f'INVALID IMAGE: {image} ({e})'); problems+=1
            label=lbl_dir/f'{image.stem}.txt'
            if not label.exists():
                print(f'MISSING LABEL: {image}'); problems+=1; continue
            for n,line in enumerate(label.read_text(encoding='utf-8').splitlines(),1):
                if not line.strip(): continue
                v=line.split()
                if len(v)!=5:
                    print(f'BAD LABEL: {label}:{n}'); problems+=1; continue
                try: cls=int(float(v[0])); vals=[float(x) for x in v[1:]]
                except ValueError:
                    print(f'NON-NUMERIC LABEL: {label}:{n}'); problems+=1; continue
                if not 0<=cls<nc or any(x<0 or x>1 for x in vals) or vals[2]<=0 or vals[3]<=0:
                    print(f'INVALID BOX: {label}:{n}'); problems+=1
                objects+=1
        labels=list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []
        for label in labels:
            if label.stem not in image_stems:
                print(f'ORPHAN LABEL: {label}'); problems+=1
        print(f'{split}: {len(images)} images, {objects} objects')
    print('DATASET CHECK: PASSED' if problems==0 else f'DATASET CHECK: {problems} problem(s) found')

if __name__=='__main__': main()
