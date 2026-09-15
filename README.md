# Top-View Vehicle Detection

AI-based vehicle detection from top-view, aerial, drone, and UAV imagery.

## Objective

Detect vehicles and produce bounding boxes, class labels, confidence scores, evaluation metrics, and deployment benchmarks.

Initial dataset configuration uses one class: `vehicle`. Expand `data.yaml` when the dataset contains multiple vehicle classes.

## Project structure

```text
vehicle-detection/
├── data.yaml
├── 01_check_dataset.py
├── 02_train.py
├── 03_evaluate.py
├── 04_kfold_cv.py
├── 05_ablation.py
├── 06_compare_runs.py
├── 07_benchmark_export.py
├── 08_predict_visual.py
├── requirements.txt
├── environment.yml
├── .gitignore
├── README.md
├── datasets/              # local only, ignored by Git
│   └── vehicle/
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
└── runs/                  # auto-created, ignored by Git
```

## Dataset format

YOLO detection format:

```text
class_id x_center y_center width height
```

Coordinates are normalized from 0 to 1.

## Environment

Using the existing AI environment:

```powershell
conda run -n uav-ai python -m pip install -r requirements.txt
```

Or create the dedicated environment:

```powershell
conda env create -f environment.yml
```

## Workflow

```text
01_check_dataset.py
        ↓
02_train.py
        ↓
03_evaluate.py
        ↓
04_kfold_cv.py / 05_ablation.py
        ↓
06_compare_runs.py
        ↓
07_benchmark_export.py
        ↓
08_predict_visual.py
```

## Commands

```powershell
python 01_check_dataset.py
python 02_train.py
python 03_evaluate.py
python 04_kfold_cv.py
python 05_ablation.py
python 06_compare_runs.py
python 07_benchmark_export.py --weights runs/detect/vehicle_baseline/weights/best.pt
python 08_predict_visual.py --source path/to/image.jpg
```

All generated training outputs and model weights are excluded from Git. The repository is intended to contain reproducible source code and configuration, while datasets and large artifacts remain local or are stored separately.

## Future direction

The detector is intended to become a reusable UAV computer-vision component for real-time image/video inference, tracking, TensorRT/ONNX deployment, and eventual multi-UAV/swarm integration.
