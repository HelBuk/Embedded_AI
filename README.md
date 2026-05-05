# Embedded AI — Conveyor Belt Screw Detection on Raspberry Pi 5 + Hailo-8

Semester project at NTNU (2025)

The goal is to evaluate real-time object detection of screws, nuts, and washers on a conveyor belt using an **IDS uEye industrial camera**, **Raspberry Pi 5**, and the **Hailo-8 AI accelerator**. Several YOLO variants (v8/v10/v11) and RT-DETR are compared, then the best candidates are compiled to Hailo's `.hef` format for edge inference.

---

## Hardware

| Component | Details |
|---|---|
| SBC | Raspberry Pi 5 |
| AI accelerator | Hailo-8 (M.2 HAT, PCIe) |
| Camera | IDS uEye industrial camera |
| Host (training) | PC with CUDA 12.1 GPU |

---

## Project structure

```
.
├── capture_pictures.py       # Capture stills from uEye camera
├── capture_video.py          # Capture video + extract frames for labelling
├── upload_to_roboflow.py     # Push frames + YOLO labels to Roboflow
├── download_from_roboflow.py # Pull annotated dataset from Roboflow
├── train_yolov8.py           # Training entry-point (wraps Ultralytics CLI)
├── calibration_data.py       # Build calibration .npy for Hailo quantization
├── ueye_yolo_rt.py           # Real-time inference via Ultralytics (no Hailo)
├── hailo_yolo_track.py       # Real-time inference via Hailo .hef
├── rtdetr_speed.py           # RT-DETR CPU benchmark
├── rtdetr_prediction.py      # RT-DETR inference demo
├── model_params_comparison.py# Plot params / GFLOPs / speed across models
├── plot_stats.py             # Plot RT-DETR training curves
├── requirements.txt
├── setup.sh
├── data.dvc                  # DVC pointer to dataset (235 MB, 232 files)
├── calib/                    # Calibration image lists
├── runs/                     # Ultralytics training outputs + compiled .hef files
└── report/                   # LaTeX source + compiled PDF report
```

---

## Installation

### System dependencies (Raspberry Pi / Debian)

```bash
sudo apt install python3-venv
```

For Hailo SDK, follow the [Hailo developer zone](https://hailo.ai/developer-zone/) and install `hailo_platform` separately — it is not in `requirements.txt`.

### Python environment

```bash
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a `.venv`, upgrades pip, and installs `requirements.txt`. If `python3-venv` is unavailable it falls back to `virtualenv`.

### GPU training dependencies (host PC)

PyTorch with CUDA 12.1 is listed in `requirements.txt`:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Workflow

### 1 — Data capture

Capture images with the uEye camera for manual annotation:

```bash
python capture_pictures.py   # stills → data/captures/<date>/runXX/
python capture_video.py      # video + per-frame JPEG export
```

Both scripts expose controls for exposure, gain, AOI (Area of Interest), white balance, and gamma.

### 2 — Annotation

Upload captured frames to Roboflow for labelling:

```bash
python upload_to_roboflow.py
```

Classes: **screw**, **nut**, **washer** (and optional additional classes).

### 3 — Dataset download

```bash
python download_from_roboflow.py
```

Downloads the annotated dataset in YOLOv8 format. The raw data is also tracked with DVC (`data.dvc`).

### 4 — Training

Training is driven by the Ultralytics CLI (results land in `runs/detect/`):

```bash
yolo detect train model=yolov8n.pt data=<dataset.yaml> imgsz=960 epochs=100
```

Trained model variants:

| Model | Size | Notes |
|---|---|---|
| YOLOv8n | nano | baseline |
| YOLOv8s | small | |
| YOLOv10n | nano | |
| YOLOv10s | small | |
| YOLOv11n | nano | compiled to .hef |
| YOLOv11s | small | |
| RT-DETR-L | large | CPU benchmark only |

### 5 — Model benchmarking

```bash
python model_params_comparison.py   # params / GFLOPs / speed comparison plots
python rtdetr_speed.py              # RT-DETR pre/inf/post timing on CPU
```

### 6 — Hailo quantization & compilation

Prepare calibration data (NHWC float32 numpy array):

```bash
python calibration_data.py
```

Then compile on a host with the Hailo Dataflow Compiler:

```bash
hailomz compile --hw-arch hailo8 --calib-path calib/run_v7-yolov8/calib_nhwc_960.npy \
    --ckpt runs/detect/<run>/weights/best.pt
```

Compiled `.hef` files are stored alongside their source checkpoints in `runs/detect/*/weights/`.

### 7 — Real-time inference

**Without Hailo** (Ultralytics, runs on any machine with the uEye SDK):

```bash
python ueye_yolo_rt.py
```

**With Hailo-8** (requires `hailo_platform`, run on Raspberry Pi 5):

```bash
python hailo_yolo_track.py
```

Edit the `HEF_PATH`, `VIDEO_IN`, and `VIDEO_OUT` constants at the top of the script to point to your paths.

---

## Results

See the full report: [`report/out/EmbeddedAI_Project_Bukueva_Elena_2025.pdf`](report/out/EmbeddedAI_Project_Bukueva_Elena_2025.pdf)

Key findings:
- YOLOv11n compiled to Hailo `.hef` achieves real-time inference on Raspberry Pi 5 at 960×960.
- YOLOv8n/s and YOLOv10n/s also successfully compiled and evaluated.
- RT-DETR-L is not practical for edge deployment in this setup due to parameter count and lack of Hailo support at the time of writing.

---

## Dependencies overview

| Package | Purpose |
|---|---|
| `ultralytics>=8.2.0` | YOLO training and inference |
| `opencv-python` | Image/video processing |
| `pyueye` | IDS uEye camera SDK bindings |
| `roboflow<2.3` | Dataset management |
| `torch` / `torchvision` | Deep learning backend |
| `numpy`, `pandas` | Data utilities |
| `hailo_platform` | Hailo-8 runtime (install separately) |

---

## Repository notes

- Large training artefacts (`runs/`) are stored locally and are not pushed to the remote.
- The dataset is versioned with DVC. Run `dvc pull` after cloning to fetch `data/`.
- Calibration image paths in `calib/list.txt` reference the original capture machine; update them before running `calibration_data.py` on a new system.