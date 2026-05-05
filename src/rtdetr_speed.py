# -*- coding: utf-8 -*-
# @File  : rtdetr_speed.py
# @Explain: Measure RT-DETR speed like Ultralytics (pre/inference/post per image)
# ======================================================================================================================
import os
import sys
import time
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
RTDETR_ROOT = os.path.join(ROOT, "RT-DETR")
sys.path.insert(0, RTDETR_ROOT)

from src.core import YAMLConfig

import cv2
import numpy as np
import torch
import torch.nn as nn



def load_rtdetr(config, resume):
    cfg = YAMLConfig(config, resume=resume)

    if resume:
        checkpoint = torch.load(resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("only support resume to load model.state_dict by now.")

    # load weights
    cfg.model.load_state_dict(state)

    # separate model and postprocessor so we can time them
    model = cfg.model.deploy()
    postprocessor = cfg.postprocessor.deploy()

    model.eval()
    postprocessor.eval()

    return model, postprocessor, cfg


def preprocess_frame(frame, imgsz):
    """
    YOLO-like preprocess:
      - BGR -> RGB
      - resize to (imgsz, imgsz)
      - to float32 / 255
      - CHW, add batch dim
    """
    # original size (H, W)
    h0, w0 = frame.shape[:2]

    frame_resized = cv2.resize(frame, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = frame_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0)        # -> (1, 3, H, W)
    img_tensor = torch.from_numpy(img)

    # orig_target_sizes: use original frame size (for proper box scaling)
    orig_size = torch.tensor([[h0, w0]], dtype=torch.float32)

    return img_tensor, orig_size


def main(args):
    device = torch.device("cpu")  # to match YOLO run (device=cpu)

    model, postprocessor, cfg = load_rtdetr(args.config, args.resume)
    model.to(device)

    # Model summary (params) similar to YOLO line
    params = sum(p.numel() for p in model.parameters())
    print(
        f"RT-DETR summary (deploy): "
        f"{len(list(model.modules()))} layers, {params:,} parameters (~{params/1e6:.3f} M)"
    )

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")

    n_frames = int(args.frames)
    pre_times, inf_times, post_times = [], [], []

    # warmup (can stabilize timings a bit)
    with torch.no_grad():
        ret, frame = cap.read()
        if ret:
            img, orig_sizes = preprocess_frame(frame, args.imgsz)
            img = img.to(device)
            _ = postprocessor(model(img), orig_sizes)  # full pipeline once

    # reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with torch.no_grad():
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # preprocess
            t0 = time.perf_counter()
            img, orig_sizes = preprocess_frame(frame, args.imgsz)
            img = img.to(device)
            orig_sizes = orig_sizes.to(device)
            t1 = time.perf_counter()

            # inference (backbone/head only)
            outputs = model(img)
            t2 = time.perf_counter()

            # postprocess (decode / scale boxes / etc.)
            _ = postprocessor(outputs, orig_sizes)
            t3 = time.perf_counter()

            pre_times.append((t1 - t0) * 1000.0)
            inf_times.append((t2 - t1) * 1000.0)
            post_times.append((t3 - t2) * 1000.0)

            print(
                f"video 1/1 (frame {i+1}/{n_frames}) {args.source}: "
                f"{args.imgsz}x{args.imgsz}, "
                f"{(t3 - t0)*1000.0:.1f}ms"
            )

    cap.release()

    if pre_times:
        pre_mean = sum(pre_times) / len(pre_times)
        inf_mean = sum(inf_times) / len(inf_times)
        post_mean = sum(post_times) / len(post_times)
        print(
            f"Speed: {pre_mean:.1f}ms preprocess, "
            f"{inf_mean:.1f}ms inference, "
            f"{post_mean:.1f}ms postprocess per image "
            f"at shape (1, 3, {args.imgsz}, {args.imgsz})"
        )
    else:
        print("No frames processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="../configs/rtdetr/rtdetr_r18_coco.yml",  # put your RT-DETR-R18 config here
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default="../weights/rtdetr_r18_best.pth",         # your trained checkpoint
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="~/EmbeddedAiProject/Embedded_AI/data/all.mp4",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,                                      # to match your YOLO run
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,                                       # same 50 frames
    )

    args = parser.parse_args()
    main(args)
