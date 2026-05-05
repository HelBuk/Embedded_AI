#!/usr/bin/env python3
"""
Compare Ultralytics YOLO (.pt) vs Hailo HEF on the same frame.

Outputs:
- Ultralytics: preprocess / inference / postprocess / total / FPS(total)
- Hailo: infer-only (pure pipe.infer) / optional letterbox+infer total / FPS

Example:
  python bench_yolo_vs_hailo.py \
    --ultra yolo8n:~/.../yolov8n.pt yolo11n:~/.../yolo11n.pt yolo11s:~/.../yolo11s.pt \
    --hef   yolo8n:~/.../yolov8n_960.hef yolo11n:~/.../yolo11n_960.hef yolo11s:~/.../yolo11s_960.hef \
    --source ~/EmbeddedAiProject/Embedded_AI/data/all.mp4 \
    --device cpu --imgsz 960,960 --warmup 50 --iters 300
"""
import argparse, time, os, sys
from statistics import mean, median
import numpy as np
import cv2

_ultra_ok = True
try:
    from ultralytics import YOLO
except Exception:
    _ultra_ok = False

_hailo_ok = True
try:
    from hailo_platform import (
        VDevice, HEF, ConfigureParams, InferVStreams,
        HailoStreamInterface as SIF,
        InputVStreamParams, OutputVStreamParams
    )
except Exception:
    _hailo_ok = False

def parse_imgsz(s):
    if "," in s:
        h, w = [int(x) for x in s.split(",")]
        return (h, w)
    return int(s)

def grab_one_frame(path, target_hw):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        # fallback random
        if isinstance(target_hw, int):
            h = w = target_hw
        else:
            h, w = target_hw
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    # Resize to requested size for Ultralytics timing parity
    if isinstance(target_hw, int):
        target_hw = (target_hw, target_hw)
    frame = cv2.resize(frame, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    return frame

def letterbox(im, new_size_hw, color=(114,114,114)):
    # Matching your earlier letterbox (center pad)
    if isinstance(new_size_hw, int):
        new = (new_size_hw, new_size_hw)
    else:
        new = new_size_hw
    h, w = im.shape[:2]
    nh, nw = new
    s = min(nh / h, nw / w)
    nh2, nw2 = int(round(h * s)), int(round(w * s))
    imr = cv2.resize(im, (nw2, nh2), interpolation=cv2.INTER_LINEAR)
    top = (nh - nh2) // 2; bottom = nh - nh2 - top
    left = (nw - nw2) // 2; right = nw - nw2 - left
    out = cv2.copyMakeBorder(imr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out

# ---------------- Ultralytics ----------------
def bench_ultra(models, source_frame, device, imgsz, warmup, iters):
    if not _ultra_ok:
        print("Ultralytics not available; skipping PyTorch side.", file=sys.stderr)
        return []
    rows = []
    for name, path in models:
        if not os.path.isfile(os.path.expanduser(path)):
            print(f"[ultra] Missing model: {path}", file=sys.stderr); continue
        model = YOLO(os.path.expanduser(path))
        # Warmup
        for _ in range(warmup):
            _ = model.predict(source_frame, imgsz=imgsz, device=device, save=False, verbose=False)
        pre, inf, post = [], [], []
        for _ in range(iters):
            r = model.predict(source_frame, imgsz=imgsz, device=device, save=False, verbose=False)
            sp = r[0].speed  # {'preprocess','inference','postprocess'}
            pre.append(sp['preprocess']); inf.append(sp['inference']); post.append(sp['postprocess'])
        mpre, minf, mpost = mean(pre), mean(inf), mean(post)
        total = mpre + minf + mpost
        rows.append({
            "kind": "ultra",
            "name": name,
            "pre_ms": mpre, "inf_ms": minf, "post_ms": mpost,
            "total_ms": total, "fps": 1000.0 / total if total > 0 else float("inf"),
            "imgsz": imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
        })
    return rows

# ---------------- Hailo ----------------
def pick_nms_output(out_params):
    for k in out_params:
        key = k.lower()
        if "nms" in key or "postprocess" in key:
            return k
    return next(iter(out_params))

def bench_hailo(hefs, input_for_letterbox, warmup, iters, include_letterbox_total=True):
    if not _hailo_ok:
        print("Hailo platform not available; skipping HEF side.", file=sys.stderr)
        return []
    rows = []
    for name, path in hefs:
        p = os.path.expanduser(path)
        if not os.path.isfile(p):
            print(f"[hailo] Missing HEF: {path}", file=sys.stderr); continue
        hef = HEF(p)
        with VDevice() as vdev:
            params = ConfigureParams.create_from_hef(hef, SIF.PCIe)
            ng = vdev.configure(hef, params)[0]
            try:
                in_params  = InputVStreamParams.make_from_network_group(ng)
                out_params = OutputVStreamParams.make_from_network_group(ng)
            except AttributeError:
                in_params  = InputVStreamParams.make(ng)
                out_params = OutputVStreamParams.make(ng)
            in_name = next(iter(in_params))
            out_name = pick_nms_output(out_params)
            out_params = {out_name: out_params[out_name]}
            # find input (H,W,C)
            ishape = None
            for ii in ng.get_input_vstream_infos():
                if ii.name == in_name:
                    ishape = ii.shape
                    break
            if ishape is None:
                print(f"[hailo] Could not read input shape for {name}", file=sys.stderr); continue
            H, W, C = ishape
            # Prepare one NHWC uint8 batch
            # If requested, include letterbox time (resize + pad input_for_letterbox -> (H,W))
            if include_letterbox_total:
                # Build a letterboxed frame only once; we’ll time letterbox per-iter to match Ultralytics 'preprocess'
                base_src = input_for_letterbox
            dummy = np.random.randint(0, 256, size=(1, H, W, C), dtype=np.uint8)
            dummy = np.ascontiguousarray(dummy)

            lat_infer_ms = []
            lat_letter_ms = []  # optional
            with ng.activate():
                with InferVStreams(ng, in_params, out_params) as pipe:
                    # Warmup
                    for _ in range(warmup):
                        _ = pipe.infer({in_name: dummy})
                    # Measured
                    for _ in range(iters):
                        if include_letterbox_total:
                            t0 = time.perf_counter_ns()
                            lb = letterbox(base_src, (H, W))  # produces HxW
                            inp = np.ascontiguousarray(lb.astype(np.uint8)[None, ...])
                            t1 = time.perf_counter_ns()
                            _ = pipe.infer({in_name: inp})
                            t2 = time.perf_counter_ns()
                            lat_letter_ms.append((t1 - t0) / 1e6)
                            lat_infer_ms.append((t2 - t1) / 1e6)
                        else:
                            t0 = time.perf_counter_ns()
                            _ = pipe.infer({in_name: dummy})
                            t1 = time.perf_counter_ns()
                            lat_infer_ms.append((t1 - t0) / 1e6)

            mean_infer = mean(lat_infer_ms)
            mean_letter = mean(lat_letter_ms) if lat_letter_ms else 0.0
            total_ms = mean_infer + mean_letter
            rows.append({
                "kind": "hailo",
                "name": name,
                "input_hw": (H, W),
                "infer_ms": mean_infer,
                "letter_ms": mean_letter,
                "total_ms": total_ms,
                "fps_infer_only": 1000.0 / mean_infer if mean_infer > 0 else float("inf"),
                "fps_total": 1000.0 / total_ms if total_ms > 0 else float("inf")
            })
    return rows

def main():
    ap = argparse.ArgumentParser(description="Compare Ultralytics (.pt) vs Hailo (.hef) FPS on one frame.")
    ap.add_argument("--ultra", nargs="*", default=[],
                    help="name:path pairs for Ultralytics .pt (e.g., yolo8n:/path/to.pt)")
    ap.add_argument("--hef", nargs="*", default=[],
                    help="name:path pairs for Hailo .hef (e.g., yolo8n:/path/to.hef)")
    ap.add_argument("--source", required=True, help="Video or image to take one frame from")
    ap.add_argument("--device", default="cpu", help="Ultralytics device (cpu or cuda:0)")
    ap.add_argument("--imgsz", default="960,960", help="Ultralytics imgsz (int or H,W)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--hailo-no-letterbox", action="store_true",
                    help="If set, Hailo total excludes letterbox; reports infer-only.")
    args = ap.parse_args()

    def parse_np(pairs):
        out = []
        for s in pairs:
            if ":" not in s:
                raise SystemExit(f"Bad pair '{s}', expected name:path")
            n, p = s.split(":", 1)
            out.append((n.strip(), p.strip()))
        return out

    ultra_pairs = parse_np(args.ultra)
    hef_pairs   = parse_np(args.hef)
    imgsz = parse_imgsz(args.imgsz)

    src_frame_for_ultra = grab_one_frame(args.source, imgsz)
    src_frame_for_hailo = grab_one_frame(args.source, (imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)))

    ultra_rows = bench_ultra(ultra_pairs, src_frame_for_ultra, args.device, imgsz, args.warmup, args.iters) if ultra_pairs else []
    hailo_rows = bench_hailo(hef_pairs, src_frame_for_hailo, args.warmup, args.iters, include_letterbox_total=not args.hailo_no_letterbox) if hef_pairs else []

    if ultra_rows:
        print("\n================ ULTRALYTICS (PyTorch) ================")
        print(f"{'model':12s} {'img(HxW)':10s} {'pre(ms)':>8s} {'infer(ms)':>10s} {'post(ms)':>9s} {'total(ms)':>10s} {'FPS':>8s}")
        print("-"*74)
        for r in ultra_rows:
            h,w = r["imgsz"]
            print(f"{r['name']:12s} {h}x{w:<7d} {r['pre_ms']:8.1f} {r['inf_ms']:10.1f} {r['post_ms']:9.1f} {r['total_ms']:10.1f} {r['fps']:8.1f}")

    if hailo_rows:
        print("\n====================== HAILO (HEF) =====================")
        print(f"{'model':12s} {'in(HxW)':10s} {'letter(ms)':>11s} {'infer(ms)':>10s} {'total(ms)':>10s} {'FPS(infer)':>12s} {'FPS(total)':>12s}")
        print("-"*84)
        for r in hailo_rows:
            h,w = r["input_hw"]
            print(f"{r['name']:12s} {h}x{w:<7d} {r['letter_ms']:11.1f} {r['infer_ms']:10.1f} {r['total_ms']:10.1f} {r['fps_infer_only']:12.1f} {r['fps_total']:12.1f}")

    if ultra_rows and hailo_rows:
        u_map = {r["name"]: r for r in ultra_rows}
        h_map = {r["name"]: r for r in hailo_rows}
        common = [n for n in u_map.keys() if n in h_map]
        if common:
            print("\n================== PAIRED COMPARISON ===================")
            print("(Ultralytics total vs Hailo infer-only and total)\n")
            print(f"{'model':12s} {'U_total(ms)':>11s} {'U_FPS':>7s} || {'H_infer(ms)':>11s} {'H_total(ms)':>11s} {'H_FPS(inf)':>11s} {'H_FPS(tot)':>11s}")
            print("-"*92)
            for n in sorted(common):
                u = u_map[n]; h = h_map[n]
                print(f"{n:12s} {u['total_ms']:11.1f} {u['fps']:7.1f} || {h['infer_ms']:11.1f} {h['total_ms']:11.1f} {h['fps_infer_only']:11.1f} {h['fps_total']:11.1f}")

if __name__ == "__main__":
    main()
