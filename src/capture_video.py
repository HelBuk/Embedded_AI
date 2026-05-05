from pyueye import ueye
import numpy as np
import json, time, os, sys, argparse, logging, csv, re
from logging.handlers import RotatingFileHandler
import cv2
from datetime import datetime

# ---------- helpers ----------
def set_exposure_us(hCam, exposure_us: int):
    """Set manual exposure (µs). uEye API expects milliseconds."""
    exp_ms = ueye.double(exposure_us / 1000.0)
    # turn off auto shutter first (deterministic)
    ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, ueye.double(0), ueye.double(0))
    ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exp_ms, ueye.sizeof(exp_ms))

def get_exposure_ms(hCam) -> float:
    cur = ueye.double()
    ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, cur, ueye.sizeof(cur))
    return float(cur.value)

def set_master_gain(hCam, percent: int):
    """Master gain 0..100% (use small values to avoid noise)."""
    pct = max(0, min(int(percent), 100))
    ueye.is_SetHardwareGain(hCam, pct, 0, 0, 0)

def set_gain_auto_off(hCam):
    ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, ueye.double(0), ueye.double(0))

def set_color_mode(hCam, want_color: bool):
    """Return (bits_per_pixel, channels)."""
    if want_color:
        ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED); return 24, 3
    else:
        ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8);       return 8, 1

def get_sensor_size(hCam):
    si = ueye.SENSORINFO(); ueye.is_GetSensorInfo(hCam, si)
    return int(si.nMaxWidth), int(si.nMaxHeight)

def set_frame_rate(hCam, fps: float) -> float:
    new = ueye.double()
    ueye.is_SetFrameRate(hCam, ueye.double(fps), new)
    return float(new.value)

def set_aoi(hCam, x, y, w, h):
    rect = ueye.IS_RECT()
    rect.s32X = int(x)
    rect.s32Y = int(y)
    rect.s32Width = int(w)
    rect.s32Height = int(h)
    if ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rect, ueye.sizeof(rect)) != 0:
        logging.error("is_AOI set failed")
        sys.exit(1)
    return w, h

def apply_isp_options(hCam, *, color: bool, enable_cc: bool, gamma_100: int,
                      auto_wb_once: bool, warmup_snaps: int, sat_u: int = 0, sat_v: int = 0):
    """Optional 'Demo look': color correction, gamma, one-shot white balance, saturation."""
    # One-shot white balance (color only)
    if color and auto_wb_once:
        ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE, ueye.double(1), ueye.double(0))
        for _ in range(max(3, warmup_snaps)):
            ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
        ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE, ueye.double(0), ueye.double(0))

    # Color correction
    if enable_cc:
        try:
            ueye.is_SetColorCorrection(hCam, ueye.IS_CCOR_ENABLE, ueye.INT(100))
        except Exception:
            pass

    # Gamma (100=1.0; 120≈1.2; 140≈1.4)
    try:
        ueye.is_SetGamma(hCam, int(gamma_100))
    except Exception:
        pass

    # Saturation tweak (color only; optional)
    if color:
        try:
            ueye.is_Saturation(hCam, ueye.IS_SATURATION_U, int(sat_u))
            ueye.is_Saturation(hCam, ueye.IS_SATURATION_V, int(sat_v))
        except Exception:
            pass

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parse_resize(s):
    # format: "WxH" or "0x0"
    m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", s.lower())
    if not m: return (0, 0)
    return (int(m.group(1)), int(m.group(2)))

def setup_run_dirs(out_root, tag):
    # root: out_root/YYYY-MM-DD/tag/runXX/
    date_str = datetime.now().strftime("%Y-%m-%d")
    base = os.path.join(out_root, date_str, tag)
    ensure_dir(base)
    runs = sorted([d for d in os.listdir(base) if d.startswith("run") and len(d) == 5 and d[3:].isdigit()])
    next_idx = (int(runs[-1][3:]) + 1) if runs else 1
    run_dir = os.path.join(base, f"run{next_idx:02d}")
    frames_dir = os.path.join(run_dir, "frames")
    labels_dir = os.path.join(run_dir, "labels")
    ensure_dir(frames_dir); ensure_dir(labels_dir)
    return run_dir, frames_dir, labels_dir

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Record uEye camera and export annotation-ready frames for YOLO/Roboflow")
    ap.add_argument("--out", default="data/captures", help="root output folder")
    ap.add_argument("--tag", default="run", help="experiment tag (e.g., conveyor_neutral)")
    ap.add_argument("--duration", type=int, default=10, help="seconds to record")
    ap.add_argument("--fps", type=float, default=6.0, help="target frames per second")
    ap.add_argument("--exposure_us", type=int, default=18000, help="exposure in microseconds")
    ap.add_argument("--gain", type=int, default=0, help="master gain percent (0..100)")
    ap.add_argument("--color", action="store_true", help="BGR8 (default MONO8 if omitted)")
    ap.add_argument("--warmup", type=int, default=5, help="frames to discard after settings settle")
    ap.add_argument("--aoi", default=None, help="Crop AOI as x,y,w,h (e.g. 100,500,1920,1080)")

    # ISP / “Demo look” knobs
    ap.add_argument("--gamma_100", type=int, default=120, help="gamma *100 (100=1.0, 120≈1.2, 140≈1.4)")
    ap.add_argument("--enable_cc", action="store_true", help="enable color correction matrix")
    ap.add_argument("--auto_wb_once", action="store_true", help="run one-shot white balance at start (color only)")
    ap.add_argument("--sat_u", type=int, default=0, help="U saturation tweak (-100..100)")
    ap.add_argument("--sat_v", type=int, default=10, help="V saturation tweak (-100..100)")

    # Annotation/export knobs
    ap.add_argument("--extract_frames", action="store_true", default=True, help="save per-frame JPEGs for annotation")
    ap.add_argument("--every_n", type=int, default=1, help="save every Nth frame (1=all)")
    ap.add_argument("--jpg_quality", type=int, default=95, help="JPEG quality (1..100)")
    ap.add_argument("--resize", default="0x0", help='resize (e.g. "1920x1080"); "0x0" keeps native')
    ap.add_argument("--save_video", action=argparse.BooleanOptionalAction, default=True, help="also save MP4 video")
    ap.add_argument("--classes", default="", help='comma-separated class list (e.g. "screw,nut,washer")')

    args = ap.parse_args()

    # Prepare run directories
    run_dir, frames_dir, labels_dir = setup_run_dirs(args.out, args.tag)
    outfile = os.path.join(run_dir, f"{args.tag}.mp4")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Run directory: {run_dir}")
    if args.save_video:
        logging.info(f"Will save video to {outfile}")
    if args.extract_frames:
        logging.info(f"Will save frames to {frames_dir} (every {args.every_n} frames)")

    # Save classes.txt if provided
    if args.classes.strip():
        classes_path = os.path.join(run_dir, "classes.txt")
        with open(classes_path, "w") as f:
            for c in [c.strip() for c in args.classes.split(",") if c.strip()]:
                f.write(f"{c}\n")
        logging.info(f"Wrote classes.txt with {classes_path}")

    hCam = ueye.HIDS(0)
    if ueye.is_InitCamera(hCam, None) != 0:
        logging.error("is_InitCamera failed"); sys.exit(1)

    mem_ptr, mem_id = ueye.c_mem_p(), ueye.int()
    video_writer = None
    frames_csv_path = os.path.join(run_dir, "frames_manifest.csv")
    resize_w, resize_h = parse_resize(args.resize)

    try:
        # Display & pixel format
        ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
        BPP, CH = set_color_mode(hCam, args.color)

        # Deterministic settings: autos off → FPS → exposure → gain
        set_gain_auto_off(hCam)
        actual_fps = set_frame_rate(hCam, args.fps)
        set_exposure_us(hCam, args.exposure_us)
        set_master_gain(hCam, args.gain)

        # Optional ISP to match Demo look
        apply_isp_options(
            hCam,
            color=(CH == 3),
            enable_cc=args.enable_cc,
            gamma_100=args.gamma_100,
            auto_wb_once=args.auto_wb_once,
            warmup_snaps=args.warmup,
            sat_u=args.sat_u,
            sat_v=args.sat_v
        )

        # Warm-up frames so pipeline stabilizes
        for _ in range(max(0, args.warmup)):
            ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)

        # Resolution & buffer
        if args.aoi:
            x, y, w, h = map(int, args.aoi.split(","))
            W, H = set_aoi(hCam, x, y, w, h)
        else:
            maxW, maxH = get_sensor_size(hCam)
            W, H = maxW, maxH

        if ueye.is_AllocImageMem(hCam, W, H, BPP, mem_ptr, mem_id) != 0:
            logging.error("is_AllocImageMem failed"); sys.exit(1)
        ueye.is_SetImageMem(hCam, mem_ptr, mem_id)
        pitch = ueye.INT(); ueye.is_GetImageMemPitch(hCam, pitch)

        # Video writer (color frames always)
        if args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw_size = (W if resize_w == 0 else resize_w, H if resize_h == 0 else resize_h)
            video_writer = cv2.VideoWriter(outfile, fourcc, actual_fps, vw_size, isColor=True)

        # Log actuals
        fps_now = ueye.double(); ueye.is_GetFramesPerSecond(hCam, fps_now)
        logging.info(f"Mode: {'color' if CH==3 else 'mono'}  Size: {W}x{H}  Pitch: {pitch.value} B/row")
        logging.info(f"Actual FPS (live): {fps_now.value:.2f}  (freeze mode may show 0)")
        logging.info(f"Actual exposure: {get_exposure_ms(hCam):.3f} ms")
        logging.info(f"Gain: {args.gain}%  Gamma: {args.gamma_100}  CC: {args.enable_cc}  WB once: {args.auto_wb_once}")

        # Record
        num_frames = int(round(actual_fps * args.duration))
        frame_idx = 0
        saved_idx = 0
        t0 = time.time()

        # CSV manifest for frames
        csv_f = open(frames_csv_path, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame_index", "saved_index", "timestamp_ms", "filename"])

        for i in range(num_frames):
            if ueye.is_FreezeVideo(hCam, ueye.IS_WAIT) != 0:
                logging.warning(f"Frame {i} capture failed"); continue

            buf = ueye.get_data(mem_ptr, W, H, BPP, pitch.value, copy=True)

            if CH == 3:
                img = np.frombuffer(buf, np.uint8).reshape(H, pitch.value // 3, 3)[:, :W, :]
            else:
                mono = np.frombuffer(buf, np.uint8).reshape(H, pitch.value)[:, :W]
                img = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

            # Optional resize for both frames and video
            if resize_w > 0 and resize_h > 0:
                img_out = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            else:
                img_out = img

            # Write video
            if video_writer is not None:
                video_writer.write(img_out)

            # Save frame for annotation
            if args.extract_frames and (frame_idx % args.every_n == 0):
                fname = f"frame_{saved_idx+1:06d}.jpg"
                fpath = os.path.join(frames_dir, fname)
                cv2.imwrite(fpath, img_out, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)])

                # Ensure matching empty label file exists (so tools find pairs)
                open(os.path.join(labels_dir, fname.replace(".jpg", ".txt")), "a").close()

                ts_ms = int(round((time.time() - t0) * 1000))
                csv_w.writerow([frame_idx, saved_idx, ts_ms, os.path.relpath(fpath, run_dir)])
                saved_idx += 1

            frame_idx += 1

        csv_f.close()
        if video_writer is not None:
            video_writer.release()

        logging.info(f"Frames saved: {saved_idx}  |  Video saved: {bool(args.save_video)}")

        # Session metadata (your requested JSON format + full args)
        meta = {
            "file": (outfile if args.save_video else None),
            "fps": actual_fps,
            "exposure_ms": get_exposure_ms(hCam),
            "width": (resize_w if resize_w > 0 else W),
            "height": (resize_h if resize_h > 0 else H),
            "color": bool(CH == 3),
            "frames": frame_idx,
            "frames_saved": saved_idx,
            "gain_percent": int(args.gain),
            "gamma_100": int(args.gamma_100),
            "enable_cc": bool(args.enable_cc),
            "auto_wb_once": bool(args.auto_wb_once),
            "aoi": args.aoi,
            "saved_every_n": int(args.every_n),
            "jpg_quality": int(args.jpg_quality),
            "resize": args.resize,
            "extract_frames": bool(args.extract_frames),
            "save_video": bool(args.save_video),
            "classes": [c.strip() for c in args.classes.split(",") if c.strip()],
            "out_root": args.out,
            "tag": args.tag,
            "run_dir": run_dir,
            "date": datetime.now().isoformat(timespec="seconds")
        }
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logging.info(f"Meta written: {os.path.join(run_dir, 'meta.json')}")
        logging.info(f"Frames manifest: {frames_csv_path}")
        logging.info(f"Labels folder (empty placeholders): {labels_dir}")

    finally:
        try: ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
        except Exception: pass
        ueye.is_ExitCamera(hCam)

if __name__ == "__main__":
    main()
