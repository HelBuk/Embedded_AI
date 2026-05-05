# capture_pictures.py
from pyueye import ueye
import numpy as np
import json, time, os, sys, argparse, logging
from logging.handlers import RotatingFileHandler
import cv2

# --------------------------- Helpers ---------------------------

def set_exposure_us(hCam, exposure_us: int):
    """Set manual exposure in microseconds (API expects ms)."""
    exp_ms = ueye.double(exposure_us / 1000.0)
    # lock off auto shutter first
    ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, ueye.double(0), ueye.double(0))
    ret = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exp_ms, ueye.sizeof(exp_ms))
    if ret != 0:
        logging.error(f"is_Exposure SET_EXPOSURE failed (ret={ret})")

def get_exposure_ms(hCam) -> float:
    cur = ueye.double()
    ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, cur, ueye.sizeof(cur))
    return float(cur.value)

def set_gain_auto_off(hCam):
    ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, ueye.double(0), ueye.double(0))

def set_color_mode(hCam, want_color: bool):
    if want_color:
        ret = ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED); bpp, ch = 24, 3
    else:
        ret = ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8);       bpp, ch = 8, 1
    if ret != 0:
        logging.error(f"is_SetColorMode failed (ret={ret})")
    return bpp, ch

def get_sensor_size(hCam):
    si = ueye.SENSORINFO(); ueye.is_GetSensorInfo(hCam, si)
    return int(si.nMaxWidth), int(si.nMaxHeight)

def set_aoi(hCam, x, y, w, h):
    rect = ueye.IS_RECT()
    rect.s32X, rect.s32Y = ueye.int(x), ueye.int(y)
    rect.s32Width, rect.s32Height = ueye.int(w), ueye.int(h)
    ret = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rect, ueye.sizeof(rect))
    if ret != 0:
        logging.error(f"AOI set failed (ret={ret})")
    return int(rect.s32Width), int(rect.s32Height)

def next_session_dir(root_out:str, tag:str|None):
    date_str = time.strftime("%Y-%m-%d")
    date_dir = os.path.join(root_out, date_str); os.makedirs(date_dir, exist_ok=True)
    runs = [d for d in os.listdir(date_dir) if d.startswith("run") and os.path.isdir(os.path.join(date_dir, d))]
    idx = max([int(d.split("_")[0][3:]) for d in runs], default=0) + 1
    name = f"run{idx:02d}" + (f"_{tag}" if tag else "")
    out = os.path.join(date_dir, name)
    img, meta, log = os.path.join(out,"images"), os.path.join(out,"meta"), os.path.join(out,"logs")
    for p in (img, meta, log): os.makedirs(p, exist_ok=True)
    return out, img, meta, log, name

def setup_logging(log_dir:str):
    logger = logging.getLogger(); logger.handlers.clear(); logger.setLevel(logging.INFO)
    fh = RotatingFileHandler(os.path.join(log_dir, "capture.log"), maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    logging.info(f"Logging to {os.path.join(log_dir, 'capture.log')}")

def set_frame_rate(hCam, fps: float) -> float:
    new = ueye.double()
    ret = ueye.is_SetFrameRate(hCam, ueye.double(fps), new)
    if ret != 0: logging.error(f"is_SetFrameRate failed (ret={ret})")
    else: logging.info(f"Frame rate set: requested {fps:.2f} -> actual {new.value:.2f} fps")
    return float(new.value)

def set_master_gain(hCam, percent: int):
    percent = max(0, min(int(percent), 100))
    ret = ueye.is_SetHardwareGain(hCam, percent, 0, 0, 0)
    if ret != 0: logging.error(f"is_SetHardwareGain failed (ret={ret})")
    else: logging.info(f"Master gain set to {percent}%")

def enable_demo_look(hCam, enable_cc: bool, gamma_100: int, sat_u: int, sat_v: int, do_auto_wb_once: bool, warm_snaps: int, color: bool):
    """Apply ISP features similar to uEye Demo: WB (once), Color Correction, Gamma, Saturation."""
    if color and do_auto_wb_once:
        try:
            ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE, ueye.double(1), ueye.double(0))
            for _ in range(max(3, warm_snaps)):
                ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
        finally:
            ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE, ueye.double(0), ueye.double(0))
        logging.info("White balance: one-shot applied and locked")

    if enable_cc:
        try:
            strength = ueye.INT(100)  # typical strength
            ueye.is_SetColorCorrection(hCam, ueye.IS_CCOR_ENABLE, strength)
            logging.info("Color correction: enabled")
        except Exception:
            logging.info("Color correction: not supported on this model")

    if gamma_100 is not None:
        try:
            ueye.is_SetGamma(hCam, int(gamma_100))  # 100 = 1.0, 120 ≈ 1.2
            logging.info(f"Gamma set to {gamma_100} (100=1.0)")
        except Exception:
            logging.info("Gamma control not supported")

    if color:
        try:
            ueye.is_Saturation(hCam, ueye.IS_SATURATION_U, int(sat_u))
            ueye.is_Saturation(hCam, ueye.IS_SATURATION_V, int(sat_v))
            logging.info(f"Saturation U={sat_u}, V={sat_v}")
        except Exception:
            logging.info("Saturation control not supported")

def exposure_autotune_to_mean(hCam, mem_ptr, W, H, BPP, pitch, color:bool, target=120.0, iters=6):
    """Nudge exposure until mean grayscale ~ target; leaves autos OFF."""
    def mean_luma(img):
        if img.ndim == 2: return float(img.mean())
        return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())

    for _ in range(iters):
        ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
        buf = ueye.get_data(mem_ptr, W, H, BPP, pitch, copy=True)
        if color:
            img = np.frombuffer(buf, np.uint8).reshape(H, pitch//3, 3)[:, :W, :]
        else:
            img = np.frombuffer(buf, np.uint8).reshape(H, pitch)[:, :W]
        m = mean_luma(img)
        if abs(m - target) < 3:
            break
        cur_ms = get_exposure_ms(hCam)
        new_ms = max(1.0, min(100.0, cur_ms * (target / max(1.0, m))**0.5))
        set_exposure_us(hCam, int(new_ms * 1000))

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="IDS uEye capture with consistent, Demo-like or dataset-ready settings")
    ap.add_argument("--out", default="data/captures")
    ap.add_argument("--tag", default="")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--exposure_us", type=int, default=18000)  # 18 ms
    ap.add_argument("--color", action="store_true")
    ap.add_argument("--aoi", default="0,0,0,0")                 # x,y,w,h (0 = full sensor)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--gain", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5, help="discard frames after settings settle")

    # Demo-look toggles
    ap.add_argument("--auto_wb_once", action="store_true", help="one-shot white balance (color only), then lock")
    ap.add_argument("--cc_enable", action="store_true", help="enable color correction")
    ap.add_argument("--gamma_100", type=int, default=120, help="gamma where 100=1.0 (use 100 to disable boost)")
    ap.add_argument("--sat_v", type=int, default=10, help="V saturation tweak (-100..100)")

    # Optional exposure auto-tune
    ap.add_argument("--autotune_exposure", action="store_true", help="nudge exposure to mean brightness target then lock")
    ap.add_argument("--autotune_target", type=float, default=120.0)

    args = ap.parse_args()

    out_dir, img_dir, meta_dir, log_dir, session = next_session_dir(args.out, args.tag or None)
    setup_logging(log_dir)
    logging.info(f"Session: {session}")
    logging.info(f"Output:  {out_dir}")

    hCam = ueye.HIDS(0)
    if ueye.is_InitCamera(hCam, None) != 0:
        logging.error("is_InitCamera failed"); sys.exit(1)

    mem_ptr, mem_id = ueye.c_mem_p(), ueye.int()
    try:
        # 1) Display & pixel format
        ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
        BPP, CH = set_color_mode(hCam, args.color)

        # 2) Disable autos; set FPS → Exposure → Gain
        set_gain_auto_off(hCam)
        actual_fps = set_frame_rate(hCam, args.fps)
        set_exposure_us(hCam, args.exposure_us)
        set_master_gain(hCam, args.gain)

        # 3) ISP features to mimic Demo (WB once, CC, gamma, saturation)
        enable_demo_look(
            hCam,
            enable_cc=args.cc_enable,
            gamma_100=args.gamma_100,
            sat_u=0, sat_v=args.sat_v,
            do_auto_wb_once=args.auto_wb_once,
            warm_snaps=max(3, args.warmup),
            color=(CH==3)
        )

        # 4) Warmup snaps (settle ISP/exposure pipeline)
        for _ in range(max(0, args.warmup)):
            ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)

        # 5) AOI / full frame
        maxW, maxH = get_sensor_size(hCam)
        try:
            ax, ay, aw, ah = map(int, args.aoi.split(","))
        except: ax, ay, aw, ah = 0, 0, 0, 0
        if aw <= 0 or ah <= 0:
            W, H = maxW, maxH
            logging.info("Using full sensor frame")
        else:
            ax = max(0, min(ax, maxW-1)); ay = max(0, min(ay, maxH-1))
            aw = max(16, min(aw, maxW-ax)); ah = max(16, min(ah, maxH-ay))
            W, H = set_aoi(hCam, ax, ay, aw, ah)
            logging.info(f"Applied AOI: {ax},{ay},{aw},{ah} -> actual {W}x{H}")

        # 6) Allocate image buffer AFTER all formats are final
        if ueye.is_AllocImageMem(hCam, W, H, BPP, mem_ptr, mem_id) != 0:
            logging.error("is_AllocImageMem failed"); sys.exit(1)
        ueye.is_SetImageMem(hCam, mem_ptr, mem_id)
        pitch = ueye.INT(); ueye.is_GetImageMemPitch(hCam, pitch)

        # 7) Log actual exposure/FPS
        logging.info(f"Mode: {'color' if CH==3 else 'mono'}  Size: {W}x{H}  Pitch: {pitch.value} B/row")
        logging.info(f"Actual exposure: {get_exposure_ms(hCam):.3f} ms")
        fps_now = ueye.double(); ueye.is_GetFramesPerSecond(hCam, fps_now)
        logging.info(f"Actual FPS: {fps_now.value:.2f}")

        # 8) Optional exposure auto-tune (then locked)
        if args.autotune_exposure:
            exposure_autotune_to_mean(hCam, mem_ptr, W, H, BPP, pitch.value, color=(CH==3), target=args.autotune_target)
            logging.info(f"Auto-tuned exposure -> {get_exposure_ms(hCam):.3f} ms")

        # 9) Capture loop
        N = int(args.count)
        logging.info(f"Capturing {N} frames at requested exposure={args.exposure_us} us")
        for i in range(1, N+1):
            if ueye.is_FreezeVideo(hCam, ueye.IS_WAIT) != 0:
                logging.error(f"is_FreezeVideo failed at frame {i}"); continue

            buf = ueye.get_data(mem_ptr, W, H, BPP, pitch.value, copy=True)
            if CH == 3:   # BGR8 packed
                img = np.frombuffer(buf, np.uint8).reshape(H, pitch.value//3, 3)[:, :W, :]
            else:         # MONO8
                img = np.frombuffer(buf, np.uint8).reshape(H, pitch.value)[:, :W]

            ts = time.time()
            name = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(ts))}_{i:06d}.jpg"
            path = os.path.join(img_dir, name)
            if not cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)]):
                logging.error(f"cv2.imwrite failed for {path}"); continue

            meta = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int((ts%1)*1000):03d}Z",
                "session": session, "width": W, "height": H, "bpp": BPP, "channels": CH,
                "exposure_us": int(round(get_exposure_ms(hCam)*1000)),
                "fps": fps_now.value, "gain_percent": int(args.gain),
                "aoi": {"x": ax if aw>0 and ah>0 else 0, "y": ay if aw>0 and ah>0 else 0, "w": W, "h": H},
                "jpeg_quality": int(args.quality),
                "trigger": "freeze_wait",
                "demo_look": {"cc": bool(args.cc_enable), "gamma_100": int(args.gamma_100),
                              "sat_v": int(args.sat_v), "wb_one_shot": bool(args.auto_wb_once)}
            }
            with open(os.path.join(meta_dir, name.replace(".jpg",".json")), "w") as f:
                json.dump(meta, f, indent=2)

            if i % 10 == 0: logging.info(f"[{i}/{N}] saved {path}")

        logging.info("Capture finished.")

    except Exception as e:
        logging.exception(f"Fatal error: {e}")
    finally:
        try: ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
        except Exception: pass
        ueye.is_ExitCamera(hCam)
        logging.info("Camera closed.")

if __name__ == "__main__":
    main()
