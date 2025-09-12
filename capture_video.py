from pyueye import ueye
import numpy as np
import json, time, os, sys, argparse, logging
from logging.handlers import RotatingFileHandler
import cv2

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
    
def set_aoi (hCam, x, y, w, h):
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

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Record uEye camera to MP4 with CLI-adjustable brightness/ISP")
    ap.add_argument("--out", default="captures", help="output folder")
    ap.add_argument("--tag", default="run", help="basename for output video")
    ap.add_argument("--duration", type=int, default=10, help="seconds to record")
    ap.add_argument("--fps", type=float, default=6.0, help="target frames per second")
    ap.add_argument("--exposure_us", type=int, default=18000, help="exposure in microseconds")
    ap.add_argument("--gain", type=int, default=0, help="master gain percent (0..100)")
    ap.add_argument("--color", action="store_true", help="BGR8 (default MONO8 if omitted)")
    ap.add_argument("--warmup", type=int, default=5, help="frames to discard after settings settle")
    ap.add_argument("--aoi", default=None, help="Crop AOI as x, y, w, h, (e.g. 100, 500 1920, 1080)")

    # ISP / “Demo look” knobs
    ap.add_argument("--gamma_100", type=int, default=120, help="gamma *100 (100=1.0, 120≈1.2, 140≈1.4)")
    ap.add_argument("--enable_cc", action="store_true", help="enable color correction matrix")
    ap.add_argument("--auto_wb_once", action="store_true", help="run one-shot white balance at start (color only)")
    ap.add_argument("--sat_u", type=int, default=0, help="U saturation tweak (-100..100)")
    ap.add_argument("--sat_v", type=int, default=10, help="V saturation tweak (-100..100)")

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    outfile = os.path.join(args.out, f"{args.tag}.mp4")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Saving video to {outfile}")

    hCam = ueye.HIDS(0)
    if ueye.is_InitCamera(hCam, None) != 0:
        logging.error("is_InitCamera failed"); sys.exit(1)

    mem_ptr, mem_id = ueye.c_mem_p(), ueye.int()
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

        # -------- PATCH: always open writer as color and feed BGR frames --------
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(outfile, fourcc, actual_fps, (W, H), isColor=True)

        # Log actuals
        fps_now = ueye.double(); ueye.is_GetFramesPerSecond(hCam, fps_now)
        logging.info(f"Mode: {'color' if CH==3 else 'mono'}  Size: {W}x{H}  Pitch: {pitch.value} B/row")
        logging.info(f"Actual FPS: {fps_now.value:.2f}")
        logging.info(f"Actual exposure: {get_exposure_ms(hCam):.3f} ms")
        logging.info(f"Gain: {args.gain}%  Gamma: {args.gamma_100}  CC: {args.enable_cc}  WB once: {args.auto_wb_once}")

        # Record
        num_frames = int(round(actual_fps * args.duration))
        for i in range(num_frames):
            if ueye.is_FreezeVideo(hCam, ueye.IS_WAIT) != 0:
                logging.warning(f"Frame {i} capture failed"); continue

            buf = ueye.get_data(mem_ptr, W, H, BPP, pitch.value, copy=True)

            if CH == 3:
                # BGR8 packed: respect stride then crop to W
                img = np.frombuffer(buf, np.uint8).reshape(H, pitch.value // 3, 3)[:, :W, :]
            else:
                # MONO8 → convert to 3-channel for mp4 writer
                mono = np.frombuffer(buf, np.uint8).reshape(H, pitch.value)[:, :W]
                img = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

            writer.write(img)

        writer.release()
        logging.info(f"Video saved: {outfile}")

        # Session metadata (optional)
        meta = {
            "file": outfile,
            "fps": actual_fps,
            "exposure_ms": get_exposure_ms(hCam),
            "width": W, "height": H, "color": bool(CH == 3),
            "frames": num_frames,
            "gain_percent": int(args.gain),
            "gamma_100": int(args.gamma_100),
            "enable_cc": bool(args.enable_cc),
            "auto_wb_once": bool(args.auto_wb_once)
        }
        with open(outfile.replace(".mp4", ".json"), "w") as f:
            json.dump(meta, f, indent=2)

    finally:
        try: ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
        except Exception: pass
        ueye.is_ExitCamera(hCam)

if __name__ == "__main__":
    main()
