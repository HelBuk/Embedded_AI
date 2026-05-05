# ueye_yolo_rt_fixed_roi.py
# Real-time IDS uEye -> Ultralytics YOLO with:
from pyueye import ueye
import cv2, numpy as np, time, argparse, sys, signal

# python3 ueye_yolo_rt_fixed.py   \
# --model /home/pi/Documents/Embedded_AI/runs/detect/run_v7-yolov8/rf_yolov8n_fit8gb/weights/best.pt
# --imgsz 960 --fps 5 --exposure_us 200000 --gain 10 --color --device cpu --conf 0.25  --zoom 1.5


from ultralytics import YOLO

def set_color_mode(hCam, want_color: bool):
    if want_color:
        ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED); return 24, 3
    else:
        ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8); return 8, 1

def get_sensor_size(hCam):
    si = ueye.SENSORINFO(); ueye.is_GetSensorInfo(hCam, si)
    return int(si.nMaxWidth), int(si.nMaxHeight)

def get_aoi(hCam):
    rect = ueye.IS_RECT()
    ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rect, ueye.sizeof(rect))
    return int(rect.s32X), int(rect.s32Y), int(rect.s32Width), int(rect.s32Height)

def set_aoi(hCam, x, y, w, h):
    """Try to set hardware AOI; return (ok, (x,y,w,h)) with actual AOI."""
    rect = ueye.IS_RECT()
    rect.s32X, rect.s32Y, rect.s32Width, rect.s32Height = int(x), int(y), int(w), int(h)
    ret = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rect, ueye.sizeof(rect))
    ax, ay, aw, ah = get_aoi(hCam)
    return (ret == 0), (ax, ay, aw, ah)

def align_roi_to_step(x1, y1, x2, y2, Wmax, Hmax, step=2):
    """Clamp to sensor bounds and align x/y/width/height to 'step' multiples (even = 2 typical)."""
    x1 = max(0, min(Wmax-2, x1))
    y1 = max(0, min(Hmax-2, y1))
    x2 = max(x1+2, min(Wmax,   x2))
    y2 = max(y1+2, min(Hmax,   y2))
    x1 = (x1 // step) * step
    y1 = (y1 // step) * step
    w  = ((x2 - x1) // step) * step
    h  = ((y2 - y1) // step) * step
    if w < step: w = step
    if h < step: h = step
    if x1 + w > Wmax: x1 = max(0, Wmax - w)
    if y1 + h > Hmax: y1 = max(0, Hmax - h)
    return x1, y1, w, h

def set_frame_rate(hCam, fps: float) -> float:
    new = ueye.double()
    ueye.is_SetFrameRate(hCam, ueye.double(fps), new)
    return float(new.value)

def get_camera_fps(hCam) -> float:
    out = ueye.double()
    ueye.is_GetFramesPerSecond(hCam, out)
    return float(out.value)

def get_exposure_ms(hCam) -> float:
    val = ueye.double()
    ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, val, ueye.sizeof(val))
    return float(val.value)

def set_exposure_ms(hCam, exp_ms: float) -> float:
    val = ueye.double(exp_ms)
    ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, val, ueye.sizeof(val))
    return get_exposure_ms(hCam)

def last_ueye_error(hCam):
    err = ueye.INT(); p = ueye.c_char_p()
    ueye.is_GetError(hCam, err, p)
    return err.value, (p.value.decode() if p and p.value else "")

def letterbox(im, new=960, color=(114,114,114)):
    h, w = im.shape[:2]
    s = min(new / h, new / w)
    nh, nw = int(round(h*s)), int(round(w*s))
    imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new - nh) // 2; bottom = new - nh - top
    left = (new - nw) // 2; right  = new - nw - left
    return cv2.copyMakeBorder(imr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def parse_roi(roi_str, force_square=False):
    # roi_str like "x1,y1,x2,y2"
    x1, y1, x2, y2 = [int(v) for v in roi_str.split(",")]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must satisfy x2>x1 and y2>y1")
    w = x2 - x1
    h = y2 - y1
    if force_square:
        side = min(w, h)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        w = h = side
        x2 = x1 + w
        y2 = y1 + h
    return x1, y1, x2, y2, w, h

def apply_hardware_zoom(hCam, ax, ay, aw, ah, zoom, step, Wmax, Hmax):
    """
    zoom > 1 shrinks AOI (hardware) around its center.
    Returns (ok, (ax,ay,aw,ah)) with actual AOI after applying zoom.
    """
    if zoom <= 1.0:
        return True, (ax, ay, aw, ah)
    cx, cy = ax + aw // 2, ay + ah // 2
    new_w = max(step, int((aw / zoom) // step) * step)
    new_h = max(step, int((ah / zoom) // step) * step)
    x1 = cx - new_w // 2
    y1 = cy - new_h // 2
    x1, y1, new_w, new_h = align_roi_to_step(x1, y1, x1 + new_w, y1 + new_h, Wmax, Hmax, step=step)
    ok, (ax2, ay2, aw2, ah2) = set_aoi(hCam, x1, y1, new_w, new_h)
    return ok, (ax2, ay2, aw2, ah2)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to .pt (e.g., best.pt)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--exposure_us", type=int, default=18000)
    ap.add_argument("--gain", type=int, default=0)
    ap.add_argument("--color", action="store_true", help="use BGR8 (default MONO8)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:0")
    ap.add_argument("--conf", type=float, default=0.25)

    # ROI / zoom / crop
    ap.add_argument("--roi", type=str, default=None, help="ROI as x1,y1,x2,y2 (e.g., 287,122,780,840)")
    ap.add_argument("--roi_square", action="store_true", help="force a square ROI centered inside given rect (hardware if possible)")
    ap.add_argument("--roi_step", type=int, default=2, help="AOI alignment step (2 or 4 typical)")
    ap.add_argument("--zoom", type=float, default=1.0, help="Hardware AOI zoom (>1 shrinks AOI around center; 2 halves width/height)")
    ap.add_argument("--crop", type=str, default=None, help="Software crop after capture: x1,y1,x2,y2 (sensor coords; will be clamped to active AOI)")

    args = ap.parse_args()

    # Load YOLO
    model = YOLO(args.model)

    # Init uEye
    hCam = ueye.HIDS(0)
    if ueye.is_InitCamera(hCam, None) != 0:
        print("is_InitCamera failed", file=sys.stderr); sys.exit(1)

    # Clean Ctrl-C
    running = {"flag": True}
    def handle_sigint(sig, frm): running["flag"] = False
    signal.signal(signal.SIGINT, handle_sigint)

    # Display & pixel format
    ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
    BPP, CH = set_color_mode(hCam, args.color)

    # Disable autos
    ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, ueye.double(0), ueye.double(0))
    ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_GAIN,    ueye.double(0), ueye.double(0))
    ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE, ueye.double(0), ueye.double(0))

    # FPS then exposure (exposure must fit in frame period)
    fps_set = set_frame_rate(hCam, args.fps)
    frame_period_ms = 1000.0 / max(1e-6, fps_set)
    req_exp_ms = args.exposure_us / 1000.0
    exp_ms = min(req_exp_ms, max(0.5, frame_period_ms - 2.0))
    exp_applied_ms = set_exposure_ms(hCam, exp_ms)

    # ROI: try hardware AOI first (align to step for better acceptance)
    soft_crop = False
    roi_info = None
    if args.roi:
        Wmax, Hmax = get_sensor_size(hCam)
        x1, y1, x2, y2, w_req, h_req = parse_roi(args.roi, force_square=args.roi_square)
        x_al, y_al, w_al, h_al = align_roi_to_step(x1, y1, x2, y2, Wmax, Hmax, step=args.roi_step)
        ok, (ax, ay, aw, ah) = set_aoi(hCam, x_al, y_al, w_al, h_al)
        if ok:
            print(f"AOI set (requested {x1},{y1},{x2-x1}x{y2-y1} -> aligned {x_al},{y_al},{w_al}x{h_al})"
                  f" -> actual {ax},{ay},{aw}x{ah}")
        else:
            code, msg = last_ueye_error(hCam)
            print(f"AOI hardware set failed ({code}: {msg}). Falling back to software crop.", file=sys.stderr)
            soft_crop = True
            roi_info = (x1, y1, x2, y2)
            ax, ay, aw, ah = get_aoi(hCam)
    else:
        ax, ay, aw, ah = get_aoi(hCam)

    # Apply hardware zoom around the currently active AOI
    if args.zoom and args.zoom > 1.0:
        Wmax, Hmax = get_sensor_size(hCam)
        ok_zoom, (ax2, ay2, aw2, ah2) = apply_hardware_zoom(
            hCam, ax, ay, aw, ah, args.zoom, args.roi_step, Wmax, Hmax
        )
        if ok_zoom:
            print(f"Hardware zoom {args.zoom:.2f} applied: AOI {ax},{ay},{aw}x{ah} -> {ax2},{ay2},{aw2}x{ah2}")
            ax, ay, aw, ah = ax2, ay2, aw2, ah2
        else:
            print("Warning: camera did not accept zoomed AOI; keeping previous AOI.")

    # Optional extra software crop region (always after capture)
    soft_extra_crop = None
    if args.crop:
        try:
            cx1, cy1, cx2, cy2, _, _ = parse_roi(args.crop, force_square=False)
            # We will clamp to [ax..ax+aw, ay..ay+ah] later per-frame
            soft_extra_crop = (cx1, cy1, cx2, cy2)
            print(f"Software crop requested: {cx1},{cy1}–{cx2},{cy2} (will be clamped to AOI)")
        except Exception as e:
            print(f"--crop parse error: {e}. Ignoring.", file=sys.stderr)

    # Allocate a small sequence of buffers for the current AOI
    W, H = aw, ah
    buffers = []
    for _ in range(3):
        mem_ptr, mem_id = ueye.c_mem_p(), ueye.int()
        if ueye.is_AllocImageMem(hCam, W, H, BPP, mem_ptr, mem_id) != 0:
            print("is_AllocImageMem failed", file=sys.stderr); sys.exit(1)
        ueye.is_AddToSequence(hCam, mem_ptr, mem_id)
        buffers.append((mem_ptr, mem_id))

    if ueye.is_InitImageQueue(hCam, 0) != 0:
        print("is_InitImageQueue failed", file=sys.stderr); sys.exit(1)

    if ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT) != 0:
        print("is_CaptureVideo failed", file=sys.stderr); sys.exit(1)

    cam_fps_now = get_camera_fps(hCam)
    exp_now_ms  = get_exposure_ms(hCam)
    print(f"uEye ready: AOI {ax},{ay} size {W}x{H}, {'BGR' if CH==3 else 'MONO'}, "
          f"FPS req={args.fps:.2f} set={fps_set:.2f} cam={cam_fps_now:.2f}, exp={exp_now_ms:.2f} ms")

    # Window
    cv2.namedWindow("uEye YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("uEye YOLO", max(800, args.imgsz), max(600, int(args.imgsz*0.75)))

    # FPS counters
    t_last = time.time()
    pipe_frames = 0
    overlay_fps = ""
    last_cam_stats = time.time()
    shown_exp_ms = exp_now_ms
    shown_cam_fps = cam_fps_now

    try:
        while running["flag"]:
            if cv2.getWindowProperty("uEye YOLO", cv2.WND_PROP_VISIBLE) < 1:
                break

            mem_ptr = ueye.c_mem_p()
            mem_id  = ueye.int()
            ret = ueye.is_WaitForNextImage(hCam, 200, mem_ptr, mem_id)
            if ret != 0:
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # Esc/q
                    break
                continue

            pitch = ueye.INT()
            ueye.is_GetImageMemPitch(hCam, pitch)
            buf = ueye.get_data(mem_ptr, W, H, BPP, pitch.value, copy=True)

            if CH == 3:
                frame = np.frombuffer(buf, np.uint8).reshape(H, pitch.value // 3, 3)[:, :W, :]
            else:
                mono  = np.frombuffer(buf, np.uint8).reshape(H, pitch.value)[:, :W]
                frame = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

            # unlock buffer ASAP
            ueye.is_UnlockSeqBuf(hCam, mem_id, mem_ptr)

            # software crop if hardware AOI failed (fallback to user's ROI)
            if soft_crop and roi_info:
                x1, y1, x2, y2 = roi_info
                x1c = max(0, min(W-1, x1 - ax))
                y1c = max(0, min(H-1, y1 - ay))
                x2c = max(0, min(W,   x2 - ax))
                y2c = max(0, min(H,   y2 - ay))
                if x2c > x1c and y2c > y1c:
                    frame = frame[y1c:y2c, x1c:x2c]

            # optional extra software crop (always AOI-relative clamp)
            if soft_extra_crop is not None:
                cx1, cy1, cx2, cy2 = soft_extra_crop
                sx1 = max(0, min(W-1, cx1 - ax))
                sy1 = max(0, min(H-1, cy1 - ay))
                sx2 = max(0, min(W,   cx2 - ax))
                sy2 = max(0, min(H,   cy2 - ay))
                if sx2 > sx1 and sy2 > sy1:
                    frame = frame[sy1:sy2, sx1:sx2]

            # preprocess & infer
            inp = letterbox(frame, new=args.imgsz)
            res = model.predict(inp, imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)[0]
            vis = res.plot()

            # update FPS once/sec
            pipe_frames += 1
            now = time.time()
            if now - t_last >= 1.0:
                overlay_fps = f"{pipe_frames/(now - t_last):.1f} FPS   (cam {shown_cam_fps:.1f})"
                t_last = now
                pipe_frames = 0

            # refresh cam stats ~2 Hz
            if now - last_cam_stats >= 0.5:
                shown_cam_fps = get_camera_fps(hCam)
                shown_exp_ms  = get_exposure_ms(hCam)
                last_cam_stats = now

            if overlay_fps:
                cv2.putText(vis, overlay_fps, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"exp {shown_exp_ms:.1f} ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"AOI {ax},{ay} {W}x{H}"
                             + (f"  zoom {args.zoom:.2f}" if args.zoom and args.zoom>1.0 else "")
                             + ("  (soft ROI crop)" if soft_crop and roi_info else "")
                             + ("  (+ soft extra crop)" if soft_extra_crop else ""),
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

            cv2.imshow("uEye YOLO", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # Esc or q
                break

    finally:
        try: ueye.is_StopLiveVideo(hCam, ueye.IS_FORCE_VIDEO_STOP)
        except Exception: pass
        try: ueye.is_ExitImageQueue(hCam)
        except Exception: pass
        try: ueye.is_ClearSequence(hCam)
        except Exception: pass
        for mem_ptr, mem_id in buffers:
            try: ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
            except Exception: pass
        ueye.is_ExitCamera(hCam)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
