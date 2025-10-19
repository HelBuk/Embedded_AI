# ueye_yolo_rt.py
# Real-time uEye → YOLOv8/YOLOv5 inference (CPU or CUDA if available)

from pyueye import ueye
import cv2, numpy as np, time, argparse, sys
from ultralytics import YOLO

# python ueye_yolo_rt.py   --model /home/pi/Documents/Embedded_AI/runs/detect/run_v7-yolov8/rf_yolov8n_fit8gb/weights/best.pt  
# --imgsz 960 --fps 10 --exposure_us 18000 --gain 0 --color --device cpu --conf 0.25

# ---------- minimal helpers ----------
def set_color_mode(hCam, want_color: bool):
    if want_color:
        ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED); return 24, 3
    else:
        ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8); return 8, 1

def get_sensor_size(hCam):
    si = ueye.SENSORINFO(); ueye.is_GetSensorInfo(hCam, si)
    return int(si.nMaxWidth), int(si.nMaxHeight)

def set_frame_rate(hCam, fps: float) -> float:
    new = ueye.double()
    ueye.is_SetFrameRate(hCam, ueye.double(fps), new)
    return float(new.value)

def letterbox(im, new=960, color=(114,114,114)):
    h, w = im.shape[:2]
    s = min(new / h, new / w)
    nh, nw = int(round(h*s)), int(round(w*s))
    imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new - nh) // 2; bottom = new - nh - top
    left = (new - nw) // 2; right  = new - nw - left
    return cv2.copyMakeBorder(imr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

# ---------- main ----------
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
    args = ap.parse_args()

    # Load YOLO
    model = YOLO(args.model)

    # Init uEye
    hCam = ueye.HIDS(0)
    if ueye.is_InitCamera(hCam, None) != 0:
        print("is_InitCamera failed", file=sys.stderr); sys.exit(1)

    # Display & pixel format
    ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
    BPP, CH = set_color_mode(hCam, args.color)

    # Disable autos (determinism), set fps/exposure/gain
    ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, ueye.double(0), ueye.double(0))
    ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_GAIN,    ueye.double(0), ueye.double(0))
    set_frame_rate(hCam, args.fps)
    ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ueye.double(args.exposure_us/1000.0), ueye.sizeof(ueye.double()))
    ueye.is_SetHardwareGain(hCam, int(args.gain), 0, 0, 0)

    # Max resolution & buffer
    W, H = get_sensor_size(hCam)
    mem_ptr, mem_id = ueye.c_mem_p(), ueye.int()
    if ueye.is_AllocImageMem(hCam, W, H, BPP, mem_ptr, mem_id) != 0:
        print("is_AllocImageMem failed", file=sys.stderr); sys.exit(1)
    ueye.is_SetImageMem(hCam, mem_ptr, mem_id)
    pitch = ueye.INT(); ueye.is_GetImageMemPitch(hCam, pitch)

    # Warmup a few frames
    for _ in range(4): ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)

    print(f"uEye ready: {W}x{H}, {'BGR' if CH==3 else 'MONO'}, FPS~{args.fps}")
    t_last, frames = time.time(), 0

    try:
        while True:
            if ueye.is_FreezeVideo(hCam, ueye.IS_WAIT) != 0:
                continue

            buf = ueye.get_data(mem_ptr, W, H, BPP, pitch.value, copy=True)
            if CH == 3:
                frame = np.frombuffer(buf, np.uint8).reshape(H, pitch.value // 3, 3)[:, :W, :]
            else:
                mono  = np.frombuffer(buf, np.uint8).reshape(H, pitch.value)[:, :W]
                frame = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

            # Resize/letterbox to model size (Ultralytics handles arbitrary sizes, but this is faster/stable)
            inp = letterbox(frame, new=args.imgsz)

            # Inference
            # stream=True yields a generator; here we do single call for clarity
            res = model.predict(inp, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
            vis = res.plot()  # annotated BGR image

            # FPS overlay
            frames += 1
            now = time.time()
            if now - t_last >= 1.0:
                fps_txt = f"{frames/(now - t_last):.1f} FPS"
                t_last, frames = now, 0
            else:
                fps_txt = ""
            if fps_txt:
                cv2.putText(vis, fps_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("uEye YOLO", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        try: ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
        except Exception: pass
        ueye.is_ExitCamera(hCam)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
