from pyueye import ueye
import numpy as np, json, time, os, cv2, sys

# --- session folders ---
sess = time.strftime("%Y%m%d") + "_A"
out_dir = os.path.join("dataset", "raw", sess)
img_dir, meta_dir = os.path.join(out_dir,"images"), os.path.join(out_dir,"meta")
os.makedirs(img_dir, exist_ok=True); os.makedirs(meta_dir, exist_ok=True)

# --- open camera ---
hCam = ueye.HIDS(0)
ret = ueye.is_InitCamera(hCam, None)
if ret != 0:
    print(f"is_InitCamera failed: {ret}", file=sys.stderr); sys.exit(1)

try:
    # Display/Color mode
    ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
    ueye.is_SetColorMode(hCam, ueye.IS_CM_MONO8)   # change to IS_CM_BGR8_PACKED for color
    BPP = 8

    # Lock exposure (tune this for your lighting)
    EXPOSURE_MS = 5.0
    ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, ueye.double(0), ueye.double(0))
    ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ueye.double(EXPOSURE_MS), ueye.sizeof(ueye.double(EXPOSURE_MS)))

    # Full sensor size (set an AOI if you want smaller images / higher FPS)
    si = ueye.SENSORINFO(); ueye.is_GetSensorInfo(hCam, si)
    W, H = int(si.nMaxWidth), int(si.nMaxHeight)

    # Allocate image buffer
    mem_ptr = ueye.c_mem_p(); mem_id = ueye.int()
    ueye.is_AllocImageMem(hCam, W, H, BPP, mem_ptr, mem_id)
    ueye.is_SetImageMem(hCam, mem_ptr, mem_id)
    pitch = ueye.INT(); ueye.is_GetImageMemPitch(hCam, pitch)

    # --- capture exactly 100 frames ---
    N = 100
    for i in range(1, N+1):
        # capture one frame (blocking)
        ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
        arr = ueye.get_data(mem_ptr, W, H, BPP, pitch.value, copy=True)
        img = np.reshape(arr, (H, W)).astype(np.uint8)

        # filenames
        fname = f"{time.strftime('%Y%m%d')}_A_{i:06d}.jpg"
        fpath = os.path.join(img_dir, fname)

        # write image
        cv2.imwrite(fpath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # write metadata
        meta = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z",
            "exposure_us": int(EXPOSURE_MS*1000),
            "gain": 0.0, "gamma": 1.0,
            "belt_speed_mm_s": None, "frame_idx": i, "trigger": "free_run",
            "width": W, "height": H, "bpp": BPP
        }
        with open(os.path.join(meta_dir, fname.replace(".jpg",".json")), "w") as f:
            json.dump(meta, f)

        if i % 10 == 0:
            print(f"[{i}/{N}] saved {fpath}")

finally:
    # Always free memory + close camera
    try:
        ueye.is_FreeImageMem(hCam, mem_ptr, mem_id)
    except Exception:
        pass
    ueye.is_ExitCamera(hCam)
    print("Done.")
