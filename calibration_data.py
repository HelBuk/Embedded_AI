import numpy as np, cv2, os, sys

LIST = "/home/elenabuk/EmbeddedAiProject/Embedded_AI/calib/run_v7-yolov8/list.txt"

OUT = "/home/elenabuk/EmbeddedAiProject/Embedded_AI/calib/run_v7-yolov8/calib_nhwc_960.npy"

IM_SIZE = 960

def letterbox(im, new_size=IM_SIZE, color=(114,114,114)):
    h, w = im.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left

    return cv2.copyMakeBorder(im_resized, top, bottom, left, right,cv2.BORDER_CONSTANT, value=color)


paths = [p.strip() for p in open(LIST) if p.strip() and os.path.isfile(p.strip())][:400]
batch = []
for p in paths:
    im = cv2.imread(p)
    if im is None: continue
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = letterbox(im, IM_SIZE).astype(np.float32) / 255.0 # NHWC float32
    batch.append(im)

if not batch:
    print("No valid images"); sys.exit(1)
arr = np.stack(batch, axis=0)# (N,960,960,3)
np.save(OUT, arr)
print(f"Saved {arr.shape} float32 to {OUT}")

