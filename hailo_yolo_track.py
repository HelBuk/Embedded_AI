from hailo_platform import (VDevice, HEF, ConfigureParams, InferVStreams, HailoStreamInterface as SIF,
                            InputVStreamParams, OutputVStreamParams)

import cv2, numpy as np, time, os

HEF_PATH  = "/home/pi/Documents/Embedded_AI/runs/detect/run_v7-yolov8/rf_yolov8n_fit8gb/weights/yolov8_custom.hef"
VIDEO_IN  = "/home/pi/Downloads/all.mp4"
VIDEO_OUT = "/home/pi/Downloads/all_out_new4.mp4"
CONF_THRES = 0.5
TOPK_PER_CLS = 10
INPUT_SIZE = 960  # must match HEF

# ---- styling ----
CLASSES = ["screw", "nut", "washer", "cls3", "cls4"]  # put your real class names here

# BGR colors (distinct, high-contrast)
PALETTE = [
    (40, 180, 99),   # green
    (60, 76, 231),   # red-ish
    (255, 191, 0),   # cyan-ish (remember: BGR)
    (208, 80, 192),  # magenta
    (0, 165, 255),   # orange
]

def class_color(i: int):
    return PALETTE[i % len(PALETTE)]

def draw_box_with_label(img, xyxy, label, color, lw=3, fs=0.8):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw, lineType=cv2.LINE_AA)

    # text background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(lw-1, 1))
    th = int(th * 1.3)  # a little padding
    x2_bg = min(x1 + tw + 6, img.shape[1] - 1)
    y2_bg = max(y1 - th - 4, 0)
    cv2.rectangle(img, (x1, y1), (x2_bg, y2_bg), color, -1, lineType=cv2.LINE_AA)
    fs = 10
    # text
    cv2.putText(img, label, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)



def letterbox(im, new=INPUT_SIZE, color=(114,114,114)):
    h, w = im.shape[:2]
    s = min(new / h, new / w)
    nh, nw = int(round(h*s)), int(round(w*s))
    imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new - nh) // 2; bottom = new - nh - top
    left = (new - nw) // 2; right = new - nw - left
    out = cv2.copyMakeBorder(imr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, s, left, top

def main():
    assert os.path.isfile(HEF_PATH), f"Missing HEF: {HEF_PATH}"
    assert os.path.isfile(VIDEO_IN), f"Missing input video: {VIDEO_IN}"

    cap = cv2.VideoCapture(VIDEO_IN)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    hef = HEF(HEF_PATH)
    with VDevice() as vdev:
        # Configure for PCIe Hailo-8
        params_dict = ConfigureParams.create_from_hef(hef, SIF.PCIe)
        ng = vdev.configure(hef, params_dict)[0]

        try:
            in_params  = InputVStreamParams.make_from_network_group(ng)
            out_params = OutputVStreamParams.make_from_network_group(ng)
        except AttributeError:
            # some builds expose .make(...) instead of .make_from_network_group(...)
            in_params  = InputVStreamParams.make(ng)
            out_params = OutputVStreamParams.make(ng)
    
        # Debug once
        print("Input keys:", list(in_params.keys()))
        print("Output keys:", list(out_params.keys()))

        # Prefer fused postprocess output (NMS)
        nms_name = next((n for n in out_params if "yolov8_nms_postprocess" in n), next(iter(out_params)))
        in_name = next(iter(in_params))  # e.g. "yolov8_custom/input_layer1"

        out_params = {nms_name: out_params[nms_name]}

        # (Optional) sanity prints
        in_infos  = ng.get_input_vstream_infos()
        out_infos = ng.get_output_vstream_infos()
        print("IN :", in_infos[0].name, in_infos[0].shape, in_infos[0].format.type, in_infos[0].format.order)
        for oi in out_infos:
            print("OUT:", oi.name, oi.shape, oi.format.type, oi.format.order)
        print("[hailo] Using OUT stream:", nms_name)

        # Inference pipe using the PARAM dicts
        with ng.activate():
            with InferVStreams(ng, in_params, out_params) as pipe:
                n_frames, t0 = 0, time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    n_frames += 1

                    
                    # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    lb, scale, pad_x, pad_y = letterbox(frame, INPUT_SIZE)  # feed BGR directly
                    inp = lb.astype(np.uint8)[None, ...]                    # (1,768,768,3) NHWC


                    # Feed inputs in same order as in_params keys (dict preserves insertion)
                    outs = pipe.infer({in_name: np.ascontiguousarray(inp)})  # inp shape (1,768,768,3), uint8, NHWC
                    raw  = outs[nms_name]  # e.g. "yolov8_custom/yolov8_nms_postprocess"

                    # Take batch 0 (Hailo returns list per batch)
                    batch0 = raw[0] if isinstance(raw, list) else raw

                    # If we got a list-of-classes (each is (N,5))
                    if isinstance(batch0, list):
                        for cls_idx, cls_det in enumerate(batch0):
                            if cls_det is None or len(cls_det) == 0:
                                continue

                            arr = np.asarray(cls_det, dtype=np.float32)   # (N,5)
                            boxes = arr[:, :4]                            # [ymin, xmin, ymax, xmax] in 0..1
                            scores = arr[:, 4]

                            keep = scores >= CONF_THRES
                            if not np.any(keep):
                                # continue
                             boxes = boxes[keep]
                             scores = scores[keep]
                             
                            if len(scores) > TOPK_PER_CLS:
                                idx = np.argsort(-scores)[:TOPK_PER_CLS]
                                boxes, scores = boxes[idx], scores[idx] 

                            # letterboxed pixels (0..768)
                            y1_lb = boxes[:, 0] * INPUT_SIZE
                            x1_lb = boxes[:, 1] * INPUT_SIZE
                            y2_lb = boxes[:, 2] * INPUT_SIZE
                            x2_lb = boxes[:, 3] * INPUT_SIZE

                            # undo padding + unscale to original frame
                            x1o = (x1_lb - pad_x) / max(scale, 1e-6)
                            y1o = (y1_lb - pad_y) / max(scale, 1e-6)
                            x2o = (x2_lb - pad_x) / max(scale, 1e-6)
                            y2o = (y2_lb - pad_y) / max(scale, 1e-6)

                            # clamp
                            x1o = np.clip(x1o, 0, W - 1);  y1o = np.clip(y1o, 0, H - 1)
                            x2o = np.clip(x2o, 0, W - 1);  y2o = np.clip(y2o, 0, H - 1)

                            # scale line width and font size to frame size
                            lw = max(2, int(round((W + H) / 800)))   # thicker lines
                            fs = 0.7 + 0.3 * (W >= 1280)             # bigger font on larger frames

                            for j in range(len(scores)):
                                p1 = (int(x1o[j]), int(y1o[j]))
                                p2 = (int(x2o[j]), int(y2o[j]))
                                color = class_color(cls_idx)
                                cls_name = CLASSES[cls_idx] if cls_idx < len(CLASSES) else f"cls{cls_idx}"
                                label = f"{cls_name} {float(scores[j]):.2f}"
                                draw_box_with_label(frame, (p1[0], p1[1], p2[0], p2[1]), label, color, lw=lw, fs=fs)

                    else:
                        print(f"[debug] unexpected NMS structure: type={type(batch0)}, shape={getattr(batch0,'shape',None)}")
                        break

                    writer.write(frame)

                dt = max(time.time() - t0, 1e-6)
                print(f"Done. Frames: {n_frames}, time: {dt:.1f}s, avg FPS: {n_frames/dt:.2f}")

    cap.release(); writer.release()
    print(f"Saved -> {VIDEO_OUT}")

if __name__ == "__main__":
    main()
