import json
import math
import matplotlib.pyplot as plt

log_path = "/home/elenabuk/EmbeddedAiProject/Embedded_AI/RT-DETR/output/rtdetr_r18vd_6x_coco/log.txt"

epochs = []
train_losses = []
lrs = []
map_main = []      # first COCO metric
map_50 = []        # second COCO metric (AP50)

with open(log_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        epoch = d.get("epoch")
        if epoch is None:
            continue

        epochs.append(epoch)
        train_losses.append(d.get("train_loss", math.nan))
        lrs.append(d.get("train_lr", math.nan))

        bbox_metrics = d.get("test_coco_eval_bbox", [])
        map_main.append(bbox_metrics[0] if len(bbox_metrics) > 0 else math.nan)
        map_50.append(bbox_metrics[1] if len(bbox_metrics) > 1 else math.nan)

# train loss
plt.figure(figsize=(6,4))
plt.plot(epochs, train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Train loss")
plt.grid(True)
plt.title("RT-DETR train loss")
plt.tight_layout()
plt.savefig("/home/elenabuk/EmbeddedAiProject/Embedded_AI/RT-DETR/output/predictions/prediction_1/train_loss.png", dpi=200)

# learning rate
plt.figure(figsize=(6,4))
plt.plot(epochs, lrs, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.grid(True)
plt.title("Learning rate")
plt.tight_layout()
plt.savefig("/home/elenabuk/EmbeddedAiProject/Embedded_AI/RT-DETR/output/predictions/prediction_1/lr.png", dpi=200)

# mAP (COCO metric index 0) and AP50 (index 1)
plt.figure(figsize=(6,4))
plt.plot(epochs, map_main, label="mAP (idx 0)", marker="o")
plt.plot(epochs, map_50, label="AP50 (idx 1)", marker="x")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.grid(True)
plt.legend()
plt.title("Validation COCO metrics")
plt.tight_layout()
plt.savefig("/home/elenabuk/EmbeddedAiProject/Embedded_AI/RT-DETR/output/predictions/prediction_1/metrics.png", dpi=200)

plt.show()
