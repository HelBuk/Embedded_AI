import os
import pandas as pd
import matplotlib.pyplot as plt

# ---- Data ----
data = [
    ["YOLOv8n",     72,   3.01,  8.1,  2.1,  28.8,   0.5,  31.4],
    ["YOLOv8s",     72,  11.13, 28.4,  2.2,  61.6,   0.5,  64.3],
    ["YOLOv10n",   102,   2.27,  6.5,  2.3,  31.8,   0.2,  34.3],
    ["YOLOv10s",   106,   7.22, 21.4,  2.2,  64.4,   0.2,  66.8],
    ["YOLOv11n",   100,   2.58,  6.3,  2.1,  31.4,   0.5,  34.0],
    ["YOLOv11s",   100,   9.41, 21.3,  2.2,  61.7,   0.5,  64.4],
    ["RT-DETR-R18",462,  21.86, 60.0,  1.9, 148.3,   0.2, 150.4],
]

cols = ["Model", "Layers", "Params_M", "GFLOPs",
        "Pre_ms", "Inf_ms", "Post_ms", "Total_ms"]

df = pd.DataFrame(data, columns=cols)
print(df)

out_dir = os.path.expanduser("~/PycharmProjects/Embedded_AI/report/pics")
os.makedirs(out_dir, exist_ok=True)

# 1) Inference time vs Params
plt.figure()
for _, row in df.iterrows():
    plt.scatter(row["Params_M"], row["Inf_ms"], label=row["Model"])
# Remove duplicate labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper left")

plt.xlabel("Parameters [M]")
plt.ylabel("Inference time [ms]")
plt.title("Inference time vs. Parameters (640×640, CPU)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "detectors_640_parameters.pdf"), dpi=300)

# 2) Inference time vs GFLOPs
plt.figure()
for _, row in df.iterrows():
    plt.scatter(row["GFLOPs"], row["Inf_ms"], label=row["Model"])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper left")

plt.xlabel("GFLOPs")
plt.ylabel("Inference time [ms]")
plt.title("Inference time vs. GFLOPs (640×640, CPU)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "detectors_640_gflops.pdf"), dpi=300)

plt.show()
