# Run this to generate validation plots

import json
import matplotlib.pyplot as plt
import os

input_paths = [
    "paths/to/validation.json files created by extract.py"
]

output_dir = "results/combined_validation_plots"
os.makedirs(output_dir, exist_ok=True)

datasets = {}
final_acc = {}
models = set()

for path in input_paths:
    label = os.path.basename(os.path.dirname(path))
    models.add(label)

    with open(path, "r") as f:
        data = json.load(f)

    for entry in data:
        ds = entry["Dataset"]
        step = entry["Step"]
        acc = entry["AccuracyFlip"]

        if ds not in datasets:
            datasets[ds] = {
                "AccuracyFlip": {
                    "ylabel": "Accuracy",
                    "title": f"Accuracy vs Step ({ds})",
                    "data": []
                }
            }
        datasets[ds]["AccuracyFlip"]["data"].append((step, acc, label))

        final_acc.setdefault(ds, {})
        if label not in final_acc[ds] or step > final_acc[ds][label][0]:
            final_acc[ds][label] = (step, acc)

def plot_metric(y_label, title, data_tuples, output_path):
    plt.figure(figsize=(12, 7))
    for (x, y, label) in data_tuples:
        plt.plot(x, y, linewidth=1.2, alpha=0.9, label=label)
    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

for ds_name, metrics in datasets.items():
    info = metrics["AccuracyFlip"]

    model_data = {}
    for step, acc, label in info["data"]:
        model_data.setdefault(label, []).append((step, acc))

    sorted_data = []
    for label, points in model_data.items():
        points = sorted(points, key=lambda p: p[0])
        x, y = zip(*points)
        sorted_data.append((x, y, label))

    out_path = os.path.join(output_dir, f"{ds_name}_accuracy_vs_step.png")
    plot_metric(info["ylabel"], info["title"], sorted_data, out_path)

print(f"Saved accuracy plots to '{output_dir}'")

datasets_order = ["lfw", "cfp_fp", "agedb_30"]

print("\n=== Final AccuracyFlip Table ===")
print(f"{'Model':30} | {'lfw':8} | {'cfp_fp':8} | {'agedb_30':8}")
print("-" * 70)

for model in sorted(models):
    row = [model]

    for ds in datasets_order:
        if ds in final_acc and model in final_acc[ds]:
            acc = final_acc[ds][model][1]
            row.append(f"{acc:.6f}")
        else:
            row.append("-")

    print(f"{row[0]:30} | {row[1]:8} | {row[2]:8} | {row[3]:8}")

print()
