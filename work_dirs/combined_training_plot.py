# Run this to generate training plots

import json
import matplotlib.pyplot as plt
import os

input_paths = [
    "paths/to/training.json files created by extract.py"
]

output_dir = "results/combined_plots"
os.makedirs(output_dir, exist_ok=True)

plots = {
    "Loss": {"ylabel": "Loss", "title": "Loss vs Global Step", "data": []},
    "LearningRate": {"ylabel": "Learning Rate", "title": "Learning Rate vs Global Step", "data": []},
    "Fp16GradScale": {"ylabel": "Fp16 Grad Scale", "title": "Fp16 Grad Scale vs Global Step", "data": []},
}

final_losses = {}

for path in input_paths:
    label = os.path.basename(os.path.dirname(path))

    with open(path, "r") as f:
        data = json.load(f)

    global_steps = [d["GlobalStep"] for d in data]
    loss = [d["Loss"] for d in data]
    lr = [d["LearningRate"] for d in data]
    fp16 = [d["Fp16GradScale"] for d in data]

    final_losses[label] = loss[-1]

    plots["Loss"]["data"].append((global_steps, loss, label))
    plots["LearningRate"]["data"].append((global_steps, lr, label))
    plots["Fp16GradScale"]["data"].append((global_steps, fp16, label))

def plot_metric(metric_name, y_label, title, data_tuples, output_path):
    plt.figure(figsize=(12, 7))
    for (x, y, label) in data_tuples:
        plt.plot(x, y, linewidth=1.2, alpha=0.9, label=label)
    plt.xlabel("Global Step")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

for metric, info in plots.items():
    out_path = os.path.join(output_dir, f"{metric.lower()}_vs_globalstep.png")
    plot_metric(metric, info["ylabel"], info["title"], info["data"], out_path)

print(f"Saved combined plots to '{output_dir}'")

print("\n=== Final Loss Per Model ===")
print(f"{'Model':30} | Final Loss")
print("-" * 50)
for model, loss in sorted(final_losses.items()):
    print(f"{model:30} | {loss:.6f}")

print()
