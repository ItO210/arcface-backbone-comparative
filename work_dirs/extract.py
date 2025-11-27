# Run this to get the .json files with validation and training metrics used for the graphs.

import re
import json
import os

input_path = "path/to/training.log file"
training_out = "path/for/training_metrics.json"
validation_out = "path/for/validation_metrics.json"

def ensure_dir_for(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

ensure_dir_for(training_out)
ensure_dir_for(validation_out)

training_re = re.compile(
    r"^Training:\s*([\d-]+\s[\d:,]+).*?Loss\s+([\d.eE+-]+)\s+LearningRate\s+([\d.eE+-]+)\s+Epoch:\s+(\d+)\s+Global Step:\s+(\d+)\s+Fp16 Grad Scale:\s+(\d+)",
    re.IGNORECASE
)

validation_re = re.compile(
    r"^Training:\s*([\d-]+\s[\d:,]+)-\[(?P<dataset>[\w\-]+)\]\[(?P<step>\d+)\](?P<metric>XNorm|Accuracy-Flip|Accuracy-Highest):\s+(?P<value>[\d.+-eE]+)",
    re.IGNORECASE
)

training_results = []
validation_map = {}

with open(input_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        t = training_re.search(line)
        if t:
            _, loss_s, lr_s, epoch_s, gstep_s, fp16_s = t.groups()
            try:
                training_results.append({
                    "Loss": float(loss_s),
                    "LearningRate": float(lr_s),
                    "Epoch": int(epoch_s),
                    "GlobalStep": int(gstep_s),
                    "Fp16GradScale": int(fp16_s)
                })
            except ValueError:
                training_results.append({
                    "Loss": loss_s,
                    "LearningRate": lr_s,
                    "Epoch": epoch_s,
                    "GlobalStep": gstep_s,
                    "Fp16GradScale": fp16_s
                })

        v = validation_re.search(line)
        if v:
            dataset = v.group("dataset")
            step = int(v.group("step"))
            metric = v.group("metric")
            value = v.group("value")

            key = f"{dataset}_{step}"
            if key not in validation_map:
                validation_map[key] = {
                    "Dataset": dataset,
                    "Step": step,
                    "XNorm": None,
                    "AccuracyFlip": None,
                    "AccuracyFlipStd": None,
                    "AccuracyHighest": None
                }

            if metric.lower() == "xnorm":
                validation_map[key]["XNorm"] = float(value)
            elif metric.lower() == "accuracy-flip":
                mean, sep, std = value.partition("+-")
                validation_map[key]["AccuracyFlip"] = float(mean)
                if sep:
                    validation_map[key]["AccuracyFlipStd"] = float(std)
            elif metric.lower() == "accuracy-highest":
                validation_map[key]["AccuracyHighest"] = float(value)

validation_results = list(validation_map.values())

with open(training_out, "w") as t_out:
    json.dump(training_results, t_out, indent=4)

with open(validation_out, "w") as v_out:
    json.dump(validation_results, v_out, indent=4)

print(f"Saved {len(training_results)} training entries to: {training_out}")
print(f"Saved {len(validation_results)} validation entries to: {validation_out}")
