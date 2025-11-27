# Run this to replot evaluation results on IJBB and IJBC, this file just plots using the .npy files created by the multiModel_eval.py file.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from prettytable import PrettyTable

RESULT_DIR = "path/for/results"
LABEL_FILE = "path/to/ijbc_template_pair_label.txt or ijbb_template_pair_label.txt files"
TARGET = "IJBC" # Change this to IJBB if needed.

# Models you have evaluated
MODELS = [
    "mbf",
    "mfn_m",
    "mfn_s",
    "mfn_xs",
    "r18",
    "r50",
    "r100",
    "sfn0_5",
    "sfn1_0",
    "sfn1_5",
    "sfn2_0",
    "smfn_m",
    "smfn_s",
    "smfn_xs",
]

# Colors for plotting
COLOURS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
]

X_LABELS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

print("[INFO] Loading IJB-C labels...")
label = np.loadtxt(LABEL_FILE, usecols=[2]).astype(int)

scores_dict = {}
for model_name in MODELS:
    npy_path = os.path.join(RESULT_DIR, f"{model_name.lower()}_{TARGET.lower()}.npy")
    if os.path.exists(npy_path):
        print(f"[INFO] Loading {npy_path}")
        scores_dict[model_name] = np.load(npy_path)
    else:
        print(f"[WARNING] Missing file: {npy_path}")

def compute_eer(fpr, tpr):
    """Compute Equal Error Rate (EER)"""
    fnr = 1 - tpr
    diff = np.abs(fpr - fnr)
    return fpr[np.argmin(diff)]

tpr_fpr_table = PrettyTable(['Model', 'AUC (%)', 'EER (%)'] + [f"TPR@{x}" for x in X_LABELS])

fig = plt.figure(figsize=(7, 6))

for i, (model_name, scores) in enumerate(scores_dict.items()):
    fpr, tpr, _ = roc_curve(label, scores)
    roc_auc = auc(fpr, tpr)
    eer = compute_eer(fpr, tpr) * 100

    fpr, tpr = np.flipud(fpr), np.flipud(tpr)

    plt.plot(fpr, tpr, color=COLOURS[i % len(COLOURS)], lw=1.5,
             label=f"{model_name} (AUC={roc_auc*100:.2f}%)")

    row = [model_name, f"{roc_auc*100:.2f}", f"{eer:.2f}"]
    for x in X_LABELS:
        idx = np.argmin(np.abs(fpr - x))
        row.append(f"{tpr[idx]*100:.2f}")
    tpr_fpr_table.add_row(row)

plt.xscale('log')
plt.xlim([1e-6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves on {TARGET}')
plt.legend(
    loc="lower left",
    fontsize=7,
    handlelength=1.5,
    borderpad=0.3,
    labelspacing=0.3,
    framealpha=0.8
)
plt.tight_layout()

out_path = os.path.join(RESULT_DIR, f"{TARGET.lower()}_roc_replot.png")
plt.savefig(out_path)
print(f"\n[SAVED] ROC plot saved to {out_path}")
print(tpr_fpr_table)
