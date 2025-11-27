# Run this to get evaluation on IJBB and IJBC

import os
import warnings
from pathlib import Path
from tqdm import tqdm

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import transform as trans
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
from prettytable import PrettyTable
from backbones import get_model

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

DATASET_PATH = "path/to/IJBB or IJBC dataset"
RESULT_DIR = "path/for/output"
TARGET = "IJBC" # change this to IJBB if needed.
BATCH_SIZE = 128
USE_NORM_SCORE = True
USE_DETECTOR_SCORE = True
USE_FLIP_TEST = True

MODELS = [
    #{"name": "mbf", "path": "path/to/model.pt", "network": "mbf"},
    #{"name": "mfn_m", "path": "path/to/model.pt", "network": "mfn_m"},
    #{"name": "mfn_s", "path": "path/to/model.pt", "network": "mfn_s"},
    #{"name": "mfn_xs", "path/to/model.pt", "network": "mfn_xs"},
    #{"name": "r18", "path": "path/to/model.pt", "network": "r18"},
    #{"name": "r50", "path": "path/to/model.pt", "network": "r50"},
    #{"name": "r100", "path": "path/to/model.pt", "network": "r100"},
    #{"name": "sfn0_5", "path": "path/to/model.pt", "network": "sfn0_5"},
    #{"name": "sfn1_0", "path": "path/to/model.pt", "network": "sfn1_0"},
    #{"name": "sfn1_5", "path": "path/to/model.pt", "network": "sfn1_5"},
    #{"name": "sfn2_0", "path": "path/to/model.pt", "network": "sfn2_0"},
    #{"name": "smfn_m", "path": "path/to/model.pt", "network": "smfn_m"},
    #{"name": "smfn_s", "path": "path/to/model.pt", "network": "smfn_s"},
    #{"name": "smfn_xs", "path": "path/to/model.pt", "network": "smfn_xs"},
]

class Embedding:
    def __init__(self, model_path, network, batch_size=1):
        self.image_size = (112, 112)
        print(f"[INIT] Loading model: {model_path} ({network})")
        weight = torch.load(model_path)
        model = get_model(network, dropout=0, fp16=False).cuda()
        model.load_state_dict(weight)
        self.model = torch.nn.DataParallel(model)
        self.model.eval()

        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size

    def preprocess(self, img, landmark):
        if landmark.shape[0] == 68:
            landmark5 = np.array([
                (landmark[36] + landmark[39]) / 2,
                (landmark[42] + landmark[45]) / 2,
                landmark[30],
                landmark[48],
                landmark[54]
            ], dtype=np.float32)
        else:
            landmark5 = landmark

        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        aligned = cv2.warpAffine(img, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        flipped = np.fliplr(aligned)

        aligned = np.transpose(aligned, (2, 0, 1))
        flipped = np.transpose(flipped, (2, 0, 1))

        return np.stack([aligned, flipped])

    @torch.no_grad()
    def forward_batch(self, batch_data):
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()

def read_template_media_list(path):
    meta = pd.read_csv(path, sep=" ", header=None).values
    return meta[:, 1].astype(int), meta[:, 2].astype(int)

def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=" ", header=None).values
    return pairs[:, 0].astype(int), pairs[:, 1].astype(int), pairs[:, 2].astype(int)

def get_image_feature(img_path, files, model_path, network, batch_size):
    data_shape = (3, 112, 112)
    embedding = Embedding(model_path, network, batch_size)
    img_feats = np.empty((len(files), 1024), dtype=np.float32)
    faceness_scores = np.empty(len(files), dtype=np.float32)
    batch_data = np.empty((2 * batch_size, *data_shape))

    print(f"[INFO] Extracting features from {len(files)} images...")

    for batch_start in tqdm(range(0, len(files), batch_size), desc="Extracting Batches"):
        batch_end = min(batch_start + batch_size, len(files))
        current_batch = batch_end - batch_start

        for i, line in enumerate(files[batch_start:batch_end]):
            parts = line.strip().split(" ")
            img_name = os.path.join(img_path, parts[0])
            lmk = np.array([float(x) for x in parts[1:-1]], dtype=np.float32).reshape((5, 2))
            faceness_scores[batch_start + i] = float(parts[-1])

            img = cv2.imread(img_name)
            input_blob = embedding.preprocess(img, lmk)
            batch_data[2 * i] = input_blob[0]
            batch_data[2 * i + 1] = input_blob[1]

        embedding.batch_size = current_batch
        feats = embedding.forward_batch(batch_data[:2 * current_batch])
        img_feats[batch_start:batch_end] = feats

    return img_feats, faceness_scores

def image2template_feature(img_feats, templates, medias):
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]), dtype=np.float32)

    for i, temp in enumerate(tqdm(unique_templates, desc="Aggregating Templates")):
        inds = np.where(templates == temp)[0]
        feats = img_feats[inds]
        medias_for_temp = medias[inds]
        unique_medias = np.unique(medias_for_temp)

        media_feats = []
        for m in unique_medias:
            media_feats.append(np.mean(feats[medias_for_temp == m], axis=0))
        template_feats[i] = np.sum(media_feats, axis=0)

    return normalize(template_feats), unique_templates

def verification(template_norm_feats, unique_templates, p1, p2):
    template2id = np.zeros((max(unique_templates) + 1,), dtype=int)
    for idx, uqt in enumerate(unique_templates):
        template2id[uqt] = idx
    score = np.zeros(len(p1), dtype=np.float32)
    batchsize = 100000

    for i in tqdm(range(0, len(p1), batchsize), desc="Verifying"):
        s = slice(i, min(i + batchsize, len(p1)))
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        score[s] = np.sum(feat1 * feat2, axis=1)
    return score

def evaluate_model(model_name, model_path, network):
    print(f"\n=== Evaluating {model_name} ({network}) ===")

    templates, medias = read_template_media_list(
        os.path.join(DATASET_PATH, "meta", f"{TARGET.lower()}_face_tid_mid.txt"))
    p1, p2, label = read_template_pair_list(
        os.path.join(DATASET_PATH, "meta", f"{TARGET.lower()}_template_pair_label.txt"))
    img_list_path = os.path.join(DATASET_PATH, "meta", f"{TARGET.lower()}_name_5pts_score.txt")

    with open(img_list_path) as f:
        files = f.readlines()

    img_feats, faceness_scores = get_image_feature(
        os.path.join(DATASET_PATH, "loose_crop"), files, model_path, network, BATCH_SIZE)

    if USE_FLIP_TEST:
        img_input_feats = img_feats[:, :img_feats.shape[1]//2] + img_feats[:, img_feats.shape[1]//2:]
    else:
        img_input_feats = img_feats[:, :img_feats.shape[1]//2]

    if not USE_NORM_SCORE:
        img_input_feats /= np.linalg.norm(img_input_feats, axis=1, keepdims=True)

    if USE_DETECTOR_SCORE:
        img_input_feats *= faceness_scores[:, np.newaxis]

    template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)
    score = verification(template_norm_feats, unique_templates, p1, p2)

    os.makedirs(RESULT_DIR, exist_ok=True)
    np.save(os.path.join(RESULT_DIR, f"{model_name}_{TARGET.lower()}.npy"), score)
    print(f"[DONE] Saved scores for {model_name} in {RESULT_DIR}")

    return score, label

if __name__ == "__main__":
    methods = []
    scores_dict = {}
    label = None

    for m in MODELS:
        score, label = evaluate_model(m["name"], m["path"], m["network"])
        scores_dict[m["name"]] = score
        methods.append(m["name"])

    colours = ['blue', 'green', 'red', 'purple', 'orange', 'black']
    x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    tpr_fpr_table = PrettyTable(['Model'] + [f"TPR@{x}" for x in x_labels])

    fig = plt.figure(figsize=(7, 6))
    for i, method in enumerate(methods):
        fpr, tpr, _ = roc_curve(label, scores_dict[method])
        roc_auc = auc(fpr, tpr)
        fpr, tpr = np.flipud(fpr), np.flipud(tpr)
        plt.plot(fpr, tpr, color=colours[i % len(colours)], lw=1.5,
                 label=f"{method} (AUC={roc_auc*100:.2f}%)")

        row = [method]
        for x in x_labels:
            idx = np.argmin(np.abs(fpr - x))
            row.append(f"{tpr[idx]*100:.2f}")
        tpr_fpr_table.add_row(row)

    plt.xscale('log')
    plt.xlim([1e-6, 0.1])
    plt.ylim([0.3, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Comparison on {TARGET}')
    plt.legend(loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(RESULT_DIR, f"{TARGET.lower()}_comparison.pdf")
    fig.savefig(out_path)
    print(f"\n[RESULT] ROC comparison saved to {out_path}")
    print(tpr_fpr_table)
