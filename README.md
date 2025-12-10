# ArcFace Backbone Comparative

PyTorch-based research repository focused on comparing lightweight facial recognition backbones under a unified ArcFace training pipeline. This project is a simplified and modernized adaptation of the original InsightFace implementation, restructured for single-GPU training and PyTorch-only execution, while preserving the original system’s behavior and evaluation methodology.

Developed as part of a research study exploring the performance of lightweight architectures for face recognition. 

Check out the [paper](https://github.com/ItO210/arcface-backbone-comparative/blob/6d1f08b3231823101d965f00a5c9dae01a1bf92c/Lightweight%20Backbone%20Evaluation%20for%20Face%20Recognition%20with%20ArcFace.pdf)


## Disclaimer

This repository is a simplified adaptation of the original InsightFace framework. It has been refactored for educational and research use only. No datasets are included due to size restrictions.

## Features

- **Unified ArcFace Pipeline**: Single-GPU, fully PyTorch implementation with FP16 support.
- **Lightweight Backbones**: Includes iResNet, MixFaceNet, MobileFaceNet, and ShuffleFaceNet (all with 512-D embeddings).
- **Configurable Training**: Modular config files for each backbone architecture.
- **Dataset Conversion Tools**: Convert original RecordIO datasets to PyTorch-friendly image folders.
- **Multi-Model Evaluation**: Compare multiple trained models on IJB-B or IJB-C datasets.
- **Model Complexity Analysis**: Compute FLOPs and parameter counts for all backbones.
- **Visualization Scripts**: Generate ROC curves with AUC percentages, and training/validation metrics.

## Contents

- **backbones/**: Implementations of iResNet, MixFaceNet, MobileFaceNet, and ShuffleFaceNet.
- **configs/**: configuration files for training each backbone.
- **datasets/**:
  - `convert_recordio_to_images.py`: Converts RecordIO datasets to image folders.
  - `.bin` files for verification (AgeDB-30, CFP-FP, LFW).
- **work_dirs/**:
  - `extract.py`: Parses training logs into JSON metrics.
  - `combined_training_plot.py`: Plots loss, learning rate, and FP16 scale vs steps.
  - `combined_validation_plot.py`: Plots validation accuracy for LFW, CFP-FP, and AgeDB-30.
- **multiModel_eval.py**: Evaluates multiple backbones on IJB-B/IJB-C and saves results as .npy.
- **multiModel_replot.py**: Plots IJB-B/IJB-C ROC curves and AUC% from saved .npy results.
- **flops.py**: Uses the ptflops library to calculate and display each model’s GFLOPs and parameter count (M) in a formatted table.

## Usage

### Train a model

```bash
python train.py configs/{configFileName}
```
Replace {configFileName} with the configuration of the backbone you want to train (e.g., iresnet100.py).

### Evaluate Models on IJB-B/IJB-C

Evaluation is controlled inside the script `multiModel_eval.py`:
1. Open multiModel_eval.py and define:
- The models to evaluate.
- Whether to evaluate on IJB-B, IJB-C.
2. Run the script:
```bash
python multiModel_eval.py
```
- This will evaluate the specified models and generate .npy files.
- ROC curves with AUC% plots are created automatically for the evaluated models.

### Plot Multiple Models Together

If you want to compare multiple models in a single plot and dont want to run the evaluation again, use `multiModel_replot.py`:
1. Open multiModel_replot.py and define the .npy files or models to include in the plot.
2. Run the script:
```bash
python multiModel_replot.py
```
- This generates ROC curves and AUC% plots for all defined models.

### Compute FLOPs and Parameter Count

Model complexity is measured using `flops.py`:
Define the models to analyze inside the script.
Run:
```bash
python flops.py
```
- Outputs a table showing GFLOPs and parameters (M) for each defined backbone.

### Training and Validation Plots

#### Extract metrics from the `.log` file generated during training using `extract.py`:
1. Define the models to analyze inside the script.
2. Run:
```bash
python work_dirs/extract.py
```
- Produces:
  - `training_metrics.json`: Loss, LearningRate, Epoch, GlobalStep, Fp16GradScale
  - `validation_metrics.json`: XNorm, Accuracy-Flip, Accuracy-Highest on LFW, CFP-FP, AgeDB-30
#### Plot training metrics with `combined_training_plot.py`:
1. Define the paths to the `training_metrics.json` files for the models you want to plot inside the script.
2. Run:
```bash
python work_dirs/combined_training_plot.py
```
- Generates plots for:
  - FP16 Grad Scale vs Global Step
  - Learning Rate vs Global Step
  - Loss vs Global Step
#### Plot validation metrics with `combined_validation_plot.py`:
1. Define the paths to the `validation_metrics.json` files inside the script.
2. Run:
```bash
python work_dirs/combined_validation_plot.py
```
- Generates plots for:
  - AgeDB-30 Accuracy vs Step
  - CFP-FP Accuracy vs Step
  - LFW Accuracy vs Step
