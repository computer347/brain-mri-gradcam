# Brain Tumor MRI Classification with Grad-CAM

A convolutional neural network that classifies brain MRI scans into four categories —
**glioma**, **meningioma**, **pituitary tumor**, and **no tumor** — and uses
**Grad-CAM** to visualize which regions of the scan drove each prediction.

The goal is not only accuracy but *explainability*: a heatmap overlay lets a viewer
check whether the model is looking at the tumor or at irrelevant parts of the image.

## How it works

1. Loads a class-balanced sample of images (`N_PER_CLASS`, default 200) from `Training/` and `Testing/`.
2. Resizes to 224×224, normalizes to [0, 1].
3. Trains a compact CNN (3 conv blocks → dense head) with Adam + sparse categorical cross-entropy.
4. Plots training vs. validation accuracy.
5. Runs Grad-CAM on the last convolutional layer and shows a 3×3 grid of test scans with
   heatmap overlays, true label, predicted label and confidence (green = correct, red = wrong).

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle).
The images are not included in this repository. Download and unpack so the layout is:

```
Training/{glioma,meningioma,notumor,pituitary}/*.jpg
Testing/{glioma,meningioma,notumor,pituitary}/*.jpg
```

next to `MRI_Project_Exploration.py`.

## Run

```bash
pip install -r requirements.txt
python MRI_Project_Exploration.py
```

## Next steps

- Train on the full dataset with a separate validation split (keep `Testing/` held out)
- Data augmentation and transfer learning (e.g. EfficientNet / ResNet backbones)
- Confusion matrix and per-class precision/recall
- Simple web demo: upload a scan → prediction + Grad-CAM overlay

## Disclaimer

Research/educational project only. Not a medical device and not intended for clinical diagnosis.
