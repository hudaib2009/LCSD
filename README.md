# Clinical Support Dashboard

Clinical Support Dashboard (`CSD`) is an AI-assisted lung cancer decision-support project built around medical image analysis. This public repository contains the application code, deployment configuration, tests, and a sanitized subset of the training materials.

The system combines:

- `backend/`: FastAPI backend for inference and decision-support workflows
- `frontend/`: Next.js clinician-facing interface
- `tests/`: automated and manual test scaffolding
- `infra/`: deployment and reverse-proxy configuration
- `ml/training/`: cleaned training notebooks and utilities

This public version does not include private datasets, trained model weights, uploaded case storage, generated outputs, or private research assets.

## Background

Lung cancer remains one of the most common and dangerous cancers worldwide. A major reason for its high mortality rate is late detection. In many cases, early-stage disease is difficult to identify from medical imaging, and manual review can be time-consuming and error-prone.

This project was created to explore how artificial intelligence and deep learning can support earlier detection of lung cancer from medical images and help clinicians work faster and more consistently.

## Problem Statement

Diagnosing lung cancer from medical images such as `CT` scans and chest `X-ray` images presents several challenges:

- small lung nodules can be difficult to detect in early stages
- healthy and abnormal tissues may look visually similar
- image quality varies across devices and acquisition settings
- reviewing large numbers of images requires time and sustained attention
- false negatives are especially dangerous because they may delay treatment

The goal of this project is not to replace clinicians, but to provide an intelligent support system that helps improve speed, consistency, and diagnostic confidence.

## Project Objectives

- build a `CNN`-based model for lung-related medical image analysis
- support early detection of possible lung cancer findings
- reduce avoidable diagnostic delay in image review workflows
- create a foundation that can later evolve into a practical clinical support system

## Datasets Used

The project draws from several well-known public datasets in medical imaging:

- `NIH CXR14`: chest X-ray images used for thoracic disease classification
- `LIDC-IDRI`: lung `CT` scans with radiologist annotations
- `LC25000`: histopathology images for tissue-level classification tasks

Using multiple datasets improves diversity and helps the models generalize across different imaging modalities.

## Methodology

### 1. Image Preprocessing

Before training, `CT` data is prepared through a sequence of preprocessing steps:

1. **Original CT Scan**
   A `CT` study begins as a 3D volume made up of many image slices.
2. **Slicing into 2D Images**
   The 3D volume is converted into 2D slices for easier processing with convolutional models.
3. **Lung Segmentation**
   The lung region is isolated to reduce irrelevant background structures.
4. **Low-Pass Filtering**
   Noise is reduced to improve image clarity.
5. **Intensity Normalization**
   Brightness and contrast are standardized across scans.
6. **Resize and Rescale**
   Images are resized to a fixed input size such as `224x224`, and pixel values are rescaled to a model-friendly range.

The output of this pipeline is a cleaned lung image ready for machine learning.

### 2. CNN-Based Modeling

Convolutional Neural Networks (`CNNs`) were chosen because they are highly effective for image analysis and can automatically learn visual features from data.

The model pipeline typically includes:

- `Convolution Layers` for feature extraction
- `ReLU` activations for non-linearity
- `Pooling Layers` for dimensionality reduction
- `Flatten` to convert feature maps into vectors
- `Fully Connected Layers` for final classification
- `Dropout` to reduce overfitting

## Evaluation

Model performance is assessed using standard medical-AI evaluation metrics:

- `Confusion Matrix`
- `Accuracy`
- `Precision`
- `Recall (Sensitivity)`
- `F1-score`
- `ROC Curve`
- `AUC`

### Why Recall Matters

In medical classification tasks, `Recall` is especially important because the most dangerous mistake is:

- `False Negative`: the model predicts that a patient is healthy when the patient is actually diseased

## Example CT Experiment Results

One of the `CT` classification experiments referenced in the project materials reported:

- validation accuracy of about `90.26%`
- test accuracy of `88.18%`
- ROC AUC of `0.9393`

Confusion matrix:

```text
[[618  54]
 [ 48 143]]
```

Classification summary:

```text
Class 0:
  Precision: 0.9279
  Recall:    0.9196
  F1-score:  0.9238

Class 1:
  Precision: 0.7259
  Recall:    0.7487
  F1-score:  0.7371

Overall accuracy: 0.8818
Macro average F1: 0.8304
Weighted average F1: 0.8825
```

These results suggest strong discriminative performance in the experiment, but they do not imply direct clinical readiness without broader validation.

## Results Interpretation

The shared evaluation figures support several important observations:

### CT Confusion Matrix

The `CT` confusion matrix:

```text
[[618  54]
 [ 48 143]]
```

shows that the model correctly identified a large number of both negative and positive cases, while still producing some false positives and false negatives. This is expected in a practical medical-classification setting and reinforces why recall must be tracked carefully.

### CT ROC Curve

The reported `ROC AUC = 0.939` indicates strong class separation ability.  
This suggests that the model is not simply memorizing labels, but is learning useful discriminative signal from the input images.

### Additional Confusion Matrix Example

Another shared confusion matrix:

```text
[[140   4]
 [  2 130]]
```

shows very strong balanced classification performance on that experiment, with only a small number of misclassifications. This kind of result is especially encouraging because both classes were recognized accurately.

### Training vs Validation Loss

The training and validation loss curves show an important pattern:

- training loss keeps decreasing over time
- validation loss improves early, then becomes unstable and increases

This is a classic sign of `Overfitting`. The model continues improving on the training set while generalization becomes less stable on unseen data.

That behavior is one reason why this project emphasizes:

- `Dropout`
- `Early Stopping`
- stronger regularization
- better data diversity
- augmentation and more balanced training data

## Visual Findings Summary

From the figures shared for this project:

- the `CT` classifier demonstrates strong discrimination with high `AUC`
- the confusion matrices show useful predictive performance with manageable error rates
- the loss curves indicate that regularization and validation-driven stopping are important
- the overall results support the use of the model as a decision-support tool, not a standalone diagnostic system

## Overfitting and Model Reliability

One of the key risks in medical imaging models is `Overfitting`, where the model memorizes the training set instead of learning patterns that generalize.

Common causes include:

- limited data volume
- model complexity that is too high for the dataset size
- insufficient diversity in images
- too many training epochs
- class imbalance

Typical mitigation strategies include:

- `Data Augmentation`
- `Dropout`
- `Early Stopping`
- `Regularization`
- simplifying the model architecture
- expanding the dataset with more real-world examples

## Expected Impact

If developed further, this project could support:

- earlier identification of suspicious lung findings
- faster image review workflows
- reduced clinician workload
- fewer avoidable human errors
- better chances of timely treatment through earlier detection

## Future Work

- apply stronger architectures such as `ResNet` and `EfficientNet`
- improve class balance and data quality
- add explainability methods such as `Grad-CAM`
- integrate the system into a more practical clinical workflow
- validate performance across more diverse sources to reduce bias

## Public Repository Scope

This public repository includes:

- `backend/`
- `frontend/`
- `tests/`
- `infra/`
- cleaned `ml/training/`
- core project files:
  `README.md`, `.gitignore`, `.env.example`, `requirements.txt`, `main.py`, `docker-compose.yml`, `docker-compose.prod.yml`

### Excluded From The Public Repo

- `ml/data/`
- `ml/models/`
- `ml/artifacts/`
- `frontend/storage/`
- patient data or PHI-like content
- generated metrics, figures, checkpoints, and trained artifacts
- local environment files, secrets, and private assets

## Local Development

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

Create a local environment file:

```bash
cp .env.example .env
```

Important variables:

- `FASTAPI_BASE_URL`
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`
- `OPENROUTER_FALLBACK_MODEL`
- `OPENROUTER_FALLBACK_MODEL_2`
- `CSD_USE_GPU`
- `CSD_CT_MODEL_PATH`
- `CSD_XRAY_MODEL_PATH`
- `CSD_PATHOLOGY_MODEL_PATH`

Model files are not bundled in this public repository. If you want to run real inference, you must provide those artifacts separately.

## Deployment

For local multi-service startup:

```bash
docker compose up --build
```

For the production stack:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

## Testing

```bash
pytest tests
```

## Conclusion

This project presents a deep-learning-based approach to medical image analysis for lung cancer support workflows. Its purpose is to assist specialists, accelerate review, and improve consistency in early detection efforts while remaining clearly positioned as a support tool rather than a replacement for clinical judgment.
