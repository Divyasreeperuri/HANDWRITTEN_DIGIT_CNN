# DigitAI — Handwritten Digit Recognition Using CNN: Technical Project Report

## 1. Executive Summary
**DigitAI** is an end-to-end handwritten digit recognition web application capable of classifying single digits (`0–9`), double-digit numbers (`10–99`), and multi-digit sequences (`100+`, e.g., `2026`, `987654`) from live pointer/touch/stylus canvas strokes and uploaded images (`PNG`, `JPG`, `JPEG`).

Instead of treating multi-digit images as a single MNIST sample, DigitAI combines an **OpenCV contour and vertical-projection segmentation pipeline** with a **TensorFlow/Keras Convolutional Neural Network (CNN)** trained on MNIST (`60,000` training / `10,000` test images). After addressing startup loading and runtime initialization, both the Flask + Keras backend and the Express + TypeScript CNN runtime automatically resolve project-relative model paths, support `.keras`, `.h5`, `.weights.h5`, and `cnn_weights.json` artifacts, automatically trigger MNIST training if trained weights are absent, block inference until ready (`503`), and log full backend diagnostics safely.

---

## 2. Measured Model Performance & Comparative Evaluation

All metrics below are directly measured on the official MNIST test dataset (`10,000` images) and verified via `models/model_metrics.json` and `models/evaluation_report.json`:

| Metric | Baseline CNN (2 epochs) | Enhanced CNN (4 epochs, OpenCV Aug + BN + Dropout) |
| --- | ---: | ---: |
| **MNIST Test Accuracy (10,000 samples)** | `97.89%` | **`99.12%`** |
| **MNIST Test Loss** | `0.0668` | **`0.0280`** |
| **Macro Precision** | `97.88%` | **`99.13%`** |
| **Macro Recall** | `97.89%` | **`99.10%`** |
| **Macro F1-Score** | `97.87%` | **`99.11%`** |
| **Trainable/Total Parameters** | `105,866` | `238,186` |
| **Training Wall Time (2 CPU threads)** | `37.58 s` | `181.83 s` |
| **Batch Inference Latency** | `0.2777 ms / digit` | `0.3529 ms / digit` |
| **Single-Digit Inference Latency** | — | `12.2393 ms` |

### Standalone 2,000-Sample MNIST Evaluation (`training/evaluate_model.py`)
- **Samples Evaluated**: `2,000` MNIST test digits
- **Accuracy**: `98.75%` (`0.9875`)
- **Loss**: `0.037436`
- **Macro Precision / Recall / F1**: `98.78%` / `98.74%` / `98.75%`
- **Inference Throughput**: `634.31 ms` total (`0.3172 ms / digit`)

### Multi-Digit Synthetic Evaluation (`training/evaluate_model.py`)
| Expected Number | Mode | Background Polarity | Predicted Output | Confidence | Processing Time |
| --- | --- | --- | --- | ---: | ---: |
| `7` | Single | Dark | `7` | `99.99%` | `73.78 ms` |
| `48` | Double | White paper | `48` | `99.79%` | `71.69 ms` |
| `96` | Double | Dark | `96` | `99.54%` | `71.44 ms` |
| `2026` | Multi | White paper | `2026` | `99.68%` | `80.10 ms` |
| `987654` | Multi | White paper | `987654` | `99.56%` | `75.86 ms` |

---

## 3. Enhanced CNN Architecture (`DigitAI_Enhanced_CNN`)

Input tensor shape: `(28, 28, 1)` normalized to `[0.0, 1.0]`.

1. `Conv2D(24, 3×3, padding='same', activation='relu')` -> `BatchNormalization` -> `MaxPooling2D(2×2)` -> `Dropout(0.20)`
2. `Conv2D(48, 3×3, padding='same', activation='relu')` -> `BatchNormalization` -> `MaxPooling2D(2×2)` -> `Dropout(0.25)`
3. `Flatten()` -> `2,352` features
4. `Dense(96, activation='relu')` -> `BatchNormalization` -> `Dropout(0.35)`
5. `Dense(10, activation='softmax')` -> 10 digit class probabilities (`0–9`)

Training uses **Adam** (`lr=1.4e-3`), **Sparse Categorical Cross-Entropy**, **EarlyStopping** (`monitor='val_accuracy'`, `patience=3`), **ReduceLROnPlateau** (`factor=0.5`, `patience=2`), **ModelCheckpoint**, and OpenCV affine augmentation (rotations `±9°`, scale `0.92–1.08`, translations `±2 px`).

---

## 4. Image Preprocessing & Segmentation Pipeline

1. **Safe Validation & Transparency Handling**: Validates payload headers/dimensions (`PNG`, `JPG`, `JPEG`, `WEBP`, `BMP`) and composites RGBA canvas transparency onto an appropriate background before grayscale conversion.
2. **Polarity Auto-Inversion**: Inspects border and global pixel medians to convert black-on-white paper scans into MNIST's bright-stroke-on-dark-background format.
3. **Denoising & Binarization**: Applies Gaussian smoothing, morphological open background subtraction (removing uneven paper shadows), and Otsu adaptive thresholding.
4. **Contour Extraction & Fragment Merging**: Extracts external contours (`cv2.findContours`), filters micro-noise, and merges vertically aligned broken strokes (such as two-stroke `5` or `4`).
5. **Touching-Digit Splitting**: Wide merged contours (`aspect > 1.25` or single-region double-digit mode) are recursively split at smoothed vertical-projection minima.
6. **Left-to-Right Sorting & MNIST Normalization**: Each digit crop is aspect-preserving resized into `20×20`, centered by center-of-mass inside `28×28`, peak-normalized, and classified in batch by the CNN.

---

## 5. Segmentation Limitations & Sequence-Recognition Alternative

Explicit contour + vertical-projection segmentation assumes digits can be separated horizontally along a vertical cut line. It can struggle when:
- Handwritten digits slant strongly (`45°+`) or overlap horizontally so no clear vertical projection valley exists.
- Heavy cursive ligatures, smudges, or ruled notebook lines connect adjacent digits.
- Multi-line numbers or extreme background clutter appear in the same crop.

**Alternative End-to-End Sequence Approach**: Replace explicit segmentation with a **CRNN (Convolutional Recurrent Neural Network) trained with Connectionist Temporal Classification (CTC) loss**, or a **CNN + Transformer/Attention decoder**. In a CRNN+CTC pipeline, convolutional feature maps across image width are fed directly into bidirectional LSTMs (`BiLSTM`) and a softmax+CTC layer, allowing the network to decode arbitrary-length digit sequences without brittle bounding-box cuts.

---

## 6. Model Lifecycle, Initialization & Error Handling

- **Reliable Absolute Paths**: Both `backend/model_service.py` and `src/server/cnnEngine.ts` resolve `models/` relative to project root (`Path(__file__).resolve().parent.parent`), avoiding working-directory mismatches.
- **Multi-Format Support**: Loads `digitai_cnn.keras`, falls back to `digitai_cnn.h5` (including `compile=False` compatibility retry), reconstructs the CNN from `digitai_cnn.weights.h5` or `cnn_weights.json`, or automatically runs MNIST training in `.venv` if no model exists.
- **Readiness Gate & Clear User Errors**: `/api/health` exposes live model readiness (`loading`, `training`, `ready`, `error`). Predictions are disabled in the UI and blocked server-side (`HTTP 503`) until ready, returning clear status messages (`"Initializing CNN model."`, `"Training CNN for the first time."`, `"CNN model ready."`, `"CNN model initialization failed."`, `"Please upload a valid handwritten digit image."`, `"No handwritten digits detected."`, `"Unable to process this image. Please try another image."`) while logging full diagnostics server-side.
- **Verified Automated Suite**: `pytest tests/test_digitai.py -v` passes all `9/9` unit and integration checks.
