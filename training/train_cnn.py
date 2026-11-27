"""Train Baseline and Enhanced CNN models on MNIST, evaluate, and export artifacts."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow import keras
from tensorflow.keras import layers

from backend.model_service import ensure_sample_gallery

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR = ROOT_DIR / "static" / "plots"


def build_baseline_cnn() -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="base_conv1"),
            layers.MaxPooling2D((2, 2), name="base_pool1"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="base_conv2"),
            layers.MaxPooling2D((2, 2), name="base_pool2"),
            layers.Flatten(name="base_flatten"),
            layers.Dense(64, activation="relu", name="base_dense1"),
            layers.Dense(10, activation="softmax", name="base_output"),
        ],
        name="DigitAI_Baseline_CNN",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_enhanced_cnn() -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(24, (3, 3), activation="relu", padding="same", name="conv1_1"),
            layers.BatchNormalization(name="bn1_1"),
            layers.MaxPooling2D((2, 2), name="pool1"),
            layers.Dropout(0.2, name="drop1"),

            layers.Conv2D(48, (3, 3), activation="relu", padding="same", name="conv2_1"),
            layers.BatchNormalization(name="bn2_1"),
            layers.MaxPooling2D((2, 2), name="pool2"),
            layers.Dropout(0.25, name="drop2"),

            layers.Flatten(name="flatten"),
            layers.Dense(96, activation="relu", name="dense1"),
            layers.BatchNormalization(name="bn_dense"),
            layers.Dropout(0.35, name="drop_dense"),
            layers.Dense(10, activation="softmax", name="output"),
        ],
        name="DigitAI_Enhanced_CNN",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1.4e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def augment_dataset_opencv(images: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(images)
    out = np.empty_like(images)
    angles = rng.uniform(-9.0, 9.0, size=n)
    scales = rng.uniform(0.92, 1.08, size=n)
    txs = rng.uniform(-2.0, 2.0, size=n)
    tys = rng.uniform(-2.0, 2.0, size=n)

    for i in range(n):
        mat = cv2.getRotationMatrix2D((14.0, 14.0), float(angles[i]), float(scales[i]))
        mat[0, 2] += float(txs[i])
        mat[1, 2] += float(tys[i])
        out[i, :, :, 0] = cv2.warpAffine(images[i, :, :, 0], mat, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return out


def export_cnn_weights_json(model: keras.Model, output_path: Path) -> None:
    exported_layers = []
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            w, b = layer.get_weights()
            exported_layers.append(
                {
                    "type": "conv2d",
                    "name": layer.name,
                    "kernel_size": list(layer.kernel_size),
                    "padding": layer.padding,
                    "activation": "relu",
                    "weights": np.round(w, 6).tolist(),
                    "bias": np.round(b, 6).tolist(),
                }
            )
        elif isinstance(layer, layers.BatchNormalization):
            gamma, beta, mean, var = layer.get_weights()
            exported_layers.append(
                {
                    "type": "batch_norm",
                    "name": layer.name,
                    "epsilon": float(layer.epsilon),
                    "gamma": np.round(gamma, 6).tolist(),
                    "beta": np.round(beta, 6).tolist(),
                    "mean": np.round(mean, 6).tolist(),
                    "variance": np.round(var, 6).tolist(),
                }
            )
        elif isinstance(layer, layers.MaxPooling2D):
            exported_layers.append({"type": "maxpool2d", "name": layer.name, "pool_size": list(layer.pool_size)})
        elif isinstance(layer, layers.Flatten):
            exported_layers.append({"type": "flatten", "name": layer.name})
        elif isinstance(layer, layers.Dense):
            w, b = layer.get_weights()
            act = "softmax" if layer.name == "output" else "relu"
            exported_layers.append(
                {
                    "type": "dense",
                    "name": layer.name,
                    "activation": act,
                    "weights": np.round(w, 6).tolist(),
                    "bias": np.round(b, 6).tolist(),
                }
            )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"model_name": model.name, "layers": exported_layers}, f)


def _plot_training_curves(base_hist: dict, enh_hist: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    epochs_e = range(1, len(enh_hist["accuracy"]) + 1)
    epochs_b = range(1, len(base_hist["accuracy"]) + 1)

    axes[0].plot(epochs_e, enh_hist["accuracy"], label="Enhanced Train Acc", color="#0284c7", linewidth=2)
    axes[0].plot(epochs_e, enh_hist["val_accuracy"], label="Enhanced Val Acc", color="#0f766e", linewidth=2)
    axes[0].plot(epochs_b, base_hist["val_accuracy"], label="Baseline Val Acc", color="#64748b", linestyle="--")
    axes[0].set_title("Model Accuracy Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs_e, enh_hist["loss"], label="Enhanced Train Loss", color="#0284c7", linewidth=2)
    axes[1].plot(epochs_e, enh_hist["val_loss"], label="Enhanced Val Loss", color="#0f766e", linewidth=2)
    axes[1].plot(epochs_b, base_hist["val_loss"], label="Baseline Val Loss", color="#64748b", linestyle="--")
    axes[1].set_title("Model Loss Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sparse Categorical Cross-Entropy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=140)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
        title="MNIST Test Confusion Matrix (Enhanced CNN)",
        ylabel="True Digit",
        xlabel="Predicted Digit",
    )
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "#0f172a",
                fontsize=8.5,
            )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate(epochs_baseline: int = 2, epochs_enhanced: int = 4) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(42)
    np.random.seed(42)

    (x_train_raw, y_train), (x_test_raw, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train_raw.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test_raw.astype("float32") / 255.0)[..., np.newaxis]

    # Generate sample gallery using the already-loaded MNIST test images
    ensure_sample_gallery((x_test_raw, y_test))
    del x_train_raw, x_test_raw

    x_val, y_val = x_train[-6000:], y_train[-6000:]
    x_tr, y_tr = x_train[:-6000], y_train[:-6000]
    batch_size = 256

    print("[DigitAI] Training Baseline CNN...", flush=True)
    baseline_model = build_baseline_cnn()
    t0 = time.perf_counter()
    base_history = baseline_model.fit(
        x_tr,
        y_tr,
        validation_data=(x_val, y_val),
        epochs=epochs_baseline,
        batch_size=batch_size,
        verbose=2,
    )
    base_train_time = time.perf_counter() - t0
    base_loss, base_acc = baseline_model.evaluate(x_test, y_test, batch_size=512, verbose=0)

    t_inf0 = time.perf_counter()
    _ = baseline_model.predict(x_test[:1000], batch_size=256, verbose=0)
    base_inf_ms = ((time.perf_counter() - t_inf0) / 1000.0) * 1000.0
    base_preds_full = np.argmax(baseline_model.predict(x_test, batch_size=512, verbose=0), axis=1)
    base_p, base_r, base_f1, _ = precision_recall_fscore_support(y_test, base_preds_full, average="macro", zero_division=0)
    base_params = int(baseline_model.count_params())
    base_hist_clean = {
        k: [round(float(v), 6) for v in vals]
        for k, vals in base_history.history.items()
        if k in {"accuracy", "val_accuracy", "loss", "val_loss"}
    }
    del baseline_model
    keras.backend.clear_session()
    gc.collect()

    print("[DigitAI] Applying OpenCV data augmentation (rotations, translations, zoom)...", flush=True)
    x_tr_aug = x_tr.copy()
    x_tr_aug[:18000] = augment_dataset_opencv(x_tr[:18000], seed=42)

    print("[DigitAI] Training Enhanced CNN...", flush=True)
    enhanced_model = build_enhanced_cnn()
    ckpt_path = MODELS_DIR / "digitai_cnn.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path), monitor="val_accuracy", save_best_only=True),
    ]

    t1 = time.perf_counter()
    enh_history = enhanced_model.fit(
        x_tr_aug,
        y_tr,
        validation_data=(x_val, y_val),
        epochs=epochs_enhanced,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )
    enh_train_time = time.perf_counter() - t1
    del x_tr_aug, x_train
    gc.collect()

    best_model = keras.models.load_model(ckpt_path)
    best_model.save_weights(str(MODELS_DIR / "digitai_cnn.weights.h5"))
    enh_loss, enh_acc = best_model.evaluate(x_test, y_test, batch_size=512, verbose=0)

    t_inf1 = time.perf_counter()
    _ = best_model.predict(x_test[:1000], batch_size=256, verbose=0)
    enh_inf_ms_batch = ((time.perf_counter() - t_inf1) / 1000.0) * 1000.0

    t_single = time.perf_counter()
    for i in range(100):
        _ = best_model(x_test[i : i + 1], training=False)
    enh_inf_ms_single = ((time.perf_counter() - t_single) / 100.0) * 1000.0

    enh_preds = np.argmax(best_model.predict(x_test, batch_size=512, verbose=0), axis=1)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, enh_preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, enh_preds)
    per_class = classification_report(y_test, enh_preds, output_dict=True, zero_division=0)

    enh_hist_clean = {
        k: [round(float(v), 6) for v in vals]
        for k, vals in enh_history.history.items()
        if k in {"accuracy", "val_accuracy", "loss", "val_loss"}
    }
    _plot_training_curves(base_hist_clean, enh_hist_clean, PLOTS_DIR / "training_curves.png")
    _plot_confusion_matrix(cm, PLOTS_DIR / "confusion_matrix.png")

    export_cnn_weights_json(best_model, MODELS_DIR / "cnn_weights.json")

    metrics_payload = {
        "model_name": "DigitAI Enhanced CNN",
        "dataset": "MNIST (60,000 train / 10,000 test)",
        "input_shape": [28, 28, 1],
        "num_classes": 10,
        "parameters": int(best_model.count_params()),
        "test_samples": int(len(y_test)),
        "enhanced_cnn": {
            "test_accuracy": round(float(enh_acc), 6),
            "test_loss": round(float(enh_loss), 6),
            "precision_macro": round(float(p_macro), 6),
            "recall_macro": round(float(r_macro), 6),
            "f1_macro": round(float(f1_macro), 6),
            "training_seconds": round(float(enh_train_time), 2),
            "batch_inference_ms_per_digit": round(float(enh_inf_ms_batch), 4),
            "single_inference_ms": round(float(enh_inf_ms_single), 4),
            "epochs_run": len(enh_hist_clean["accuracy"]),
            "history": enh_hist_clean,
        },
        "baseline_cnn": {
            "test_accuracy": round(float(base_acc), 6),
            "test_loss": round(float(base_loss), 6),
            "precision_macro": round(float(base_p), 6),
            "recall_macro": round(float(base_r), 6),
            "f1_macro": round(float(base_f1), 6),
            "training_seconds": round(float(base_train_time), 2),
            "batch_inference_ms_per_digit": round(float(base_inf_ms), 4),
            "parameters": base_params,
            "epochs_run": len(base_hist_clean["accuracy"]),
            "history": base_hist_clean,
        },
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            str(d): {
                "precision": round(float(per_class[str(d)]["precision"]), 4),
                "recall": round(float(per_class[str(d)]["recall"]), 4),
                "f1_score": round(float(per_class[str(d)]["f1-score"]), 4),
                "support": int(per_class[str(d)]["support"]),
            }
            for d in range(10)
        },
        "architecture_layers": [
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(getattr(layer, "output_shape", None) or getattr(getattr(layer, "output", None), "shape", "")),
                "params": int(layer.count_params()),
            }
            for layer in best_model.layers
        ],
    }

    with open(MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(
        f"[DigitAI] Training complete | Baseline Acc: {base_acc:.4%} | Enhanced Acc: {enh_acc:.4%} | "
        f"Macro F1: {f1_macro:.4%}",
        flush=True,
    )
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate()
