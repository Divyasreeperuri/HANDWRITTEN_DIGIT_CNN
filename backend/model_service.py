"""Model loading, automatic initialization, training fallback, and inference service for DigitAI."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow import keras
from tensorflow.keras import layers

from backend.preprocessing import (
    PreprocessingResult,
    decode_base64_image,
    decode_image_bytes,
    preprocess_and_segment,
)

logger = logging.getLogger("digitai.model_service")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = (ROOT_DIR / "models").resolve()
KERAS_MODEL_PATH = MODELS_DIR / "digitai_cnn.keras"
H5_MODEL_PATH = MODELS_DIR / "digitai_cnn.h5"
WEIGHTS_H5_PATH = MODELS_DIR / "digitai_cnn.weights.h5"
WEIGHTS_JSON_PATH = MODELS_DIR / "cnn_weights.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
SAMPLES_DIR = (ROOT_DIR / "static" / "samples").resolve()

LOW_CONFIDENCE_THRESHOLD = 0.72


class ModelNotReadyError(RuntimeError):
    """Raised when prediction or evaluation is requested while model initialization/training is running."""

    def __init__(self, message: str, state: str = "loading"):
        super().__init__(message)
        self.state = state


class ModelInitializationError(RuntimeError):
    """Raised when the CNN model failed to load and could not be trained."""


def build_enhanced_cnn_architecture() -> keras.Model:
    """Recreate the exact Enhanced CNN architecture for weights-only loading."""
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


def _verify_model_io(model: keras.Model) -> None:
    """Verify the loaded CNN matches (28, 28, 1) -> 10 digit classes and executes cleanly."""
    in_shape = tuple(model.input_shape)
    out_shape = tuple(model.output_shape)
    if len(in_shape) != 4 or in_shape[1:] != (28, 28, 1):
        raise ValueError(f"Incompatible CNN input shape {in_shape}; expected (None, 28, 28, 1).")
    if len(out_shape) != 2 or out_shape[1] != 10:
        raise ValueError(f"Incompatible CNN output shape {out_shape}; expected (None, 10).")
    probe = np.zeros((1, 28, 28, 1), dtype=np.float32)
    out = model(probe, training=False).numpy()
    if out.shape != (1, 10) or not np.isfinite(out).all():
        raise ValueError("Loaded CNN model produced invalid inference output.")


def _load_from_weights_json(json_path: Path) -> keras.Model:
    """Recreate the matching CNN architecture and populate weights from exported JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    exported_layers = payload.get("layers")
    if not isinstance(exported_layers, list) or not exported_layers:
        raise ValueError("Invalid cnn_weights.json structure: missing layers list.")

    model = build_enhanced_cnn_architecture()
    layer_by_name = {layer.name: layer for layer in model.layers}

    for spec in exported_layers:
        name = spec.get("name")
        ltype = spec.get("type")
        if name not in layer_by_name:
            raise ValueError(f"Layer '{name}' in cnn_weights.json does not match CNN architecture.")
        layer = layer_by_name[name]
        if ltype in ("conv2d", "dense"):
            w = np.array(spec["weights"], dtype=np.float32)
            b = np.array(spec["bias"], dtype=np.float32)
            layer.set_weights([w, b])
        elif ltype == "batch_norm":
            gamma = np.array(spec["gamma"], dtype=np.float32)
            beta = np.array(spec["beta"], dtype=np.float32)
            mean = np.array(spec["mean"], dtype=np.float32)
            var = np.array(spec["variance"], dtype=np.float32)
            layer.set_weights([gamma, beta, mean, var])

    _verify_model_io(model)
    return model


class ModelService:
    """Thread-safe lifecycle manager for CNN loading, automatic training, and inference."""

    def __init__(
        self,
        model_path: Path = KERAS_MODEL_PATH,
        metrics_path: Path = METRICS_PATH,
        auto_initialize: bool = True,
        async_init: bool = False,
        auto_train: bool = True,
    ):
        self.model_path = Path(model_path).resolve()
        self.models_dir = self.model_path.parent
        self.h5_path = self.models_dir / "digitai_cnn.h5"
        self.weights_h5_path = self.models_dir / "digitai_cnn.weights.h5"
        self.weights_json_path = self.models_dir / "cnn_weights.json"
        self.metrics_path = Path(metrics_path).resolve()

        self._lock = threading.RLock()
        self._init_thread: threading.Thread | None = None
        self.auto_train = auto_train

        self.model: keras.Model | None = None
        self.metrics: dict[str, Any] = {}
        self.state: str = "loading"  # loading | training | ready | error
        self.status_message: str = "Initializing CNN model."
        self.model_format: str | None = None
        self.active_model_path: Path | None = None
        self.last_error: str | None = None

        if auto_initialize:
            if async_init:
                self.start_background_init()
            else:
                self.initialize(auto_train=auto_train)

    def start_background_init(self) -> None:
        """Start asynchronous model initialization once at application startup."""
        with self._lock:
            if self.state == "ready" and self.model is not None:
                return
            if self._init_thread is not None and self._init_thread.is_alive():
                return
            self.state = "loading"
            self.status_message = "Initializing CNN model."
            self._init_thread = threading.Thread(
                target=self._run_init_safe,
                name="DigitAI-ModelInit",
                daemon=True,
            )
            self._init_thread.start()

    def _run_init_safe(self) -> None:
        try:
            self.initialize(auto_train=self.auto_train)
        except Exception:
            logger.exception("Background CNN model initialization failed.")

    def is_ready(self) -> bool:
        return self.state == "ready" and self.model is not None

    def _load_metrics_file(self) -> None:
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception:
                logger.exception("Failed to read model metrics file at %s", self.metrics_path)

    def _try_load_existing_model(self) -> tuple[keras.Model, str, Path] | None:
        """Attempt loading saved .keras, .h5, .weights.h5, or cnn_weights.json artifacts."""
        self.models_dir.mkdir(parents=True, exist_ok=True)

        full_model_candidates: list[tuple[Path, str]] = [
            (self.model_path, "keras"),
            (self.h5_path, "h5"),
        ]
        for candidate, fmt in full_model_candidates:
            if not candidate.exists() or candidate.stat().st_size == 0:
                continue
            try:
                logger.info("Loading saved CNN model (%s) from %s", fmt, candidate)
                try:
                    loaded = keras.models.load_model(candidate)
                except Exception as load_exc:
                    logger.warning(
                        "Standard load_model failed for %s (%s); retrying with compile=False.",
                        candidate,
                        load_exc,
                    )
                    loaded = keras.models.load_model(candidate, compile=False)
                    loaded.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=1.4e-3),
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"],
                    )
                _verify_model_io(loaded)
                return loaded, fmt, candidate
            except Exception:
                logger.exception("Failed loading full CNN model from %s", candidate)

        if self.weights_h5_path.exists() and self.weights_h5_path.stat().st_size > 0:
            try:
                logger.info("Reconstructing CNN architecture and loading weights from %s", self.weights_h5_path)
                model = build_enhanced_cnn_architecture()
                model.load_weights(str(self.weights_h5_path))
                _verify_model_io(model)
                model.save(self.model_path)
                return model, "weights.h5", self.weights_h5_path
            except Exception:
                logger.exception("Failed loading CNN weights from %s", self.weights_h5_path)

        if self.weights_json_path.exists() and self.weights_json_path.stat().st_size > 0:
            try:
                logger.info("Reconstructing CNN architecture from exported weights JSON at %s", self.weights_json_path)
                model = _load_from_weights_json(self.weights_json_path)
                model.save(self.model_path)
                model.save_weights(str(self.weights_h5_path))
                return model, "cnn_weights.json", self.weights_json_path
            except Exception:
                logger.exception("Failed reconstructing CNN from %s", self.weights_json_path)

        return None

    def initialize(self, auto_train: bool = True) -> None:
        """Load saved CNN model/weights or train automatically on MNIST if unavailable."""
        with self._lock:
            self.state = "loading"
            self.status_message = "Initializing CNN model."
            self.last_error = None

        found = self._try_load_existing_model()
        if found is not None:
            model, fmt, src_path = found
            with self._lock:
                self.model = model
                self.model_format = fmt
                self.active_model_path = src_path
                self._load_metrics_file()
                if not self.weights_h5_path.exists():
                    try:
                        model.save_weights(str(self.weights_h5_path))
                    except Exception:
                        logger.warning("Could not write fallback weights.h5 file.", exc_info=True)
                if not self.weights_json_path.exists():
                    try:
                        from training.train_cnn import export_cnn_weights_json

                        export_cnn_weights_json(model, self.weights_json_path)
                    except Exception:
                        logger.warning("Could not export cnn_weights.json fallback file.", exc_info=True)
                self.state = "ready"
                self.status_message = "CNN model ready."
                logger.info("DigitAI CNN ready (source=%s, format=%s)", src_path, fmt)
            return

        if not auto_train:
            with self._lock:
                self.state = "error"
                self.status_message = "CNN model initialization failed."
                self.last_error = f"No compatible CNN model or weights found in {self.models_dir}."
            raise FileNotFoundError(self.last_error)

        with self._lock:
            self.state = "training"
            self.status_message = "Training CNN for the first time."
            logger.warning("No compatible CNN model found in %s; starting automatic MNIST training.", self.models_dir)

        try:
            with self._lock:
                self.status_message = "Training CNN for the first time."
            from training.train_cnn import train_and_evaluate

            train_and_evaluate()
            reloaded = self._try_load_existing_model()
            if reloaded is None:
                raise ModelInitializationError("CNN model initialization failed.")
            model, fmt, src_path = reloaded
            with self._lock:
                self.model = model
                self.model_format = fmt
                self.active_model_path = src_path
                self._load_metrics_file()
                self.state = "ready"
                self.status_message = "CNN model ready."
                logger.info("Automatic MNIST training finished and model reloaded from %s", src_path)
        except Exception as exc:
            logger.exception("Automatic CNN training failed.")
            with self._lock:
                self.model = None
                self.state = "error"
                self.last_error = str(exc)
                self.status_message = "CNN model initialization failed."
            raise ModelInitializationError(self.status_message) from exc

    def load(self) -> None:
        """Backward-compatible entry point to initialize/reload the model."""
        self.initialize(auto_train=self.auto_train)

    def _require_ready(self) -> None:
        if self.state == "ready" and self.model is not None:
            return
        if self.state == "loading":
            raise ModelNotReadyError("Initializing CNN model.", state="loading")
        if self.state == "training":
            raise ModelNotReadyError("Training CNN for the first time.", state="training")
        if self.auto_train:
            self.start_background_init()
            raise ModelNotReadyError("Training CNN for the first time.", state="training")
        raise ModelInitializationError("CNN model initialization failed.")

    def get_health(self) -> dict[str, Any]:
        ready = self.is_ready()
        if ready:
            http_status_label = "ok"
        elif self.state == "training":
            http_status_label = "training"
        elif self.state == "loading":
            http_status_label = "initializing"
        else:
            http_status_label = "error"

        return {
            "status": http_status_label,
            "ready": ready,
            "model_loaded": ready,
            "model_state": self.state,
            "message": self.status_message,
            "model_name": self.metrics.get("model_name", "DigitAI Enhanced CNN"),
            "model_format": self.model_format,
            "model_path": str(self.active_model_path or self.model_path),
            "test_accuracy": self.metrics.get("enhanced_cnn", {}).get("test_accuracy"),
        }

    def get_model_info(self) -> dict[str, Any]:
        return {
            "status": "loaded" if self.is_ready() else self.state,
            "ready": self.is_ready(),
            "model_state": self.state,
            "message": self.status_message,
            "model_format": self.model_format,
            "model_path": str(self.active_model_path or self.model_path),
            "metrics": self.metrics,
            "plots": {
                "training_curves": "/static/plots/training_curves.png",
                "confusion_matrix": "/static/plots/confusion_matrix.png",
            },
            "sequence_note": (
                "DigitAI uses an OpenCV contour + vertical-projection segmentation pipeline paired with a "
                "28x28 CNN classifier. While accurate on separated and lightly touching digits, segmentation "
                "can struggle on heavily overlapping cursive numerals or noisy backgrounds; an end-to-end "
                "CRNN (CNN + BiLSTM + CTC loss) or Attention-based sequence model removes explicit segmentation."
            ),
        }

    def predict_from_bytes(self, raw_bytes: bytes, mode: str = "multi") -> dict[str, Any]:
        self._require_ready()
        return self._predict_gray(decode_image_bytes(raw_bytes), mode=mode)

    def predict_from_base64(self, image_b64: str, mode: str = "multi") -> dict[str, Any]:
        self._require_ready()
        return self._predict_gray(decode_base64_image(image_b64), mode=mode)

    def _predict_gray(self, gray: np.ndarray, mode: str = "multi") -> dict[str, Any]:
        self._require_ready()
        assert self.model is not None

        t_start = time.perf_counter()
        prep: PreprocessingResult = preprocess_and_segment(gray, mode=mode)

        batch = np.stack([d.mnist_array for d in prep.digits], axis=0)
        probs_batch = self.model.predict(batch, verbose=0)

        digit_predictions: list[dict[str, Any]] = []
        predicted_chars: list[str] = []
        confidences: list[float] = []
        warnings = list(prep.warnings)
        low_confidence = False

        for d, probs in zip(prep.digits, probs_batch):
            probs_list = [round(float(p), 6) for p in probs]
            pred_digit = int(np.argmax(probs))
            conf = float(probs[pred_digit])
            if conf < LOW_CONFIDENCE_THRESHOLD:
                low_confidence = True
            predicted_chars.append(str(pred_digit))
            confidences.append(conf)

            top3_idx = np.argsort(probs)[::-1][:3]
            digit_predictions.append(
                {
                    "index": d.index,
                    "digit": pred_digit,
                    "confidence": round(conf, 6),
                    "probabilities": probs_list,
                    "top_alternatives": [
                        {"digit": int(idx), "probability": round(float(probs[idx]), 6)}
                        for idx in top3_idx
                    ],
                    "bbox": {
                        "x": int(d.bbox[0]),
                        "y": int(d.bbox[1]),
                        "w": int(d.bbox[2]),
                        "h": int(d.bbox[3]),
                    },
                    "was_split": d.was_split,
                    "processed_digit_base64": d.crop_base64,
                }
            )

        if low_confidence:
            warnings.append(
                "One or more digits had low prediction confidence (< 72%). Try writing darker, clearer strokes."
            )

        overall_conf = float(np.prod(confidences) ** (1.0 / len(confidences))) if confidences else 0.0
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        primary_probs = digit_predictions[0]["probabilities"] if digit_predictions else [0.0] * 10

        return {
            "predicted_number": "".join(predicted_chars),
            "mode": mode,
            "confidence": round(overall_conf, 6),
            "min_confidence": round(min(confidences) if confidences else 0.0, 6),
            "digit_count": len(digit_predictions),
            "digits": digit_predictions,
            "primary_probabilities": primary_probs,
            "low_confidence_warning": low_confidence,
            "segmentation_uncertain": prep.segmentation_uncertain,
            "warnings": warnings,
            "inverted_colors": prep.inverted_colors,
            "processed_preview_base64": prep.processed_preview_base64,
            "binary_preview_base64": prep.binary_preview_base64,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def evaluate_mnist_subset(self, num_samples: int = 1000) -> dict[str, Any]:
        self._require_ready()
        assert self.model is not None

        num_samples = max(100, min(10000, int(num_samples)))
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_sub = (x_test[:num_samples].astype("float32") / 255.0)[..., np.newaxis]
        y_sub = y_test[:num_samples]

        t0 = time.perf_counter()
        probs = self.model.predict(x_sub, batch_size=256, verbose=0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        preds = np.argmax(probs, axis=1)
        acc = float(np.mean(preds == y_sub))
        loss = float(np.mean(keras.losses.sparse_categorical_crossentropy(y_sub, probs).numpy()))
        p, r, f1, _ = precision_recall_fscore_support(y_sub, preds, average="macro", zero_division=0)
        cm = confusion_matrix(y_sub, preds, labels=list(range(10)))

        return {
            "samples_evaluated": int(num_samples),
            "accuracy": round(acc, 6),
            "loss": round(loss, 6),
            "precision_macro": round(float(p), 6),
            "recall_macro": round(float(r), 6),
            "f1_macro": round(float(f1), 6),
            "total_inference_ms": round(elapsed_ms, 2),
            "ms_per_digit": round(elapsed_ms / num_samples, 4),
            "confusion_matrix": cm.tolist(),
        }


def ensure_sample_gallery(mnist_data: tuple[np.ndarray, np.ndarray] | None = None) -> list[dict[str, Any]]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    samples_spec = [
        {"id": "sample_single_7", "label": "Single Digit: 7", "mode": "single", "text": "7", "bg": "white"},
        {"id": "sample_single_3", "label": "Single Digit: 3", "mode": "single", "text": "3", "bg": "dark"},
        {"id": "sample_double_48", "label": "Double Digit: 48", "mode": "double", "text": "48", "bg": "white"},
        {"id": "sample_double_96", "label": "Double Digit: 96", "mode": "double", "text": "96", "bg": "dark"},
        {"id": "sample_multi_2026", "label": "Multi Digit: 2026", "mode": "multi", "text": "2026", "bg": "white"},
        {"id": "sample_multi_987654", "label": "Multi Digit: 987654", "mode": "multi", "text": "987654", "bg": "white"},
    ]

    all_exist = all((SAMPLES_DIR / f"{spec['id']}.png").exists() for spec in samples_spec)
    mnist_imgs, mnist_lbls = mnist_data if mnist_data is not None else (None, None)
    if not all_exist and mnist_imgs is None:
        try:
            (_, _), (mnist_imgs, mnist_lbls) = keras.datasets.mnist.load_data()
        except Exception:
            mnist_imgs, mnist_lbls = None, None

    gallery: list[dict[str, Any]] = []
    for s_idx, spec in enumerate(samples_spec):
        out_file = SAMPLES_DIR / f"{spec['id']}.png"
        digits_str = spec["text"]

        if not out_file.exists():
            if mnist_imgs is not None and mnist_lbls is not None:
                digit_tiles = []
                for d_i, ch in enumerate(digits_str):
                    matches = np.where(mnist_lbls == int(ch))[0]
                    pick = matches[(s_idx * 17 + d_i * 31) % len(matches)]
                    tile = mnist_imgs[pick]
                    ys, xs = np.where(tile > 20)
                    if len(xs) and len(ys):
                        tile = tile[max(0, ys.min() - 2) : min(28, ys.max() + 3), max(0, xs.min() - 2) : min(28, xs.max() + 3)]
                    digit_tiles.append(
                        Image.fromarray(tile).resize(
                            (46, max(16, int(46 * tile.shape[0] / max(1, tile.shape[1])))),
                            Image.Resampling.BICUBIC,
                        )
                    )

                spacing = 18
                total_w = sum(t.width for t in digit_tiles) + spacing * (len(digit_tiles) - 1) + 48
                total_h = max(t.height for t in digit_tiles) + 48
                canvas = Image.new("L", (max(120, total_w), max(96, total_h)), color=0)
                cx = (canvas.width - (sum(t.width for t in digit_tiles) + spacing * (len(digit_tiles) - 1))) // 2
                for t in digit_tiles:
                    cy = (canvas.height - t.height) // 2
                    canvas.paste(t, (cx, cy))
                    cx += t.width + spacing

                if spec["bg"] == "white":
                    canvas = Image.fromarray(255 - np.array(canvas))
                canvas.save(out_file)
            else:
                bg_color = 248 if spec["bg"] == "white" else 16
                fg_color = 20 if spec["bg"] == "white" else 242
                canvas = Image.new("L", (max(140, 48 * len(digits_str) + 40), 110), color=bg_color)
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 42), digits_str, fill=fg_color, font=ImageFont.load_default())
                canvas.save(out_file)

        with open(out_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")

        gallery.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "expected": spec["text"],
                "mode": spec["mode"],
                "background": spec["bg"],
                "image_url": f"/static/samples/{spec['id']}.png",
                "image_base64": f"data:image/png;base64,{b64}",
            }
        )

    return gallery
