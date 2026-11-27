"""Flask backend API and server for DigitAI Handwritten Digit Recognition."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("digitai.app")

from backend.model_service import (
    ModelInitializationError,
    ModelNotReadyError,
    ModelService,
    ensure_sample_gallery,
)
from backend.preprocessing import ALLOWED_EXTENSIONS, ImageValidationError, NoDigitsDetectedError


def create_app(model_service: ModelService | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT_DIR / "templates"),
        static_folder=str(ROOT_DIR / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    CORS(app)

    service = model_service or ModelService(async_init=True, auto_train=True)
    if not service.is_ready() and service.state == "loading":
        service.start_background_init()

    samples_cache = ensure_sample_gallery()

    @app.route("/")
    def index():
        dist_index = ROOT_DIR / "dist" / "index.html"
        if dist_index.exists() and os.environ.get("DIGITAI_USE_BOOTSTRAP_TEMPLATE") != "1":
            return send_from_directory(ROOT_DIR / "dist", "index.html")
        return render_template("index.html")

    @app.route("/assets/<path:filename>")
    def dist_assets(filename: str):
        dist_assets_dir = ROOT_DIR / "dist" / "assets"
        if dist_assets_dir.exists():
            return send_from_directory(dist_assets_dir, filename)
        return jsonify({"error": "Asset not found"}), 404

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify(service.get_health())

    @app.route("/api/model-info", methods=["GET"])
    def model_info():
        return jsonify(service.get_model_info())

    @app.route("/api/samples", methods=["GET"])
    def samples():
        return jsonify({"samples": samples_cache})

    @app.route("/api/predict", methods=["POST"])
    def predict():
        mode = "multi"
        try:
            if not service.is_ready():
                service._require_ready()

            if request.content_type and "multipart/form-data" in request.content_type:
                mode = request.form.get("mode", "multi")
                if "image" not in request.files:
                    return jsonify({"error": "Please upload a valid handwritten digit image.", "code": "INVALID_IMAGE"}), 400
                file_obj = request.files["image"]
                filename = file_obj.filename or ""
                if filename and "." in filename:
                    ext = filename.rsplit(".", 1)[1].lower()
                    if ext not in ALLOWED_EXTENSIONS:
                        return (
                            jsonify(
                                {
                                    "error": "Please upload a valid handwritten digit image.",
                                    "code": "INVALID_IMAGE",
                                }
                            ),
                            400,
                        )
                return jsonify(service.predict_from_bytes(file_obj.read(), mode=mode))

            payload = request.get_json(silent=True) or {}
            mode = str(payload.get("mode", "multi"))
            image_b64 = payload.get("image")
            if not image_b64:
                return jsonify({"error": "Please upload a valid handwritten digit image.", "code": "INVALID_IMAGE"}), 400
            return jsonify(service.predict_from_base64(image_b64, mode=mode))

        except ModelNotReadyError as exc:
            logger.info("Prediction blocked because CNN model is not ready (state=%s): %s", exc.state, exc)
            return (
                jsonify(
                    {
                        "error": str(exc),
                        "code": "MODEL_TRAINING" if exc.state == "training" else "MODEL_INITIALIZING",
                        "model_state": exc.state,
                    }
                ),
                503,
            )
        except ModelInitializationError:
            logger.exception("Prediction failed due to CNN model initialization error.")
            return (
                jsonify(
                    {
                        "error": "Unable to process this image. Please try another image.",
                        "code": "MODEL_UNAVAILABLE",
                    }
                ),
                503,
            )
        except NoDigitsDetectedError:
            return jsonify({"error": "No handwritten digits detected.", "code": "NO_DIGITS_DETECTED"}), 422
        except ImageValidationError:
            return jsonify({"error": "Please upload a valid handwritten digit image.", "code": "INVALID_IMAGE"}), 400
        except Exception:
            logger.exception("Unexpected error during digit prediction (mode=%s)", mode)
            return (
                jsonify(
                    {
                        "error": "Unable to process this image. Please try another image.",
                        "code": "PREDICTION_ERROR",
                    }
                ),
                500,
            )

    @app.route("/api/evaluate", methods=["POST"])
    def evaluate():
        payload = request.get_json(silent=True) or {}
        try:
            num_samples = int(payload.get("num_samples", 1000))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid evaluation sample count."}), 400
        try:
            return jsonify(service.evaluate_mnist_subset(num_samples=num_samples))
        except ModelNotReadyError as exc:
            return (
                jsonify(
                    {
                        "error": str(exc),
                        "code": "MODEL_TRAINING" if exc.state == "training" else "MODEL_INITIALIZING",
                        "model_state": exc.state,
                    }
                ),
                503,
            )
        except Exception:
            logger.exception("Unexpected error during MNIST evaluation.")
            return jsonify({"error": "Unable to evaluate the CNN model right now."}), 500

    @app.errorhandler(413)
    def file_too_large(_err):
        return jsonify({"error": "Please upload a valid handwritten digit image."}), 413

    return app


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "5001"))
    create_app().run(host="0.0.0.0", port=port, debug=False)
