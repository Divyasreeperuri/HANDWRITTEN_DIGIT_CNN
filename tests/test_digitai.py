"""Automated Pytest suite for DigitAI CNN model, preprocessing, segmentation, and Flask API."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from tensorflow import keras

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import create_app
from backend.model_service import ModelService
from backend.preprocessing import (
    ImageValidationError,
    NoDigitsDetectedError,
    decode_image_bytes,
    preprocess_and_segment,
)


@pytest.fixture(scope="session")
def mnist_test_data():
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    return x_test, y_test


@pytest.fixture(scope="session")
def model_service():
    return ModelService()


@pytest.fixture(scope="session")
def client(model_service):
    app = create_app(model_service=model_service)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _compose_digits_png(mnist_data, digits: list[int], spacing: int = 18, invert_paper: bool = False) -> bytes:
    x_test, y_test = mnist_data
    tiles = []
    for offset, d in enumerate(digits):
        idx = int(np.where(y_test == d)[0][(offset * 23 + 3) % 400])
        tile = x_test[idx]
        ys, xs = np.where(tile > 20)
        tile = tile[max(0, ys.min() - 1) : min(28, ys.max() + 2), max(0, xs.min() - 1) : min(28, xs.max() + 2)]
        tiles.append(Image.fromarray(tile).resize((40, 52), Image.Resampling.BICUBIC))

    w = max(80, sum(t.width for t in tiles) + spacing * (len(tiles) - 1) + 40)
    h = 92
    canvas = Image.new("L", (w, h), color=0)
    x = 20
    for t in tiles:
        canvas.paste(t, (x, 20))
        x += t.width + spacing

    if invert_paper:
        canvas = Image.fromarray(255 - np.array(canvas))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def test_model_loads_and_reports_metrics(model_service):
    assert model_service.model is not None
    assert model_service.is_ready() is True
    info = model_service.get_model_info()
    assert info["status"] == "loaded"
    assert info["ready"] is True
    assert info["metrics"]["enhanced_cnn"]["test_accuracy"] >= 0.98
    assert Path(model_service.weights_h5_path).exists()


def test_all_ten_digit_classes(model_service, mnist_test_data):
    correct = 0
    for digit in range(10):
        img_bytes = _compose_digits_png(mnist_test_data, [digit], invert_paper=(digit % 2 == 0))
        res = model_service.predict_from_bytes(img_bytes, mode="single")
        if res["predicted_number"] == str(digit):
            correct += 1
    assert correct >= 9, f"Expected at least 9/10 digit classes recognized, got {correct}/10"


def test_single_double_and_multi_digit_recognition(model_service, mnist_test_data):
    single_bytes = _compose_digits_png(mnist_test_data, [7], invert_paper=True)
    res_single = model_service.predict_from_bytes(single_bytes, mode="single")
    assert res_single["predicted_number"] == "7"
    assert res_single["digit_count"] == 1

    double_bytes = _compose_digits_png(mnist_test_data, [4, 8], invert_paper=True)
    res_double = model_service.predict_from_bytes(double_bytes, mode="double")
    assert res_double["predicted_number"] == "48"
    assert res_double["digit_count"] == 2

    multi_bytes = _compose_digits_png(mnist_test_data, [2, 0, 2, 6], invert_paper=False)
    res_multi = model_service.predict_from_bytes(multi_bytes, mode="multi")
    assert res_multi["predicted_number"] == "2026"
    assert res_multi["digit_count"] == 4


def test_touching_digits_segmentation(mnist_test_data):
    touching_bytes = _compose_digits_png(mnist_test_data, [2, 5], spacing=-4, invert_paper=True)
    gray = decode_image_bytes(touching_bytes)
    prep = preprocess_and_segment(gray, mode="double")
    assert len(prep.digits) == 2


def test_black_on_white_and_white_on_black_inversion(mnist_test_data):
    paper_bytes = _compose_digits_png(mnist_test_data, [5], invert_paper=True)
    dark_bytes = _compose_digits_png(mnist_test_data, [5], invert_paper=False)

    prep_paper = preprocess_and_segment(decode_image_bytes(paper_bytes), mode="single")
    prep_dark = preprocess_and_segment(decode_image_bytes(dark_bytes), mode="single")

    assert prep_paper.inverted_colors is True
    assert prep_dark.inverted_colors is False
    assert prep_paper.digits[0].mnist_array.shape == (28, 28, 1)


def test_canvas_rgba_transparency_handling(model_service):
    img = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.line([(50, 40), (130, 40), (80, 145)], fill=(15, 23, 42, 255), width=14)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    res = model_service.predict_from_base64(b64, mode="single")
    assert res["digit_count"] == 1
    assert len(res["predicted_number"]) == 1


def test_empty_corrupted_and_invalid_images():
    with pytest.raises(ImageValidationError):
        decode_image_bytes(b"")

    with pytest.raises(ImageValidationError):
        decode_image_bytes(b"not-a-valid-image-payload-12345")

    blank = Image.new("L", (128, 128), color=255)
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    with pytest.raises(NoDigitsDetectedError):
        preprocess_and_segment(decode_image_bytes(buf.getvalue()), mode="single")


def test_api_endpoints_and_error_handling(client, mnist_test_data):
    health = client.get("/api/health")
    assert health.status_code == 200
    health_json = health.get_json()
    assert health_json["model_loaded"] is True
    assert health_json["ready"] is True
    assert health_json["model_state"] == "ready"

    info = client.get("/api/model-info")
    assert info.status_code == 200
    assert "metrics" in info.get_json()

    samples = client.get("/api/samples")
    assert samples.status_code == 200
    assert len(samples.get_json()["samples"]) >= 4

    png_bytes = _compose_digits_png(mnist_test_data, [9, 6], invert_paper=True)
    resp = client.post(
        "/api/predict",
        data={"mode": "double", "image": (io.BytesIO(png_bytes), "test_96.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["predicted_number"] == "96"
    assert "processing_time_ms" in body

    jpg_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    jpg_buf = io.BytesIO()
    jpg_img.save(jpg_buf, format="JPEG", quality=92)
    resp_jpg = client.post(
        "/api/predict",
        data={"mode": "double", "image": (io.BytesIO(jpg_buf.getvalue()), "test_96.jpg")},
        content_type="multipart/form-data",
    )
    assert resp_jpg.status_code == 200
    assert resp_jpg.get_json()["predicted_number"] == "96"

    bad_ext = client.post(
        "/api/predict",
        data={"mode": "single", "image": (io.BytesIO(png_bytes), "malware.exe")},
        content_type="multipart/form-data",
    )
    assert bad_ext.status_code == 400
    assert bad_ext.get_json()["error"] == "Please upload a valid handwritten digit image."

    bad_json = client.post("/api/predict", json={"mode": "single"})
    assert bad_json.status_code == 400
    assert bad_json.get_json()["error"] == "Please upload a valid handwritten digit image."

    blank = Image.new("L", (128, 128), color=255)
    blank_buf = io.BytesIO()
    blank.save(blank_buf, format="PNG")
    blank_resp = client.post(
        "/api/predict",
        data={"mode": "single", "image": (io.BytesIO(blank_buf.getvalue()), "blank.png")},
        content_type="multipart/form-data",
    )
    assert blank_resp.status_code == 422
    assert blank_resp.get_json()["error"] == "No handwritten digits detected."

    eval_resp = client.post("/api/evaluate", json={"num_samples": 1000})
    assert eval_resp.status_code == 200
    eval_json = eval_resp.get_json()
    assert eval_json["samples_evaluated"] == 1000
    assert eval_json["accuracy"] >= 0.97


def test_not_ready_service_blocks_predictions():
    unready = ModelService(auto_initialize=False, auto_train=False)
    app = create_app(model_service=unready)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        h = test_client.get("/api/health").get_json()
        assert h["ready"] is False
        assert h["model_state"] == "loading"
        resp = test_client.post("/api/predict", json={"image": "data:image/png;base64,AAAA", "mode": "single"})
        assert resp.status_code == 503
        assert resp.get_json()["error"] == "Initializing CNN model."

