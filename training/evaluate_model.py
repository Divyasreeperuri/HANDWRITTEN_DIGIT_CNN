"""Standalone MNIST and synthetic multi-digit evaluation script for DigitAI."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image
from tensorflow import keras

from backend.model_service import ModelService


def generate_synthetic_multi_digit_image(digits: list[int], invert_to_paper: bool = False) -> bytes:
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    tiles = []
    for offset, d in enumerate(digits):
        idx = int(np.where(y_test == d)[0][(offset * 29 + 7) % 500])
        tile = x_test[idx]
        ys, xs = np.where(tile > 20)
        tile = tile[max(0, ys.min() - 1) : min(28, ys.max() + 2), max(0, xs.min() - 1) : min(28, xs.max() + 2)]
        tiles.append(Image.fromarray(tile).resize((44, 56), Image.Resampling.BICUBIC))

    spacing = 16
    w = sum(t.width for t in tiles) + spacing * (len(tiles) - 1) + 48
    h = 96
    canvas = Image.new("L", (w, h), color=0)
    x = 24
    for t in tiles:
        canvas.paste(t, (x, 20))
        x += t.width + spacing

    if invert_to_paper:
        canvas = Image.fromarray(255 - np.array(canvas))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    service = ModelService()
    print("[DigitAI] Running evaluation on 2,000 MNIST test images...")
    mnist_report = service.evaluate_mnist_subset(num_samples=2000)

    custom_cases = [
        ("7", "single", False),
        ("48", "double", True),
        ("96", "double", False),
        ("2026", "multi", True),
        ("987654", "multi", False),
    ]
    custom_results = []
    for expected, mode, paper in custom_cases:
        img_bytes = generate_synthetic_multi_digit_image([int(c) for c in expected], invert_to_paper=paper)
        res = service.predict_from_bytes(img_bytes, mode=mode)
        custom_results.append(
            {
                "expected": expected,
                "predicted": res["predicted_number"],
                "match": res["predicted_number"] == expected,
                "confidence": res["confidence"],
                "processing_time_ms": res["processing_time_ms"],
            }
        )

    full_report = {
        "mnist_evaluation": mnist_report,
        "custom_multi_digit_evaluation": custom_results,
    }
    report_path = ROOT_DIR / "models" / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(json.dumps(full_report, indent=2))


if __name__ == "__main__":
    main()
