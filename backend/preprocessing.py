"""Image preprocessing and multi-digit segmentation pipeline for DigitAI."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_IMAGE_DIM = 4096
MIN_IMAGE_DIM = 8


class ImageValidationError(ValueError):
    """Raised when an uploaded image is empty, corrupted, or unsupported."""


class NoDigitsDetectedError(ValueError):
    """Raised when an image contains no recognizable handwritten strokes."""


@dataclass
class SegmentedDigit:
    index: int
    bbox: tuple[int, int, int, int]
    mnist_array: np.ndarray
    crop_base64: str
    was_split: bool = False


@dataclass
class PreprocessingResult:
    digits: list[SegmentedDigit]
    processed_preview_base64: str
    binary_preview_base64: str
    inverted_colors: bool
    segmentation_uncertain: bool
    warnings: list[str] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "digit_count": len(self.digits),
            "inverted_colors": self.inverted_colors,
            "segmentation_uncertain": self.segmentation_uncertain,
            "warnings": self.warnings,
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
            "bounding_boxes": [
                {
                    "index": d.index,
                    "x": int(d.bbox[0]),
                    "y": int(d.bbox[1]),
                    "w": int(d.bbox[2]),
                    "h": int(d.bbox[3]),
                    "was_split": d.was_split,
                }
                for d in self.digits
            ],
        }


def _array_to_base64_png(arr: np.ndarray) -> str:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    else:
        img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes safely, handling RGBA transparency."""
    if not raw_bytes or len(raw_bytes) < 16:
        raise ImageValidationError("Please upload a valid handwritten digit image.")

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(raw_bytes))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Please upload a valid handwritten digit image.") from exc

    w, h = pil_img.size
    if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
        raise ImageValidationError("Please upload a valid handwritten digit image.")
    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
        scale = min(MAX_IMAGE_DIM / w, MAX_IMAGE_DIM / h)
        pil_img = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        rgba = pil_img.convert("RGBA")
        arr = np.array(rgba)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32)
        opaque_mask = arr[:, :, 3] > 20
        if np.any(opaque_mask):
            mean_stroke_brightness = float(rgb[opaque_mask].mean())
        else:
            raise NoDigitsDetectedError("No handwritten digits detected.")
        bg_val = 0.0 if mean_stroke_brightness > 127.0 else 255.0
        composited = rgb * alpha + bg_val * (1.0 - alpha)
        rgb_uint8 = np.clip(composited, 0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

    return np.array(pil_img.convert("L"), dtype=np.uint8)


def decode_base64_image(data_url_or_b64: str) -> np.ndarray:
    """Decode a base64 string or data URL into a grayscale numpy image."""
    if not data_url_or_b64 or not isinstance(data_url_or_b64, str):
        raise ImageValidationError("Please upload a valid handwritten digit image.")

    payload = data_url_or_b64.strip()
    if "," in payload and payload.startswith("data:"):
        header, payload = payload.split(",", 1)
        if "image/" not in header.lower():
            raise ImageValidationError("Please upload a valid handwritten digit image.")

    try:
        raw_bytes = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ImageValidationError("Please upload a valid handwritten digit image.") from exc

    return decode_image_bytes(raw_bytes)


def _detect_inversion_needed(gray: np.ndarray) -> bool:
    h, w = gray.shape
    border = max(2, min(h, w) // 20)
    top = gray[:border, :].ravel()
    bottom = gray[-border:, :].ravel()
    left = gray[:, :border].ravel()
    right = gray[:, -border:].ravel()
    border_pixels = np.concatenate([top, bottom, left, right])
    border_median = float(np.median(border_pixels))
    overall_median = float(np.median(gray))
    return (0.65 * border_median + 0.35 * overall_median) > 127.0


def _binarize_and_denoise(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    contrast_span = float(np.percentile(gray, 99) - np.percentile(gray, 1))
    if contrast_span < 18.0:
        raise NoDigitsDetectedError("No handwritten digits detected.")

    inverted = _detect_inversion_needed(gray)
    work = 255 - gray if inverted else gray.copy()

    h, w = work.shape
    if max(h, w) > 80:
        blur_k = min(5, max(3, (min(h, w) // 180) * 2 + 1))
        work_blur = cv2.GaussianBlur(work, (blur_k, blur_k), 0)
    else:
        work_blur = work

    if min(h, w) >= 64:
        bg_kernel = max(21, (min(h, w) // 6) | 1)
        bg_est = cv2.morphologyEx(work_blur, cv2.MORPH_OPEN, np.ones((bg_kernel, bg_kernel), np.uint8))
        normalized = cv2.subtract(work_blur, bg_est)
        if np.max(normalized) > 25:
            work_blur = normalized

    _, binary = cv2.threshold(work_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    if fg_ratio > 0.65:
        binary = 255 - binary
        inverted = not inverted

    if max(h, w) >= 120:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    if np.count_nonzero(binary) < 8:
        raise NoDigitsDetectedError("No handwritten digits detected.")

    return binary, inverted


def _merge_broken_boxes(boxes: list[tuple[int, int, int, int]], img_h: int) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes

    boxes = sorted(boxes, key=lambda b: b[0])
    merged = True
    while merged:
        merged = False
        out: list[tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                min_w = max(1, min(w1, w2))
                vertical_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
                if (overlap_x / min_w >= 0.45) and (vertical_gap <= max(10, int(0.2 * img_h))):
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1 + w1, x2 + w2) - nx
                    nh = max(y1 + h1, y2 + h2) - ny
                    x1, y1, w1, h1 = nx, ny, nw, nh
                    used[j] = True
                    merged = True
            used[i] = True
            out.append((x1, y1, w1, h1))
        boxes = out
    return sorted(boxes, key=lambda b: b[0])


def _split_touching_box(
    binary: np.ndarray,
    bbox: tuple[int, int, int, int],
    force_double: bool = False,
    depth: int = 0,
) -> list[tuple[tuple[int, int, int, int], bool]]:
    x, y, w, h = bbox
    aspect = w / max(1.0, float(h))
    should_try_split = (aspect > 1.25) or (force_double and aspect > 0.98)
    if not should_try_split or w < 18 or depth >= 3:
        return [(bbox, depth > 0)]

    roi = binary[y : y + h, x : x + w]
    col_sum = np.sum(roi > 0, axis=0).astype(np.float32)
    col_smooth = np.convolve(col_sum, np.ones(3, dtype=np.float32) / 3.0, mode="same") if len(col_sum) >= 5 else col_sum

    margin = max(3, int(0.22 * w))
    if w - 2 * margin < 3:
        return [(bbox, depth > 0)]

    search_slice = col_smooth[margin : w - margin]
    split_x = margin + int(np.argmin(search_slice))
    valley_val = float(col_smooth[split_x])
    left_peak = float(np.max(col_smooth[:split_x])) if split_x > 0 else 1.0
    right_peak = float(np.max(col_smooth[split_x + 1 :])) if split_x + 1 < w else 1.0
    ref_peak = max(1.0, min(left_peak, right_peak))

    if force_double or valley_val <= 0.48 * ref_peak:
        parts: list[tuple[tuple[int, int, int, int], bool]] = []
        for sub_roi, offset_x in ((roi[:, :split_x], x), (roi[:, split_x:], x + split_x)):
            ys, xs = np.where(sub_roi > 0)
            if len(xs) < 6:
                continue
            sx = offset_x + int(np.min(xs))
            sy = y + int(np.min(ys))
            sw = int(np.max(xs) - np.min(xs) + 1)
            sh = int(np.max(ys) - np.min(ys) + 1)
            if sw >= 3 and sh >= 6:
                parts.extend(_split_touching_box(binary, (sx, sy, sw, sh), force_double=False, depth=depth + 1))
        if len(parts) >= 2:
            return parts

    return [(bbox, depth > 0)]


def normalize_digit_to_mnist(roi_binary_or_gray: np.ndarray) -> np.ndarray:
    roi = roi_binary_or_gray.astype(np.float32)
    if roi.max() > 1.0:
        roi = roi / 255.0

    ys, xs = np.where(roi > 0.1)
    if len(xs) > 0 and len(ys) > 0:
        roi = roi[np.min(ys) : np.max(ys) + 1, np.min(xs) : np.max(xs) + 1]

    h, w = roi.shape
    if h == 0 or w == 0:
        return np.zeros((28, 28, 1), dtype=np.float32)

    scale = 20.0 / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = np.clip(cv2.resize(roi, (new_w, new_h), interpolation=interp), 0.0, 1.0)

    canvas = np.zeros((28, 28), dtype=np.float32)
    pad_x = (28 - new_w) // 2
    pad_y = (28 - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    total_mass = float(np.sum(canvas))
    if total_mass > 1e-4:
        yy, xx = np.mgrid[0:28, 0:28]
        cy = float(np.sum(yy * canvas)) / total_mass
        cx = float(np.sum(xx * canvas)) / total_mass
        shift_x = max(-4, min(4, int(round(13.5 - cx))))
        shift_y = max(-4, min(4, int(round(13.5 - cy))))
        if shift_x != 0 or shift_y != 0:
            mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            canvas = cv2.warpAffine(canvas, mat, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0.0)

    peak = float(np.max(canvas))
    if peak > 0.05:
        canvas = np.clip(canvas / peak, 0.0, 1.0)

    return canvas.reshape((28, 28, 1)).astype(np.float32)


def preprocess_and_segment(gray: np.ndarray, mode: str = "multi") -> PreprocessingResult:
    mode = (mode or "multi").lower().strip()
    if mode not in {"single", "double", "multi"}:
        mode = "multi"

    img_h, img_w = gray.shape
    binary, inverted = _binarize_and_denoise(gray)

    if mode == "single" and img_h <= 32 and img_w <= 32:
        work = 255 - gray if inverted else gray.copy()
        mnist_arr = normalize_digit_to_mnist(work)
        ys, xs = np.where(binary > 0)
        bbox = (
            int(np.min(xs)) if len(xs) else 0,
            int(np.min(ys)) if len(ys) else 0,
            int(np.max(xs) - np.min(xs) + 1) if len(xs) else img_w,
            int(np.max(ys) - np.min(ys) + 1) if len(ys) else img_h,
        )
        preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(preview, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (34, 197, 94), 1)
        digit = SegmentedDigit(
            index=0,
            bbox=bbox,
            mnist_array=mnist_arr,
            crop_base64=_array_to_base64_png((mnist_arr[:, :, 0] * 255).astype(np.uint8)),
            was_split=False,
        )
        return PreprocessingResult(
            digits=[digit],
            processed_preview_base64=_array_to_base64_png(preview),
            binary_preview_base64=_array_to_base64_png(binary),
            inverted_colors=inverted,
            segmentation_uncertain=False,
            warnings=[],
            image_size=(img_w, img_h),
        )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(12.0, 0.00035 * img_w * img_h)
    min_h = max(6, int(0.04 * img_h))

    raw_boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) >= min_area and (h >= min_h or w >= min_h):
            raw_boxes.append((x, y, w, h))

    if not raw_boxes:
        raise NoDigitsDetectedError("No handwritten digits detected.")

    merged_boxes = _merge_broken_boxes(raw_boxes, img_h)
    warnings: list[str] = []
    segmentation_uncertain = False

    final_boxes_with_flags: list[tuple[tuple[int, int, int, int], bool]] = []
    if mode == "single":
        if len(merged_boxes) > 1:
            segmentation_uncertain = True
            warnings.append(
                f"Detected {len(merged_boxes)} separate regions in Single-Digit mode; combined primary region. Switch to Multi-Digit mode for multi-digit numbers."
            )
            xs_min = min(b[0] for b in merged_boxes)
            ys_min = min(b[1] for b in merged_boxes)
            xs_max = max(b[0] + b[2] for b in merged_boxes)
            ys_max = max(b[1] + b[3] for b in merged_boxes)
            final_boxes_with_flags.append(((xs_min, ys_min, xs_max - xs_min, ys_max - ys_min), False))
        else:
            final_boxes_with_flags.append((merged_boxes[0], False))
    else:
        force_double = mode == "double" and len(merged_boxes) == 1
        for box in merged_boxes:
            final_boxes_with_flags.extend(_split_touching_box(binary, box, force_double=force_double))
        if any(flag for _, flag in final_boxes_with_flags):
            segmentation_uncertain = True
            warnings.append("Touching digits were detected and separated using vertical projection splitting.")

    final_boxes_with_flags.sort(key=lambda item: item[0][0])

    if mode == "double" and len(final_boxes_with_flags) != 2:
        segmentation_uncertain = True
        warnings.append(f"Double-Digit mode expected 2 digits, but segmentation found {len(final_boxes_with_flags)}.")

    soft_work = 255 - gray if inverted else gray.copy()
    segmented_digits: list[SegmentedDigit] = []
    preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for idx, (bbox, was_split) in enumerate(final_boxes_with_flags):
        x, y, w, h = bbox
        roi_bin = binary[y : y + h, x : x + w]
        roi_soft = soft_work[y : y + h, x : x + w]
        mask = cv2.dilate(roi_bin, np.ones((2, 2), np.uint8), iterations=1)
        clean_roi = cv2.bitwise_and(roi_soft, mask)
        if np.count_nonzero(clean_roi) < 4:
            clean_roi = roi_bin

        mnist_arr = normalize_digit_to_mnist(clean_roi)
        crop_b64 = _array_to_base64_png((mnist_arr[:, :, 0] * 255).astype(np.uint8))

        segmented_digits.append(
            SegmentedDigit(index=idx, bbox=bbox, mnist_array=mnist_arr, crop_base64=crop_b64, was_split=was_split)
        )

        color = (245, 158, 11) if was_split else (6, 182, 212)
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
        cv2.putText(preview, f"#{idx + 1}", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return PreprocessingResult(
        digits=segmented_digits,
        processed_preview_base64=_array_to_base64_png(preview),
        binary_preview_base64=_array_to_base64_png(binary),
        inverted_colors=inverted,
        segmentation_uncertain=segmentation_uncertain,
        warnings=warnings,
        image_size=(img_w, img_h),
    )
