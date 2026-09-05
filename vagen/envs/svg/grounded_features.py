"""Lightweight, model-independent features for SVG rubric verification.

These features deliberately do not reuse DINO, DreamSim, or the environment
reward.  They provide an independent satisfaction signal ``v`` that can be
correlated with the environment feedback ``z`` by RLCER.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional
from xml.etree import ElementTree

import numpy as np
from PIL import Image


def _rgb_array(image: Optional[Image.Image], size: int = 64) -> Optional[np.ndarray]:
    if image is None:
        return None
    return np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    # SVG rasterization uses a white background. Keep anti-aliased colored
    # pixels while ignoring small compression/numerical deviations from white.
    return np.max(np.abs(rgb - 1.0), axis=-1) > (12.0 / 255.0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float(np.array_equal(a, b))
    return float(np.clip(np.dot(a.ravel(), b.ravel()) / denom, 0.0, 1.0))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _color_histogram(rgb: np.ndarray, mask: np.ndarray, bins: int = 8) -> np.ndarray:
    pixels = rgb[mask]
    if pixels.size == 0:
        return np.zeros(3 * bins, dtype=np.float32)
    parts = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[:, channel], bins=bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        parts.append(hist)
    return np.concatenate(parts)


def _edge_map(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=-1)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return (gx + gy) > 0.08


def _element_counts(svg_code: Optional[str]) -> Optional[Counter]:
    if not svg_code:
        return None
    try:
        root = ElementTree.fromstring(svg_code)
    except (ElementTree.ParseError, TypeError, ValueError):
        return None
    ignored = {"svg", "defs", "title", "desc", "metadata"}
    tags = [node.tag.split("}")[-1].lower() for node in root.iter()]
    return Counter(tag for tag in tags if tag not in ignored)


def _counter_similarity(a: Optional[Counter], b: Optional[Counter]) -> float:
    if a is None or b is None:
        return 0.0
    keys = set(a) | set(b)
    total = sum(a.values()) + sum(b.values())
    if total == 0:
        return 1.0
    distance = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)
    return float(np.clip(1.0 - distance / total, 0.0, 1.0))


def build_svg_verifier_context(
    target_image: Optional[Image.Image],
    generated_image: Optional[Image.Image],
    target_svg: Optional[str],
    generated_svg: Optional[str],
) -> Dict[str, Any]:
    """Return compact independent similarities used by ``SVGGroundedVerifier``."""
    target = _rgb_array(target_image)
    generated = _rgb_array(generated_image)
    generated_counts = _element_counts(generated_svg)
    valid = generated is not None and generated_counts is not None

    context: Dict[str, Any] = {
        "task_type": "svg",
        "valid_svg": bool(valid),
        "layout_similarity": 0.0,
        "color_similarity": 0.0,
        "edge_similarity": 0.0,
        "element_similarity": 0.0,
    }
    if target is None or generated is None:
        return context

    target_mask = _foreground_mask(target)
    generated_mask = _foreground_mask(generated)
    context["layout_similarity"] = _mask_iou(target_mask, generated_mask)
    context["color_similarity"] = _cosine_similarity(
        _color_histogram(target, target_mask),
        _color_histogram(generated, generated_mask),
    )
    context["edge_similarity"] = _cosine_similarity(
        _edge_map(target).astype(np.float32),
        _edge_map(generated).astype(np.float32),
    )
    context["element_similarity"] = _counter_similarity(
        _element_counts(target_svg),
        generated_counts,
    )
    return context
