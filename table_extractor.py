from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import pytesseract


Point = Tuple[int, int]
CellBox = Tuple[int, int, int, int]


@dataclass
class ExtractionResult:
    dataframe: pd.DataFrame
    overlay_image: np.ndarray
    cell_boxes: List[CellBox]


def _cluster_positions(values: List[int], tolerance: int = 10) -> List[int]:
    if not values:
        return []
    values = sorted(values)
    clustered = [[values[0]]]
    for v in values[1:]:
        if abs(v - clustered[-1][-1]) <= tolerance:
            clustered[-1].append(v)
        else:
            clustered.append([v])
    return [int(sum(group) / len(group)) for group in clustered]


def _extract_grid(binary_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = binary_inv.shape
    h_kernel_len = max(25, w // 20)
    v_kernel_len = max(25, h // 20)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))

    horizontal_lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    grid = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    return horizontal_lines, vertical_lines, grid


def _get_grid_coordinates(horizontal_lines: np.ndarray, vertical_lines: np.ndarray) -> tuple[List[int], List[int]]:
    intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)
    ys, xs = np.where(intersections > 0)
    x_positions = _cluster_positions(xs.tolist(), tolerance=12)
    y_positions = _cluster_positions(ys.tolist(), tolerance=12)
    return x_positions, y_positions


def _build_cell_boxes(x_positions: List[int], y_positions: List[int], min_size: int = 10) -> List[CellBox]:
    cell_boxes: List[CellBox] = []
    if len(x_positions) < 2 or len(y_positions) < 2:
        return cell_boxes
    for r in range(len(y_positions) - 1):
        for c in range(len(x_positions) - 1):
            x1, x2 = x_positions[c], x_positions[c + 1]
            y1, y2 = y_positions[r], y_positions[r + 1]
            if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
                cell_boxes.append((x1, y1, x2, y2))
    return cell_boxes


def _ocr_cell(cell_img: np.ndarray, lang: str = "chi_sim+eng") -> str:
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(bw, lang=lang, config="--oem 3 --psm 6")
    return text.strip()


def _build_dataframe(image: np.ndarray, x_positions: List[int], y_positions: List[int], lang: str) -> pd.DataFrame:
    if len(x_positions) < 2 or len(y_positions) < 2:
        return pd.DataFrame()

    rows = []
    for r in range(len(y_positions) - 1):
        row_values = []
        for c in range(len(x_positions) - 1):
            x1, x2 = x_positions[c], x_positions[c + 1]
            y1, y2 = y_positions[r], y_positions[r + 1]
            cell_img = image[y1:y2, x1:x2]
            row_values.append(_ocr_cell(cell_img, lang=lang))
        rows.append(row_values)

    max_cols = max((len(r) for r in rows), default=0)
    cols = [f"C{idx + 1}" for idx in range(max_cols)]
    return pd.DataFrame(rows, columns=cols)


def _build_overlay_image(image: np.ndarray, x_positions: List[int], y_positions: List[int]) -> np.ndarray:
    overlay = image.copy()
    for x in x_positions:
        cv2.line(overlay, (x, 0), (x, overlay.shape[0]), (0, 255, 0), 1)
    for y in y_positions:
        cv2.line(overlay, (0, y), (overlay.shape[1], y), (0, 255, 0), 1)
    return overlay


def _fallback_whole_image_ocr(image: np.ndarray, lang: str) -> pd.DataFrame:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(bw, lang=lang, config="--oem 3 --psm 6").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame()
    return pd.DataFrame({"Text": lines})


def _merge_special_tokens(tokens: List[str]) -> List[str]:
    merged: List[str] = []
    for token in tokens:
        t = token.strip()
        if not t:
            continue
        if re.fullmatch(r"\(\s*%\s*\)", t) and merged:
            merged[-1] = f"{merged[-1]}(%)"
            continue
        if re.fullmatch(r"\[\d+\]", t) and merged:
            merged[-1] = f"{merged[-1]} {t}"
            continue
        merged.append(t)
    return merged


def _smart_split_text(text: str) -> List[str]:
    text = str(text or "")
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    text = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", text)
    text = re.sub(r"\b([A-Za-z]+)\s*\(\s*%\s*\)", r"\1(%)", text)
    text = re.sub(r"\bk\s*\(\s*%\s*\)", "k(%)", text, flags=re.IGNORECASE)

    model_matches = re.findall(r"[A-Za-z][A-Za-z0-9-]*\s*\[\d+\]", text)
    if len(model_matches) >= 2:
        first_start = text.find(model_matches[0])
        prefix = text[:first_start].strip()
        tokens = [m.strip() for m in model_matches]
        if prefix:
            return [prefix] + tokens
        return tokens

    strong_parts = [p.strip() for p in re.split(r"(?:\t+| {2,}|[|¦｜]+)", text) if p.strip()]
    if len(strong_parts) > 1:
        return _merge_special_tokens(strong_parts)

    parts_by_punct = [p.strip() for p in re.split(r"\s*[,，;；]\s*", text) if p.strip()]
    if len(parts_by_punct) > 1:
        return _merge_special_tokens(parts_by_punct)

    space_parts = [p for p in text.split(" ") if p]
    if len(space_parts) <= 1:
        return [text]

    numeric_pattern = re.compile(r"^[+-]?\d[\d,]*(?:\.\d+)?%?$")
    numeric_like = sum(1 for p in space_parts if numeric_pattern.match(p))
    if numeric_like >= 2:
        return _merge_special_tokens(space_parts)

    return [text]


def _is_numeric_token(token: str) -> bool:
    pattern = r"[+-]?\d[\d,]*(?:\.\d+)?(?:%|‰)?(?:±[+-]?\d[\d,]*(?:\.\d+)?)?"
    return re.fullmatch(pattern, token) is not None


def _infer_target_columns(token_rows: List[List[str]], fallback_cols: int) -> int:
    if not token_rows:
        return max(1, fallback_cols)

    candidate_counts: List[int] = []
    for tokens in token_rows:
        if not tokens:
            continue

        model_ref_count = sum(1 for t in tokens if re.search(r"\[\d+\]", t))
        if model_ref_count >= 3:
            if tokens and re.search(r"class", tokens[0], flags=re.IGNORECASE):
                candidate_counts.append(model_ref_count + 1)
            else:
                candidate_counts.append(max(model_ref_count, len(tokens)))
            continue

        numeric_like = sum(1 for t in tokens if _is_numeric_token(t))
        if numeric_like >= 2:
            if _is_numeric_token(tokens[0]):
                candidate_counts.append(max(numeric_like, len(tokens)))
            else:
                candidate_counts.append(max(numeric_like + 1, len(tokens)))
            continue

        candidate_counts.append(len(tokens))

    if not candidate_counts:
        return max(1, fallback_cols)

    inferred = int(pd.Series(candidate_counts).mode().iloc[0])
    return max(1, inferred, int(fallback_cols))


def _normalize_row(tokens: List[str], target_cols: int) -> List[str]:
    if not tokens:
        return [""] * target_cols
    if len(tokens) == target_cols:
        return tokens
    if len(tokens) < target_cols:
        return tokens + [""] * (target_cols - len(tokens))

    numeric_idx = [idx for idx, t in enumerate(tokens) if _is_numeric_token(t)]
    if len(numeric_idx) >= target_cols - 1 and target_cols >= 2:
        keep_numeric = numeric_idx[-(target_cols - 1) :]
        first_numeric = keep_numeric[0]
        label = " ".join(tokens[:first_numeric]).strip()
        tail = [tokens[idx] for idx in keep_numeric]
        return [label] + tail

    merged_tail = " ".join(tokens[target_cols - 1 :]).strip()
    return tokens[: target_cols - 1] + [merged_tail]


def _format_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    token_rows: List[List[str]] = []
    for _, row in df.fillna("").iterrows():
        tokens: List[str] = []
        for cell in row.tolist():
            tokens.extend(_smart_split_text(cell))
        token_rows.append(tokens)

    target_cols = _infer_target_columns(token_rows, fallback_cols=int(df.shape[1]))
    normalized_rows = [_normalize_row(tokens, target_cols) for tokens in token_rows]
    columns = [f"C{i + 1}" for i in range(target_cols)]
    return pd.DataFrame(normalized_rows, columns=columns)


def extract_table_to_dataframe(image: np.ndarray, lang: str = "chi_sim+eng") -> ExtractionResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )

    horizontal_lines, vertical_lines, _ = _extract_grid(binary_inv)
    x_positions, y_positions = _get_grid_coordinates(horizontal_lines, vertical_lines)
    cell_boxes = _build_cell_boxes(x_positions, y_positions)
    if len(x_positions) >= 2 and len(y_positions) >= 2:
        df = _build_dataframe(image, x_positions, y_positions, lang=lang)
    else:
        df = _fallback_whole_image_ocr(image, lang=lang)
    df = _format_dataframe_columns(df)
    overlay = _build_overlay_image(image, x_positions, y_positions)

    return ExtractionResult(dataframe=df, overlay_image=overlay, cell_boxes=cell_boxes)
