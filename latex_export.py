from __future__ import annotations

import re
from typing import List

import pandas as pd


def _escape_latex(text: str) -> str:
    s = str(text or "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def _extract_numeric_value(value: str) -> float | None:
    s = str(value or "").strip()
    if not s:
        return None
    m = re.search(r"[+-]?\d[\d,]*(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _is_numeric_like(value: str) -> bool:
    return _extract_numeric_value(value) is not None


def _infer_alignment(df: pd.DataFrame) -> str:
    aligns: List[str] = []
    for col_idx in range(df.shape[1]):
        col = df.iloc[:, col_idx].astype(str).fillna("")
        non_empty = [v for v in col.tolist() if v.strip()]
        if not non_empty:
            aligns.append("c")
            continue
        numeric_count = sum(1 for v in non_empty if _is_numeric_like(v))
        ratio = numeric_count / len(non_empty)
        if col_idx == 0:
            aligns.append("c")
        elif ratio < 0.35:
            aligns.append("l")
        else:
            aligns.append("c")
    return "|".join(aligns)


def _format_cell(value: str, style: str = "") -> str:
    escaped = _escape_latex(str(value))
    if style == "max":
        return rf"\textbf{{{escaped}}}"
    if style == "second":
        return rf"\underline{{{escaped}}}"
    return escaped


def _row_styles_for_top2(row_values: List[str]) -> List[str]:
    styles = [""] * len(row_values)
    if len(row_values) <= 1:
        return styles

    parsed = []
    for idx in range(1, len(row_values)):
        v = _extract_numeric_value(row_values[idx])
        if v is not None:
            parsed.append((idx, v))

    if not parsed:
        return styles

    unique_vals = sorted({v for _, v in parsed}, reverse=True)
    max_val = unique_vals[0]
    second_val = unique_vals[1] if len(unique_vals) > 1 else None

    for idx, v in parsed:
        if v == max_val:
            styles[idx] = "max"
        elif second_val is not None and v == second_val:
            styles[idx] = "second"
    return styles


def dataframe_to_latex_table(
    df: pd.DataFrame,
    caption: str = "OCR extracted table.",
    label: str = "tab:ocr_result",
    use_xhline: bool = True,
    mark_top2_per_row: bool = False,
) -> str:
    if df.empty:
        return ""

    line_cmd = r"\Xhline{1pt}" if use_xhline else r"\hline"
    align = _infer_alignment(df)

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{_escape_latex(caption)}}}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(line_cmd)

    headers = " & ".join(_escape_latex(str(c)) for c in df.columns.tolist())
    lines.append(headers + r" \\")
    lines.append(line_cmd)

    for _, row in df.iterrows():
        row_values = [str(v) for v in row.tolist()]
        styles = _row_styles_for_top2(row_values) if mark_top2_per_row else [""] * len(row_values)
        cells = " & ".join(_format_cell(v, style=styles[i]) for i, v in enumerate(row_values))
        lines.append(cells + r" \\")

    lines.append(line_cmd)
    lines.append(r"\end{tabular}")
    safe_label = re.sub(r"[^A-Za-z0-9:_\-]", "", str(label or "tab:ocr_result"))
    lines.append(r"}")
    lines.append(rf"\label{{{safe_label}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)
