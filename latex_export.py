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


def _is_numeric_like(value: str) -> bool:
    s = str(value or "").strip().replace(",", "")
    if not s:
        return False
    pattern = r"[+-]?\d+(?:\.\d+)?(?:%|‰)?(?:±[+-]?\d+(?:\.\d+)?)?"
    return re.fullmatch(pattern, s) is not None


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


def dataframe_to_latex_table(
    df: pd.DataFrame,
    caption: str = "OCR extracted table.",
    label: str = "tab:ocr_result",
    use_xhline: bool = True,
) -> str:
    if df.empty:
        return ""

    line_cmd = r"\Xhline{1pt}" if use_xhline else r"\hline"
    align = _infer_alignment(df)

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{_escape_latex(caption)}}}")
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(line_cmd)

    headers = " & ".join(_escape_latex(str(c)) for c in df.columns.tolist())
    lines.append(headers + r" \\")
    lines.append(line_cmd)

    for _, row in df.iterrows():
        cells = " & ".join(_escape_latex(str(v)) for v in row.tolist())
        lines.append(cells + r" \\")

    lines.append(line_cmd)
    lines.append(r"\end{tabular}")
    safe_label = re.sub(r"[^A-Za-z0-9:_\-]", "", str(label or "tab:ocr_result"))
    lines.append(rf"\label{{{safe_label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
