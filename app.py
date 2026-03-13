from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import pytesseract
from shutil import which

from table_extractor import extract_table_to_dataframe


st.set_page_config(page_title="图片表格转 Excel", page_icon="📊", layout="wide")

st.title("图片表格识别 -> Excel")
st.caption("上传图片，自动识别表格网格与文字，预览后下载 Excel。")

with st.sidebar:
    st.header("识别设置")
    tesseract_lang = st.text_input("Tesseract 语言包", value="chi_sim+eng")
    tesseract_path = st.text_input(
        "Tesseract 可执行文件路径（可选）",
        value="",
        placeholder=r"D:\Prog\Tesseract\tesseract.exe",
    )
    st.info("示例：中文+英文用 chi_sim+eng。需本机已安装对应语言包。")

if tesseract_path.strip():
    pytesseract.pytesseract.tesseract_cmd = tesseract_path.strip()

cmd = pytesseract.pytesseract.tesseract_cmd
if "\\" in cmd or "/" in cmd:
    resolved_tesseract = str(Path(cmd)) if Path(cmd).exists() else None
else:
    resolved_tesseract = which(cmd)
if not resolved_tesseract:
    st.error(
        "未找到 Tesseract。请安装并加入 PATH，或在左侧填写 tesseract.exe 完整路径。"
    )
    st.stop()

uploaded = st.file_uploader("上传表格图片", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded is None:
    st.stop()

file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image_bgr is None:
    st.error("图片读取失败，请更换文件后重试。")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("原图")
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

with st.spinner("正在识别表格..."):
    try:
        result = extract_table_to_dataframe(image_bgr, lang=tesseract_lang)
    except Exception as exc:  # noqa: BLE001
        st.error(
            "识别失败。请确认已安装 Tesseract OCR 并配置环境变量。"
            f"\n\n错误详情：{exc}"
        )
        st.stop()

with col2:
    st.subheader("网格可视化")
    st.image(cv2.cvtColor(result.overlay_image, cv2.COLOR_BGR2RGB), use_container_width=True)

st.subheader("识别结果预览")
if result.dataframe.empty:
    st.warning("未检测到有效表格网格。建议：提高图片清晰度、确保有表格线。")
    st.stop()

st.dataframe(result.dataframe, use_container_width=True, height=400)

buffer = BytesIO()
result.dataframe.to_excel(buffer, index=False, engine="openpyxl")
buffer.seek(0)

st.download_button(
    label="下载 Excel",
    data=buffer,
    file_name="table_result.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
