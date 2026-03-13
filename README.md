# 图片表格转 Excel（Python + Streamlit）

这个项目提供一个可视化网页应用：
- 上传图片形式的表格
- 可视化显示识别出的网格线
- 预览识别结果
- 一键下载 Excel
- 生成并下载 LaTeX 表格代码（`.tex`）

## 1. 环境准备

### Python 依赖
```powershell
pip install -r requirements.txt
```

### 安装 Tesseract OCR
本项目使用 `pytesseract`，需要系统安装 Tesseract。

Windows 常见步骤：
1. 安装 Tesseract（例如通过 UB Mannheim 发行版）。
2. 把安装目录加入系统 `PATH`，如：
   - `C:\Program Files\Tesseract-OCR`
3. 安装中文语言包（`chi_sim`）与英文语言包（`eng`）。
4. 如果未加到 `PATH`，可在应用左侧直接填写：
   - `C:\Program Files\Tesseract-OCR\tesseract.exe`

安装完成后，在终端执行：
```powershell
tesseract --version
```
若有版本输出，说明可用。

## 2. 启动应用
```powershell
streamlit run app.py
```

浏览器打开后：
1. 上传表格图片
2. 调整语言包（默认 `chi_sim+eng`）
3. 查看网格可视化和识别预览
4. 点击下载 Excel

## 3. 适用与限制
- 适合有明显表格线（横线/竖线）的图片。
- 对无边框表格、倾斜严重、低清晰度图片识别效果会下降。
- 如需更高精度可升级为深度学习表格结构模型（如 PaddleOCR PP-Structure）。
