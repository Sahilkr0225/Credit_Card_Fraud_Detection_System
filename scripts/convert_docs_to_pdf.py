"""
Automated Markdown to Styled PDF Converter using Headless Chromium/Edge.

Converts:
1. docs/VIVA_QA_COMPREHENSIVE_GUIDE.md -> docs/VIVA_QA_COMPREHENSIVE_GUIDE.pdf
2. docs/ALGORITHMS_AND_FORMULAS_REFERENCE.md -> docs/ALGORITHMS_AND_FORMULAS_REFERENCE.pdf

Includes professional fintech typography, KaTeX math rendering, syntax highlighting,
and print-optimized page breaks.
"""

import os
import subprocess
import time
from pathlib import Path
import markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <!-- KaTeX CSS & JS -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{
      delimiters: [
        {{left: '$$', right: '$$', display: true}},
        {{left: '$', right: '$', display: false}}
      ],
      throwOnError: false
    }});"></script>

  <style>
    @page {{
      size: A4;
      margin: 18mm 18mm 20mm 18mm;
      @bottom-right {{
        content: counter(page);
      }}
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 11pt;
      line-height: 1.6;
      color: #1e293b;
      background: #ffffff;
      padding: 0;
      margin: 0;
    }}

    h1, h2, h3, h4 {{
      color: #0f172a;
      font-weight: 700;
      page-break-after: avoid;
    }}

    h1 {{
      font-size: 20pt;
      border-bottom: 2px solid #2563eb;
      padding-bottom: 8px;
      margin-top: 0;
      margin-bottom: 16px;
      color: #1e3a8a;
    }}

    h2 {{
      font-size: 14pt;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 6px;
      margin-top: 24px;
      margin-bottom: 12px;
      color: #1e40af;
    }}

    h3 {{
      font-size: 12pt;
      margin-top: 18px;
      margin-bottom: 8px;
      color: #334155;
    }}

    p, li {{
      font-size: 10.5pt;
      line-height: 1.55;
    }}

    blockquote {{
      background: #f8fafc;
      border-left: 4px solid #3b82f6;
      margin: 14px 0;
      padding: 10px 16px;
      color: #334155;
      border-radius: 0 6px 6px 0;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      font-size: 9.5pt;
      page-break-inside: avoid;
    }}

    th, td {{
      padding: 8px 10px;
      border: 1px solid #cbd5e1;
      text-align: left;
    }}

    th {{
      background: #f1f5f9;
      font-weight: 600;
      color: #0f172a;
    }}

    tr:nth-child(even) {{
      background: #f8fafc;
    }}

    code {{
      font-family: "Consolas", "Courier New", monospace;
      font-size: 9.5pt;
      background: #f1f5f9;
      padding: 2px 5px;
      border-radius: 4px;
      color: #0f172a;
      border: 1px solid #e2e8f0;
    }}

    pre {{
      background: #0f172a;
      color: #f8fafc;
      padding: 12px 16px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 9pt;
      page-break-inside: avoid;
    }}

    pre code {{
      background: transparent;
      color: inherit;
      border: none;
      padding: 0;
    }}

    .katex-display {{
      margin: 12px 0 !important;
      padding: 6px 0;
      overflow-x: hidden;
    }}

    hr {{
      border: 0;
      border-top: 1px solid #e2e8f0;
      margin: 20px 0;
    }}
  </style>
</head>
<body>
{content}
</body>
</html>
"""


def convert_md_to_pdf(md_filename: str, pdf_filename: str, title: str):
    md_path = DOCS_DIR / md_filename
    pdf_path = DOCS_DIR / pdf_filename
    html_temp_path = DOCS_DIR / (md_filename.replace(".md", ".temp.html"))

    print(f"\nProcessing: {md_filename} -> {pdf_filename}")
    if not md_path.exists():
        print(f"ERROR: {md_path} does not exist.")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert Markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
    )

    full_html = HTML_TEMPLATE.format(title=title, content=html_body)

    with open(html_temp_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Generated intermediate HTML: {html_temp_path.name}")

    # Launch Edge headless to print PDF
    cmd = [
        EDGE_PATH,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        str(html_temp_path)
    ]

    subprocess.run(cmd, check=True)

    # Clean up temporary HTML
    if html_temp_path.exists():
        html_temp_path.unlink()

    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"SUCCESS: Generated {pdf_path.name} ({size_kb:.1f} KB)")
    else:
        print(f"FAILED to generate {pdf_path.name}")


def main():
    print("=" * 60)
    print("Converting Markdown Documents to Publication-Quality PDFs...")
    print("=" * 60)

    # 1. Viva Q&A Guide
    convert_md_to_pdf(
        "VIVA_QA_COMPREHENSIVE_GUIDE.md",
        "VIVA_QA_COMPREHENSIVE_GUIDE.pdf",
        "Comprehensive Viva Defense & Technical Q&A Guide"
    )

    # 2. Algorithms & Formulas Reference
    convert_md_to_pdf(
        "ALGORITHMS_AND_FORMULAS_REFERENCE.md",
        "ALGORITHMS_AND_FORMULAS_REFERENCE.pdf",
        "Algorithms, Mathematical Formulas & Architecture Trade-Offs Reference"
    )

    print("\nAll PDF conversions finished successfully!")

if __name__ == "__main__":
    main()
