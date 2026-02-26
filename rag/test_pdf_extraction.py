"""
测试PDF文本提取
用于诊断PDF文件无法提取文本的原因
"""
import os
import sys

# 设置输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 测试PyPDF2
try:
    from PyPDF2 import PdfReader as PyPDF2Reader
    print("[OK] PyPDF2 available")
    HAS_PYPDF2 = True
except ImportError:
    print("[FAIL] PyPDF2 not available")
    HAS_PYPDF2 = False

# 测试PyMuPDF
try:
    import fitz
    print("[OK] PyMuPDF (fitz) available")
    HAS_PYMUPDF = True
except ImportError:
    try:
        import pymupdf as fitz
        print("[OK] PyMuPDF (pymupdf) available")
        HAS_PYMUPDF = True
    except ImportError:
        print("[FAIL] PyMuPDF not available")
        HAS_PYMUPDF = False

# 测试pdfplumber
try:
    import pdfplumber
    print("[OK] pdfplumber available")
    HAS_PDFPLUMBER = True
except ImportError:
    print("[FAIL] pdfplumber not available")
    HAS_PDFPLUMBER = False

print("\n" + "="*50)
print("开始测试PDF文件提取")
print("="*50)

# 测试第一个PDF文件
pdf_path = os.path.join("data", "doctor", "01_第一章.pdf")

if not os.path.exists(pdf_path):
    print(f"[FAIL] File not found: {pdf_path}")
    sys.exit(1)

print(f"\nTest file: {pdf_path}")
print(f"File size: {os.path.getsize(pdf_path)} bytes")

# 方法1: PyPDF2
if HAS_PYPDF2:
    print("\n--- Method 1: PyPDF2 ---")
    try:
        reader = PyPDF2Reader(pdf_path)
        print(f"Pages: {len(reader.pages)}")
        text = ""
        for i, page in enumerate(reader.pages[:3]):  # Test first 3 pages
            page_text = page.extract_text()
            text += page_text
            print(f"  Page {i+1}: {len(page_text)} chars")
        print(f"Total text length: {len(text)} chars")
        if text.strip():
            print(f"First 100 chars: {repr(text[:100])}")
        else:
            print("[WARNING] No text extracted")
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

# 方法2: PyMuPDF
if HAS_PYMUPDF:
    print("\n--- Method 2: PyMuPDF ---")
    try:
        doc = fitz.open(pdf_path)
        print(f"Pages: {len(doc)}")
        text = ""
        for i in range(min(3, len(doc))):  # Test first 3 pages
            page = doc[i]
            page_text = page.get_text()
            text += page_text
            print(f"  Page {i+1}: {len(page_text)} chars")
        print(f"Total text length: {len(text)} chars")
        if text.strip():
            print(f"First 100 chars: {repr(text[:100])}")
        else:
            print("[WARNING] No text extracted")
        doc.close()
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

# 方法3: pdfplumber
if HAS_PDFPLUMBER:
    print("\n--- Method 3: pdfplumber ---")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Pages: {len(pdf.pages)}")
            text = ""
            for i, page in enumerate(pdf.pages[:3]):  # Test first 3 pages
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                    print(f"  Page {i+1}: {len(page_text)} chars")
                else:
                    print(f"  Page {i+1}: Cannot extract text")
            print(f"Total text length: {len(text)} chars")
            if text.strip():
                print(f"First 100 chars: {repr(text[:100])}")
            else:
                print("[WARNING] No text extracted")
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*50)
print("Test completed")
print("="*50)
