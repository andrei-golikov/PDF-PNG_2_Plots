# FILENAME: convert_pdf_to_png.py

from pdf2image import convert_from_path
import os
import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# === НАСТРОЙКИ ===
pdf_files = sorted(glob.glob("*.pdf"))

if not pdf_files:
    print("PDF file not found in current directory.")
    exit()

PDF_FILE = pdf_files[0]
OUTPUT_IMAGE = "page_1.png"
DPI = 200
POPLER_BIN = r"C:\tools\poppler-24.08.0\Library\bin"  # ← Укажи свой путь здесь

def main():
    print(f"Converting {PDF_FILE} to PNG...")
    try:
        pages = convert_from_path(PDF_FILE, dpi=DPI, poppler_path=POPLER_BIN, first_page=1, last_page=1)
        if not pages:
            print("Could not extract pages.")
            return
        pages[0].save(OUTPUT_IMAGE, "PNG")
        print(f"Saved: {OUTPUT_IMAGE}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
