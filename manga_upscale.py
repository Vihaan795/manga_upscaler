#!/usr/bin/env python3
import sys
import argparse
import subprocess
import requests
import zipfile
import shutil
from pathlib import Path
import torch
import cv2

BASE_DIR = Path("backend")
MODEL_DIR = BASE_DIR / "models"

# Model URLs from Hugging Face
MODEL_URLS = {
    "best": "https://huggingface.co/Asif782/upscaler_models/resolve/main/bestforboth.zip",
    "bw": "https://huggingface.co/Asif782/upscaler_models/resolve/main/bwmodels.zip",
    "color": "https://huggingface.co/Asif782/upscaler_models/resolve/main/colormodels.zip",
}

BW_DEFAULT = MODEL_DIR / "4x_MangaJaNai_1200p_V1_ESRGAN_70k.pth"
COLOR_DEFAULT = MODEL_DIR / "4x_IllustrationJaNai_V2standard_DAT2_27k.safetensors"

# Mapping short names → full paths (extendable)
MODEL_MAP = {
    "2x_MangaJaNai": MODEL_DIR / "2x_MangaJaNai_1920p_V1_ESRGAN_70k.pth",
    "4x_MangaJaNai": BW_DEFAULT,
    "4x_IllustrationJaNai": COLOR_DEFAULT,
}

# -------------------------
# Auto-padding utility
# -------------------------
def pad_to_multiple(img_path: Path, multiple: int):
    """Pad image so height/width are divisible by `multiple`."""
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️ Skipping invalid image: {img_path}")
        return

    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple

    if new_h == h and new_w == w:
        return  # already divisible

    pad_bottom = new_h - h
    pad_right = new_w - w

    padded = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
    cv2.imwrite(str(img_path), padded)
    print(f"Padded {img_path.name} -> ({h},{w}) -> ({new_h},{new_w})")

def preprocess_images(folder: Path, scale: int):
    """Ensure all images in folder are padded for the given scale."""
    if not folder or not folder.exists():
        return
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for img_path in folder.glob(ext):
            pad_to_multiple(img_path, scale)

def download_and_extract(model_type: str):
    """Download and extract a specific model pack."""
    if model_type not in MODEL_URLS:
        print(f"Error: Invalid model type '{model_type}'.")
        return

    url = MODEL_URLS[model_type]
    zip_filename = Path(url).name
    zip_path = MODEL_DIR / zip_filename
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_type} models from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                done = int(50 * downloaded / total) if total else 0
                sys.stdout.write(f"\r[{'#' * done}{'.' * (50 - done)}]")
                sys.stdout.flush()
    
    print(f"\nExtracting {zip_filename}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODEL_DIR)
    zip_path.unlink()
    print(f"Models extracted to {MODEL_DIR}")

# -------------------------
# Extractor
# -------------------------
def extract(input_dir, overwrite=False):
    input_dir = Path(input_dir)
    zip_files = list(input_dir.glob("*.zip")) + list(input_dir.glob("*.cbz"))
    print(f"Found {len(zip_files)} archive(s).")

    for zip_file in zip_files:
        extract_folder = zip_file.with_suffix("")  # remove .zip/.cbz
        if extract_folder.exists() and not overwrite:
            print(f"{zip_file.name} already extracted, skipping.")
            continue

        print(f"Extracting {zip_file.name}...")
        if overwrite and extract_folder.exists():
            shutil.rmtree(extract_folder)
        extract_folder.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(extract_folder)
        zip_file.unlink()  # remove archive after extraction

# -------------------------
# Upscaler
# -------------------------
def resolve_model(name_or_path: str) -> Path:
    """Resolve model alias or return full path."""
    candidate = Path(name_or_path)

    # If user gave a direct path, just use it
    if candidate.exists():
        return candidate

    # Try alias lookup in MODEL_MAP
    if name_or_path in MODEL_MAP:
        return Path(MODEL_MAP[name_or_path])

    # Try auto-detect inside MODEL_DIR (prefix match)
    matches = list(MODEL_DIR.glob(f"{name_or_path}*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Model '{name_or_path}' not found in {MODEL_DIR}")


def upscale(bw_dir, color_dir, output_dir, bw_model=None, color_model=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("No GPU detected. ESRGAN will run very slow on CPU.")

    def detect_scale(model_path: Path):
        """Infer scale factor from model name (fallback=2)."""
        name = model_path.stem
        if name.startswith("4x"): 
            return 4
        if name.startswith("2x"): 
            return 2
        return 2

    def run_upscale(label, in_dir, model_name):
        if not in_dir or not model_name:
            return
        in_dir = Path(in_dir)
        if not in_dir.exists() or not any(in_dir.iterdir()):
            return

        model = resolve_model(model_name)
        scale = detect_scale(model)

        print(f"Preprocessing {label} images (scale={scale})...")
        preprocess_images(in_dir, scale)

        print(f"Upscaling {label} pages with {model.name}...")
        subprocess.run([
            sys.executable, "backend/upscale.py",
            "-se", "-i", str(in_dir), "-o", str(output_dir), str(model)
        ], check=True)

    # Run only if args provided
    run_upscale("black & white", bw_dir, bw_model)
    run_upscale("color", color_dir, color_model)

    print(f"Done! Upscaled images saved in {output_dir}")


# -------------------------
# CLI Entry
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Manga Upscaler CLI")
    subparsers = parser.add_subparsers(dest="command")

    # download
    download_parser = subparsers.add_parser("download", help="Download model packs")
    download_parser.add_argument("model_type", choices=["best", "bw", "color"], help="Type of models to download")

    # extract
    extract_parser = subparsers.add_parser("extract", help="Extract .zip or .cbz archives")
    extract_parser.add_argument("--input", required=True, help="Input directory containing archives")
    extract_parser.add_argument("--overwrite", action="store_true", help="Re-extract even if folder exists")

    # upscale
    upscale_parser = subparsers.add_parser("upscale", help="Run manga upscaler")
    upscale_parser.add_argument("--bw", help="Directory with black & white pages")
    upscale_parser.add_argument("--color", help="Directory with color pages")
    upscale_parser.add_argument("--output", required=True, help="Directory for results")
    upscale_parser.add_argument("--model-bw", default=str(BW_DEFAULT), help="BW model path or short name")
    upscale_parser.add_argument("--model-color", default=str(COLOR_DEFAULT), help="Color model path or short name")

    args = parser.parse_args()

    if args.command == "download":
        download_and_extract(args.model_type)
    elif args.command == "extract":
        extract(args.input, overwrite=args.overwrite)
    elif args.command == "upscale":
        upscale(args.bw, args.color, args.output, args.model_bw, args.model_color)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
