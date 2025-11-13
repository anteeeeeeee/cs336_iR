#!/usr/bin/env python3
"""
OCR Extraction from Videos
===========================

Extract text from video frames using OCR (Optical Character Recognition).
Similar to get_keyframes.py but extracts text instead of visual keyframes.

Usage:
    python get_ocr.py --input-folder /path/to/videos --output-base ./output-ocr
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import torch
from tqdm import tqdm

# OCR library - using easyocr as default (can switch to paddleocr)
try:
    import easyocr
    OCR_LIB = "easyocr"
except ImportError:
    try:
        from paddleocr import PaddleOCR
        OCR_LIB = "paddleocr"
    except ImportError:
        raise ImportError(
            "Please install OCR library: pip install easyocr OR pip install paddlepaddle paddleocr"
        )


def init_ocr_reader(lang: List[str] = ['en'], gpu: bool = True):
    """Initialize OCR reader"""
    if OCR_LIB == "easyocr":
        reader = easyocr.Reader(lang, gpu=gpu and torch.cuda.is_available())
        return reader
    else:  # paddleocr
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=gpu and torch.cuda.is_available())
        return ocr


def extract_text_easyocr(reader, frame) -> List[Dict]:
    """Extract text from frame using EasyOCR"""
    results = reader.readtext(frame)
    texts = []
    for (bbox, text, confidence) in results:
        texts.append({
            "text": text,
            "confidence": float(confidence),
            "bbox": bbox
        })
    return texts


def extract_text_paddleocr(ocr, frame) -> List[Dict]:
    """Extract text from frame using PaddleOCR"""
    results = ocr.ocr(frame, cls=True)
    texts = []
    if results and results[0]:
        for line in results[0]:
            if line:
                bbox, (text, confidence) = line
                texts.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": bbox
                })
    return texts


def extract_text_from_frame(ocr_reader, frame, ocr_lib: str) -> List[Dict]:
    """Extract text from a single frame"""
    if ocr_lib == "easyocr":
        return extract_text_easyocr(ocr_reader, frame)
    else:
        return extract_text_paddleocr(ocr_reader, frame)


def process_video(
    video_path: str,
    out_dir: str,
    maps_dir: str,
    ocr_reader,
    ocr_lib: str,
    skip_frames: int,
    min_text_length: int = 2,
    min_confidence: float = 0.5,
) -> int:
    """
    Process a single video and extract OCR text from frames
    
    Args:
        video_path: Path to video file
        out_dir: Output directory for OCR data
        maps_dir: Directory for CSV mapping files
        ocr_reader: Initialized OCR reader
        ocr_lib: OCR library name ('easyocr' or 'paddleocr')
        skip_frames: Process every (skip_frames + 1)-th frame
        min_text_length: Minimum text length to keep
        min_confidence: Minimum confidence threshold
    
    Returns:
        Number of frames with extracted text
    """
    os.makedirs(out_dir, exist_ok=True)
    ocr_dir = os.path.join(out_dir, "ocr_data")
    os.makedirs(ocr_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    map_csv = os.path.join(maps_dir, f"{video_name}_ocr_map.csv")
    ocr_json = os.path.join(ocr_dir, f"{video_name}_ocr.json")

    frame_ocr_data = []
    frame_id = 0
    frames_with_text = 0

    with open(map_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["FrameID", "Seconds", "TextCount", "HasText"])

        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=f"OCR {video_name}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Process every (skip_frames + 1)-th frame
            if skip_frames < 0 or (frame_id % (skip_frames + 1) == 0):
                # Extract text from frame
                texts = extract_text_from_frame(ocr_reader, frame, ocr_lib)
                
                # Filter by confidence and length
                filtered_texts = [
                    t for t in texts
                    if len(t["text"].strip()) >= min_text_length
                    and t["confidence"] >= min_confidence
                ]

                if filtered_texts:
                    # Combine all text from frame
                    combined_text = " ".join([t["text"] for t in filtered_texts])
                    
                    frame_data = {
                        "frame_id": frame_id,
                        "seconds": frame_id / fps,
                        "text": combined_text,
                        "texts": filtered_texts,
                        "text_count": len(filtered_texts)
                    }
                    frame_ocr_data.append(frame_data)
                    
                    writer.writerow([
                        frame_id,
                        f"{frame_id / fps:.2f}",
                        len(filtered_texts),
                        "1"
                    ])
                    frames_with_text += 1
                else:
                    writer.writerow([
                        frame_id,
                        f"{frame_id / fps:.2f}",
                        0,
                        "0"
                    ])

            frame_id += 1
            pbar.update(1)

        pbar.close()
    cap.release()

    # Save OCR data as JSON
    with open(ocr_json, "w", encoding="utf-8") as f:
        json.dump(frame_ocr_data, f, ensure_ascii=False, indent=2)

    return frames_with_text


def process_all_videos(
    input_folder: str,
    output_base: str,
    skip_frames: int,
    pattern: str,
    lang: List[str],
    gpu: bool,
    min_text_length: int,
    min_confidence: float,
    start_index: int,
):
    """Process all videos in input folder"""
    # Initialize OCR reader once (reuse for all videos)
    print(f"Initializing OCR reader ({OCR_LIB})...")
    ocr_reader = init_ocr_reader(lang=lang, gpu=gpu)
    print("OCR reader ready.")

    # Discover videos
    video_files = sorted(glob.glob(os.path.join(input_folder, pattern)))
    if not video_files:
        print(f"No videos matched: {os.path.join(input_folder, pattern)}")
        return

    maps_dir = os.path.join(output_base, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    for video_path in video_files[start_index:]:
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(output_base, name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Processing video: {name}")
        try:
            text_count = process_video(
                video_path=video_path,
                out_dir=out_dir,
                maps_dir=maps_dir,
                ocr_reader=ocr_reader,
                ocr_lib=OCR_LIB,
                skip_frames=skip_frames,
                min_text_length=min_text_length,
                min_confidence=min_confidence,
            )
            print(f"Frames with text: {text_count}")
        except Exception as e:
            print(f"Error processing {name}: {e}")
        print("-" * 20)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract OCR text from videos using EasyOCR or PaddleOCR."
    )
    p.add_argument("--input-folder", type=str, required=True,
                   help="Folder containing videos.")
    p.add_argument("--output-base", type=str, default="./output-ocr",
                   help="Base output folder (per-video subfolders + maps/).")
    p.add_argument("--pattern", type=str, default="*.mp4",
                   help="Glob pattern for videos (e.g., '*.mp4').")
    p.add_argument("--start-index", type=int, default=0,
                   help="Start processing from this sorted index.")
    p.add_argument("--skip-frames", type=int, default=5,
                   help="Process every (skip_frames + 1)-th frame. Use -1 to process every frame.")
    p.add_argument("--lang", type=str, nargs="+", default=["en"],
                   help="OCR language codes (e.g., 'en' for English, 'vi' for Vietnamese).")
    p.add_argument("--min-text-length", type=int, default=2,
                   help="Minimum text length to keep.")
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="Minimum OCR confidence threshold (0.0-1.0).")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU even if CUDA is available.")
    return p.parse_args()


def main():
    args = parse_args()
    gpu = not args.cpu and torch.cuda.is_available()
    
    process_all_videos(
        input_folder=args.input_folder,
        output_base=args.output_base,
        skip_frames=args.skip_frames,
        pattern=args.pattern,
        lang=args.lang,
        gpu=gpu,
        min_text_length=args.min_text_length,
        min_confidence=args.min_confidence,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()

