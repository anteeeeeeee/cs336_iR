#!/usr/bin/env python3
"""
OCR Text Embedding Indexer for Milvus
======================================

Upload OCR text embeddings to Milvus vector database.
Similar to upload_database.py but processes OCR text instead of images.

Usage:
    python upload_ocr_database.py --root ./output-ocr --collection-name OCR_collection
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm
import torch.multiprocessing as mp

from pymilvus import (
    connections, utility, Collection, CollectionSchema,
    FieldSchema, DataType
)

import open_clip


# -------------------------
# Worker
# -------------------------
def encode_worker(
    rank: int,
    world_size: int,
    indexed_data: List[Tuple[int, dict]],
    device_ids: List[int],
    milvus_host: str,
    milvus_port: str,
    collection_name: str,
    batch_size: int,
    flush_interval: int,
    model_name: str,
    pretrained: str,
    field_names: Tuple[str, str, str, str, str, str],  # (id, filepath, embedding, video_id, frame_id, ocr_text)
):
    id_f, path_f, emb_f, vid_f, frame_f, text_f = field_names

    # --- device init
    if torch.cuda.is_available() and len(device_ids) > 0:
        device = torch.device(f"cuda:{device_ids[rank]}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # --- model
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # --- milvus connection
    connections.connect(host=milvus_host, port=milvus_port)
    collection = Collection(name=collection_name)

    # --- sharding the work
    my_data = indexed_data[rank::world_size]

    buffer = {
        id_f: [],
        path_f: [],
        emb_f: [],
        vid_f: [],
        frame_f: [],
        text_f: [],
        "texts": []
    }
    since_flush = 0

    for _id, data in tqdm(my_data, desc=f"[Worker {rank}]"):
        text = data.get("text", "").strip()
        if not text:
            continue

        buffer["texts"].append(text)
        buffer[id_f].append(_id)
        buffer[path_f].append(data.get("filepath", ""))
        buffer[vid_f].append(data.get("video_id", ""))
        buffer[frame_f].append(data.get("frame_id", 0))
        buffer[text_f].append(text)

        if len(buffer["texts"]) >= batch_size:
            _flush_batch_text(collection, buffer, device, tokenizer, model, id_f, path_f, emb_f, vid_f, frame_f, text_f)
            since_flush += batch_size
            if since_flush >= flush_interval:
                collection.flush()
                since_flush = 0

    # tail
    if buffer["texts"]:
        _flush_batch_text(collection, buffer, device, tokenizer, model, id_f, path_f, emb_f, vid_f, frame_f, text_f)
        collection.flush()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[Worker {rank}] Done.")


@torch.no_grad()
def _flush_batch_text(collection, buffer, device, tokenizer, model, id_f, path_f, emb_f, vid_f, frame_f, text_f):
    """Encode text batch and insert into Milvus"""
    try:
        # Tokenize and encode text
        text_inputs = tokenizer(buffer["texts"]).to(device)
        text_features = model.encode_text(text_inputs)
        # Normalize for cosine similarity
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        embs = text_features.cpu().tolist()

        collection.insert(
            [
                buffer[id_f],
                buffer[path_f],
                embs,
                buffer[vid_f],
                buffer[frame_f],
                buffer[text_f],
            ]
        )
    except Exception as e:
        print(f"[Insert] Error: {e}")
    finally:
        buffer.clear()
        buffer.update({
            id_f: [],
            path_f: [],
            emb_f: [],
            vid_f: [],
            frame_f: [],
            text_f: [],
            "texts": []
        })


# -------------------------
# Helper: prepare encode fn in global scope (workaround for MP pickling)
# -------------------------
def _prepare_encode_cache(model_name: str, pretrained: str, device: torch.device):
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def _encode_text(texts: List[str]) -> torch.Tensor:
        text_inputs = tokenizer(texts).to(device)
        return model.encode_text(text_inputs)

    globals()["_ENCODE_CACHE"] = {"encode_text": _encode_text, "model": model, "tokenizer": tokenizer}


# -------------------------
# Schema utils
# -------------------------
def ensure_collection(
    collection_name: str,
    dimension: int,
    milvus_host: str,
    milvus_port: str,
    recreate: bool,
    index_params: dict,
    field_names=("id", "filepath", "embedding", "video_id", "frame_id", "ocr_text"),
):
    id_f, path_f, emb_f, vid_f, frame_f, text_f = field_names

    connections.connect(host=milvus_host, port=milvus_port)
    if recreate and utility.has_collection(collection_name):
        print(f"⚠️ Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)

    if not utility.has_collection(collection_name):
        print(f"Creating collection: {collection_name}")
        schema = CollectionSchema(
            fields=[
                FieldSchema(name=id_f, dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name=path_f, dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name=emb_f, dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name=vid_f, dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name=frame_f, dtype=DataType.INT64),
                FieldSchema(name=text_f, dtype=DataType.VARCHAR, max_length=2048),  # OCR text can be long
            ],
            description="OCR text embeddings (OpenCLIP) for video retrieval",
        )
        Collection(name=collection_name, schema=schema)
    else:
        print(f"Collection exists: {collection_name}")


# -------------------------
# Discovery
# -------------------------
def discover_ocr_files(root: Path, glob_pat: str, start_index: int) -> List[dict]:
    """Discover OCR JSON files and load their data"""
    json_files = sorted(root.glob(glob_pat))
    if start_index > 0:
        json_files = json_files[start_index:]

    all_data = []
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                video_data = json.load(f)
            
            video_id = json_file.parent.name  # video folder name
            
            for frame_data in video_data:
                if frame_data.get("text", "").strip():
                    all_data.append({
                        "filepath": str(json_file),
                        "video_id": video_id,
                        "frame_id": frame_data.get("frame_id", 0),
                        "text": frame_data.get("text", ""),
                        "seconds": frame_data.get("seconds", 0.0),
                    })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    return all_data


# -------------------------
# Build index (post-insert)
# -------------------------
def build_index_and_load(collection_name: str, index_params: dict, milvus_host: str, milvus_port: str, field_name="embedding"):
    connections.connect(host=milvus_host, port=milvus_port)
    collection = Collection(name=collection_name)
    print("Flushing before indexing...")
    collection.flush()

    has_index = any(idx.field_name == field_name for idx in collection.indexes)
    if not has_index:
        print("Creating index...")
        collection.create_index(field_name, index_params)
    else:
        print("Index already exists; skipping create.")

    print("Loading collection into memory for queries...")
    collection.load()
    print("Index ready.")


# -------------------------
# Argparse + Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Milvus indexer for OCR text files encoded by OpenCLIP.")
    # IO & Data
    p.add_argument("--root", type=str, required=True,
                   help="Root folder containing per-video subfolders with ocr_data/.")
    p.add_argument("--glob", type=str, default="**/ocr_data/*_ocr.json",
                   help="Glob pattern relative to root (default: **/ocr_data/*_ocr.json).")
    p.add_argument("--start-index", type=int, default=0, help="Skip the first N files (after sort).")
    p.add_argument("--id-offset", type=int, default=0, help="Add this offset to assigned integer IDs.")

    # Milvus
    p.add_argument("--collection-name", type=str, default="OCR_collection")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=str, default="19530")
    p.add_argument("--recreate", action="store_true", help="Drop and recreate the collection.")
    p.add_argument("--build-index", action="store_true", help="Build HNSW index after inserts and load collection.")

    # Embeddings / Model
    p.add_argument("--model", type=str, default="ViT-H-14-378-quickgelu")
    p.add_argument("--pretrained", type=str, default="dfn5b")
    p.add_argument("--dimension", type=int, default=-1,
                   help="Embedding dimension. Use -1 to auto-detect from the model (recommended).")

    # Performance
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for text encoding.")
    p.add_argument("--flush-interval", type=int, default=2000)
    p.add_argument("--workers", type=str, default="auto",
                   help="'auto' = #GPUs; integer = explicit #workers; 'cpu' = 1 worker on CPU.")
    return p.parse_args()


def auto_detect_dim(model_name: str, pretrained: str, device: torch.device) -> int:
    """Auto-detect embedding dimension from model"""
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # Test with dummy text
    dummy_text = ["test"]
    with torch.no_grad():
        text_inputs = tokenizer(dummy_text).to(device)
        out = model.encode_text(text_inputs)
    return int(out.shape[-1])


def main():
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Root not found: {root}")
        sys.exit(1)

    # Workers & devices
    cuda_count = torch.cuda.device_count()
    if args.workers == "cpu":
        device_ids = []
        world_size = 1
        use_cpu = True
    elif args.workers == "auto":
        if cuda_count > 0:
            device_ids = list(range(cuda_count))
            world_size = len(device_ids)
            use_cpu = False
        else:
            device_ids = []
            world_size = 1
            use_cpu = True
    else:
        try:
            n = int(args.workers)
            if n <= 0:
                raise ValueError
        except ValueError:
            print("--workers must be 'auto', 'cpu', or a positive integer.")
            sys.exit(1)
        if cuda_count == 0:
            device_ids = []
            world_size = 1
            use_cpu = True
        else:
            device_ids = list(range(min(n, cuda_count)))
            world_size = len(device_ids)
            use_cpu = False

    device = torch.device("cpu" if use_cpu else f"cuda:{device_ids[0]}")

    # Auto-detect embedding dimension if requested
    dimension = args.dimension
    if dimension == -1:
        print("Auto-detecting embedding dimension from model...")
        dimension = auto_detect_dim(args.model, args.pretrained, device)
        print(f"Detected dimension: {dimension}")

    # Ensure collection exists (optionally recreate)
    ensure_collection(
        collection_name=args.collection_name,
        dimension=dimension,
        milvus_host=args.host,
        milvus_port=args.port,
        recreate=args.recreate,
        index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 32, "efConstruction": 512}},
    )

    # Discover OCR files
    ocr_data = discover_ocr_files(root, args.glob, args.start_index)
    if not ocr_data:
        print("No OCR data found. Check --root and --glob.")
        sys.exit(0)

    print(f"Found {len(ocr_data)} OCR text entries")

    # Enumerate with optional offset
    indexed_data = list(enumerate(ocr_data, start=args.id_offset))

    # Prepare global encode cache for CPU single-worker path
    if world_size == 1:
        _prepare_encode_cache(args.model, args.pretrained, device)
        encode_worker(
            rank=0,
            world_size=1,
            indexed_data=indexed_data,
            device_ids=device_ids,
            milvus_host=args.host,
            milvus_port=args.port,
            collection_name=args.collection_name,
            batch_size=args.batch_size,
            flush_interval=args.flush_interval,
            model_name=args.model,
            pretrained=args.pretrained,
            field_names=("id", "filepath", "embedding", "video_id", "frame_id", "ocr_text"),
        )
    else:
        _prepare_encode_cache(args.model, args.pretrained, device)
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            encode_worker,
            args=(
                world_size,
                indexed_data,
                device_ids,
                args.host,
                args.port,
                args.collection_name,
                args.batch_size,
                args.flush_interval,
                args.model,
                args.pretrained,
                ("id", "filepath", "embedding", "video_id", "frame_id", "ocr_text"),
            ),
            nprocs=world_size,
            join=True,
        )

    if args.build_index:
        build_index_and_load(
            collection_name=args.collection_name,
            index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 32, "efConstruction": 512}},
            milvus_host=args.host,
            milvus_port=args.port,
            field_name="embedding",
        )


if __name__ == "__main__":
    main()

