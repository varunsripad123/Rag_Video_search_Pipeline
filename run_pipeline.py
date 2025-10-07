"""CLI entrypoint for the RAG video search pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat

from src.config import load_config
from src.core.pipeline import run_pipeline
from src.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG video search pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Config path",
    )
    parser.add_argument(
        "--enable-labeling",
        action="store_true",
        help="Enable auto-labeling (YOLO, BLIP-2, Whisper, VideoMAE)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config)

    print("Starting pipeline with config:\n" + pformat(config.model_dump()))
    
    if args.enable_labeling:
        print("\n🎯 Auto-labeling ENABLED: YOLO + BLIP-2 + Whisper + VideoMAE\n")

    index_path = run_pipeline(config, enable_auto_labeling=args.enable_labeling)

    print("\nPipeline Complete\n")
    print(f"{'Artifact':<20} {'Location'}")
    print(f"{'-'*20} {'-'*40}")
    print(f"{'FAISS Index':<20} {str(index_path)}")
    print(f"{'Metadata':<20} {str(config.data.processed_dir / 'metadata.json')}")
    
    if args.enable_labeling:
        print(f"{'Auto-labels':<20} Stored in metadata.json (auto_labels field)")


if __name__ == "__main__":
    main()
