"""CLI entrypoint for the RAG video search pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.core.pipeline import run_pipeline
from src.utils import configure_logging

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG video search pipeline")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"), help="Config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config)
    console.log("Starting pipeline", config=config.model_dump())
    index_path = run_pipeline(config)
    table = Table(title="Pipeline Complete")
    table.add_column("Artifact")
    table.add_column("Location")
    table.add_row("FAISS Index", str(index_path))
    table.add_row("Manifest", str(config.data.processed_dir / "manifest.json"))
    console.print(table)


if __name__ == "__main__":
    main()
