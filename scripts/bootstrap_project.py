"""Bootstrap script to scaffold the RAG video search repository."""
from __future__ import annotations

import argparse
from pathlib import Path

STRUCTURE = {
    "config/pipeline.yaml": "# Populate with pipeline configuration\n",
    "docker/Dockerfile": "# Add production Dockerfile\n",
    "docker/docker-compose.yaml": "version: '3.9'\nservices: {}\n",
    "k8s/deployment.yaml": "apiVersion: apps/v1\nkind: Deployment\n",
    "k8s/service.yaml": "apiVersion: v1\nkind: Service\n",
    "src/__init__.py": """Top-level package."""\n",
    "src/api/__init__.py": """FastAPI application."""\n",
    "tests/__init__.py": "",
    "web/static/index.html": "<!DOCTYPE html>\n<html></html>\n",
}

DIRECTORIES = [
    "config",
    "docker",
    "k8s",
    "scripts",
    "src/api",
    "src/config",
    "src/core",
    "src/models",
    "src/utils",
    "tests",
    "web/static",
]


def create_structure(root: Path) -> None:
    for directory in DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    for relative, content in STRUCTURE.items():
        path = root / relative
        if not path.exists():
            path.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap project structure")
    parser.add_argument("root", nargs="?", default=".", help="Target directory")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    create_structure(root)
    print(f"Project structure created at {root}")


if __name__ == "__main__":
    main()
