"""Configuration management using Pydantic models."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ProjectSettings(BaseModel):
    """Metadata about the project deployment."""

    name: str = "RAG Video Search"
    environment: str = "development"


class DataSettings(BaseModel):
    """Location and chunking parameters for the dataset."""

    root_dir: Path = Path("datasets")
    processed_dir: Path = Path("data/processed")
    chunk_duration: float = 4.0
    frame_rate: int = 2
    min_frames: int = 8
    batch_size: int = 4
    default_tenant: str = "tenant-default"
    stream_prefix: str = "stream"

    @model_validator(mode="before")
    def _expand_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("root_dir", "processed_dir"):
            if key in values:
                values[key] = Path(values[key]).expanduser().resolve()
        return values


class ModelSettings(BaseModel):
    """Model selection and inference configuration."""

    device: str = "cuda"
    precision: str = Field("fp16", pattern=r"fp16|fp32|int8")
    quantize: bool = False
    clip_model_name: str = "openai/clip-vit-base-patch32"
    videomae_model_name: str = "MCG-NJU/videomae-base"
    videoswin_model_name: str = "microsoft/videoswin-base"


class CodecSettings(BaseModel):
    """Neural codec hyper-parameters."""

    model_config = ConfigDict(protected_namespaces=())

    enable_motion_estimation: bool = True
    entropy_bottleneck: bool = True
    quantization_bits: int = Field(ge=4, le=16, default=8)
    residual_channels: int = 128
    latent_channels: int = 256
    codebook_id: str = "cb_v1"
    model_id: str = "video_encoder_v1"


class IndexSettings(BaseModel):
    """FAISS index parameters."""

    faiss_index_path: Path = Path("data/index/faiss.index")
    dim: int = 1536
    nlist: int = 1024
    nprobe: int = 8
    use_gpu: bool = True
    refresh_interval: int = Field(3600, description="Seconds between index refreshes")

    @model_validator(mode="before")
    def _expand_index_path(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "faiss_index_path" in values:
            values["faiss_index_path"] = Path(values["faiss_index_path"]).expanduser().resolve()
        return values


class APISettings(BaseModel):
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    query_expansion_top_k: int = 5
    max_history: int = 5


class SecuritySettings(BaseModel):
    """Authentication and authorization configuration."""

    api_keys: List[str] = Field(default_factory=lambda: ["changeme"])
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 40
    redis_url: Optional[str] = None


class MonitoringSettings(BaseModel):
    """Monitoring configuration for Prometheus and tracing."""

    prometheus_port: int = 9090
    enable_tracing: bool = False


class LoggingSettings(BaseModel):
    """Logging configuration for structlog + stdlib."""

    model_config = ConfigDict(populate_by_name=True)

    level: str = Field("INFO", pattern=r"DEBUG|INFO|WARNING|ERROR|CRITICAL")
    json_output: bool = Field(True, alias="json")
    file: Optional[Path] = None

    @model_validator(mode="before")
    def _expand_file(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        file = values.get("file")
        if file:
            values["file"] = Path(file).expanduser().resolve()
        return values


class FrontendSettings(BaseModel):
    """Static assets served alongside the API."""

    static_dir: Path = Path("web/static")

    @model_validator(mode="before")
    def _expand_dir(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "static_dir" in values:
            values["static_dir"] = Path(values["static_dir"]).expanduser().resolve()
        return values


class AppConfig(BaseModel):
    """Top-level configuration object."""

    project: ProjectSettings = ProjectSettings()
    data: DataSettings = DataSettings()
    models: ModelSettings = ModelSettings()
    codec: CodecSettings = CodecSettings()
    index: IndexSettings = IndexSettings()
    api: APISettings = APISettings()
    security: SecuritySettings = SecuritySettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    logging: LoggingSettings = LoggingSettings()
    frontend: FrontendSettings = FrontendSettings()

    @classmethod
    def from_file(cls, path: Path | str) -> "AppConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.model_validate(data)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base.get(key, {}), value)
        else:
            base[key] = value
    return base


def _load_environment_overrides(prefix: str = "RAG_") -> Dict[str, Any]:
    """Parse environment variables into nested dictionaries."""

    overrides: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        *sections, field = key[len(prefix) :].lower().split("__")
        cursor: Dict[str, Any] = overrides
        for section in sections:
            cursor = cursor.setdefault(section, {})  # type: ignore[assignment]
        cursor[field] = _coerce_value(value)
    return overrides


def _coerce_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@lru_cache(maxsize=1)
def load_config(path: Path | str | None = None) -> AppConfig:
    """Load configuration from YAML and environment overrides."""

    if path is None:
        path = Path(os.environ.get("RAG_CONFIG_PATH", "config/pipeline.yaml"))
    base = AppConfig.from_file(path)
    overrides = _load_environment_overrides()
    if overrides:
        merged = _deep_update(base.model_dump(), overrides)
        base = AppConfig.model_validate(merged)
    return base


__all__ = [
    "AppConfig",
    "ProjectSettings",
    "DataSettings",
    "ModelSettings",
    "CodecSettings",
    "IndexSettings",
    "APISettings",
    "SecuritySettings",
    "MonitoringSettings",
    "LoggingSettings",
    "FrontendSettings",
    "load_config",
]
