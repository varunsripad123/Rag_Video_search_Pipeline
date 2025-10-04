from pathlib import Path
from uuid import uuid4

from src.utils.io import ManifestRecord, read_manifest, write_manifest


def test_manifest_roundtrip(tmp_path: Path) -> None:
    record = ManifestRecord(
        manifest_id=str(uuid4()),
        tenant_id="tenant",
        stream_id="stream",
        label="label",
        t0="2024-01-01T00:00:00Z",
        t1="2024-01-01T00:00:02Z",
        start_time=0.0,
        end_time=2.0,
        codebook_id="cb",
        model_id="model",
        chunk_path="/tmp/chunk.mp4",
        token_uri="/tmp/token.npz",
        sideinfo_uri="/tmp/token.json",
        embedding_path="/tmp/embed.npy",
        byte_size=1024,
        ratio=2.5,
        hash="blake3:abc",
        quality_stats={"psnr": 30.0, "vmaf": 95.0},
        tags=["label"],
    )

    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, [record])

    loaded = read_manifest(manifest_path)
    assert loaded == [record]
