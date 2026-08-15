"""Pure helpers shared by the Streamlit portfolio application."""

import json
from json import JSONDecodeError
from pathlib import Path
from uuid import uuid4

from semanticnav.pipeline import RunSummary


ARTIFACT_NAMES = (
    "annotated.mp4",
    "results.json",
    "metrics.csv",
    "semantic_map.png",
    "depth_preview.png",
    "path.csv",
    "run_metadata.json",
)


def save_uploaded_video(
    filename: str,
    data: bytes,
    upload_root: Path,
) -> Path:
    safe_name = Path(filename).name
    if Path(safe_name).suffix.lower() != ".mp4":
        raise ValueError("只支持MP4视频")
    if not data:
        raise ValueError("上传视频不能为空")

    upload_root.mkdir(parents=True, exist_ok=True)
    output_path = upload_root / f"{Path(safe_name).stem}-{uuid4().hex[:8]}.mp4"
    output_path.write_bytes(data)
    return output_path


def collect_artifacts(run_dir: Path) -> dict[str, Path]:
    return {
        name: run_dir / name
        for name in ARTIFACT_NAMES
        if (run_dir / name).is_file()
    }


def format_metrics(summary: RunSummary) -> dict[str, str]:
    return {
        "平均FPS": f"{summary.average_fps:.2f}",
        "YOLO平均推理": f"{summary.average_inference_ms:.2f} ms",
        "P95总延迟": f"{summary.p95_total_ms:.2f} ms",
        "处理帧数": str(summary.frame_count),
    }


def load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError) as error:
        raise ValueError(f"无法读取JSON文件 {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}的根节点必须是JSON对象")
    return payload
