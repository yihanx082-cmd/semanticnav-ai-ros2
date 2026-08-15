"""Interactive portfolio demo for the SemanticNav video pipeline."""

from pathlib import Path

import streamlit as st

from semanticnav.app_support import (
    collect_artifacts,
    format_metrics,
    load_json,
    save_uploaded_video,
)
from semanticnav.config import AppConfig, load_config
from semanticnav.depth import DepthModelUnavailable, RelativeDepthEstimator
from semanticnav.pipeline import RunSummary, VideoPipeline
from semanticnav.tracking import YOLOByteTracker


SAFETY_NOTICE = (
    "本页面展示相对深度、局部示意地图和动作建议，尚未接入真实机器人底盘；"
    "结果不构成安全控制指令。"
)


@st.cache_resource(show_spinner=False)
def load_tracker(
    model_name: str,
    confidence: float,
    image_size: int,
    tracker_name: str,
) -> YOLOByteTracker:
    return YOLOByteTracker(
        model_name=model_name,
        confidence=confidence,
        image_size=image_size,
        tracker_name=tracker_name,
    )


@st.cache_resource(show_spinner=False)
def load_depth_estimator(model_name: str) -> RelativeDepthEstimator:
    return RelativeDepthEstimator(model_name)


def build_pipeline(
    config: AppConfig,
    confidence: float,
    image_size: int,
    enable_depth: bool,
    depth_frame_interval: int,
) -> VideoPipeline:
    tracker = load_tracker(
        config.model.name,
        confidence,
        image_size,
        config.tracker.name,
    )
    depth_estimator = None
    if enable_depth:
        depth_estimator = load_depth_estimator(config.depth.model_name)
    return VideoPipeline(
        tracker,
        depth_estimator,
        depth_frame_interval=depth_frame_interval,
        near_threshold=config.depth.near_threshold,
        far_threshold=config.depth.far_threshold,
        grid_shape=(config.mapping.rows, config.mapping.columns),
        obstacle_inflation_cells=config.mapping.obstacle_inflation_cells,
    )


def render_results(summary: RunSummary) -> None:
    metrics = format_metrics(summary)
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items(), strict=True):
        column.metric(label, value)

    st.subheader("感知与规划结果")
    video_path = summary.run_dir / "annotated.mp4"
    if video_path.is_file():
        st.video(str(video_path))

    image_columns = st.columns(2)
    depth_path = summary.run_dir / "depth_preview.png"
    if depth_path.is_file():
        image_columns[0].image(
            str(depth_path),
            caption="相对深度预览（非米制）",
            use_container_width=True,
        )
    else:
        image_columns[0].info("本次运行未启用相对深度。")
    map_path = summary.run_dir / "semantic_map.png"
    if map_path.is_file():
        image_columns[1].image(
            str(map_path),
            caption="局部示意地图（非米制）",
            use_container_width=True,
        )

    if summary.planning_result.success:
        st.success(
            "A*示意路径规划成功，"
            f"长度 {summary.planning_result.path_length_cells:.2f} 个栅格。"
        )
    else:
        st.warning(
            "A*示意路径规划失败："
            f"{summary.planning_result.failure_reason or '原因未知'}"
        )

    st.subheader("结构化任务")
    st.json(summary.task.model_dump(mode="json"))
    results_path = summary.run_dir / "results.json"
    if results_path.is_file():
        with st.expander("查看逐帧结果JSON"):
            st.json(load_json(results_path))

    st.subheader("下载结果")
    for name, path in collect_artifacts(summary.run_dir).items():
        st.download_button(
            label=f"下载 {name}",
            data=path.read_bytes(),
            file_name=name,
            key=f"download-{summary.run_id}-{name}",
        )


def main() -> None:
    st.set_page_config(page_title="SemanticNav AI", page_icon="🤖", layout="wide")
    st.title("SemanticNav AI")
    st.caption("室内机器人语义感知与非米制路径规划软件原型")
    st.info(SAFETY_NOTICE)

    config = load_config(Path("configs/default.yaml"))
    with st.sidebar:
        st.header("运行设置")
        uploaded_file = st.file_uploader("上传室内MP4", type=["mp4"])
        task_text = st.text_area(
            "机器人任务",
            value="去椅子附近，避开人和宠物",
        )
        confidence = st.slider(
            "检测置信度",
            min_value=0.10,
            max_value=0.90,
            value=float(config.model.confidence),
            step=0.05,
        )
        image_size = st.selectbox(
            "YOLO输入尺寸",
            options=[320, 480, 640],
            index=[320, 480, 640].index(config.model.image_size),
        )
        enable_depth = st.toggle("启用相对深度", value=False)
        depth_frame_interval = st.slider(
            "深度帧间隔",
            min_value=1,
            max_value=10,
            value=config.depth.frame_interval,
            disabled=not enable_depth,
        )
        max_frames = st.number_input(
            "最大处理帧数",
            min_value=1,
            max_value=1000,
            value=30,
            step=1,
        )
        run_clicked = st.button("开始处理", type="primary", use_container_width=True)

    if run_clicked:
        if uploaded_file is None:
            st.error("请先上传MP4视频。")
        else:
            progress_bar = st.progress(0.0, text="准备处理视频")
            try:
                input_path = save_uploaded_video(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    Path(config.output.root) / "uploads",
                )
                pipeline = build_pipeline(
                    config,
                    confidence,
                    image_size,
                    enable_depth,
                    depth_frame_interval,
                )

                def report_progress(processed: int, total: int) -> None:
                    fraction = min(processed / max(total, 1), 1.0)
                    progress_bar.progress(
                        fraction,
                        text=f"正在处理 {processed}/{total} 帧",
                    )

                summary = pipeline.run(
                    input_path,
                    task_text,
                    config.output.root,
                    progress=report_progress,
                    max_frames=int(max_frames),
                )
                st.session_state["last_summary"] = summary
                progress_bar.progress(1.0, text="处理完成")
            except DepthModelUnavailable as error:
                st.error(f"相对深度模型不可用：{error}")
                with st.expander("错误详情"):
                    st.code(type(error).__name__)
            except Exception as error:
                st.error(str(error))
                with st.expander("错误详情"):
                    st.code(type(error).__name__)

    summary = st.session_state.get("last_summary")
    if isinstance(summary, RunSummary):
        render_results(summary)
    elif not run_clicked:
        st.markdown(
            "上传一段室内MP4后即可查看检测、跟踪、相对深度、"
            "局部示意地图、A*路径和性能指标。"
        )


if __name__ == "__main__":
    main()
