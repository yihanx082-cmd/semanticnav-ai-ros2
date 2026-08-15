import importlib
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
SAFETY_NOTICE = (
    "本页面展示相对深度、局部示意地图和动作建议，尚未接入真实机器人底盘；"
    "结果不构成安全控制指令。"
)


def test_importing_streamlit_app_does_not_load_yolo(monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        "semanticnav.tracking.YOLOByteTracker.__init__",
        lambda *args, **kwargs: calls.append(1),
    )
    sys.modules.pop("app.streamlit_app", None)

    module = importlib.import_module("app.streamlit_app")

    assert calls == []
    assert module.SAFETY_NOTICE == SAFETY_NOTICE


def test_initial_page_shows_controls_and_safety_notice() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "SemanticNav AI"
    assert app.info[0].value == SAFETY_NOTICE
    assert app.sidebar.text_area[0].label == "机器人任务"
    assert app.sidebar.slider[0].label == "检测置信度"
    assert app.sidebar.selectbox[0].label == "YOLO输入尺寸"
    assert app.sidebar.toggle[0].label == "启用相对深度"
    assert app.sidebar.button[0].label == "开始处理"
