# SemanticNav AI 快速版实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 14 天交付可演示的室内机器人语义导航软件原型，第 3～4 周补充公开 RGB-D 三维定位和一个作品集增强项。

**Architecture:** 本地 MP4 经 YOLO 和 ByteTrack 产生持续轨迹，轻量单目深度模型给出相对远近，再映射到局部示意栅格并由 A* 规划路径。Streamlit 仅负责交互与展示，算法均封装为可测试的纯 Python 模块，后续可继续封装为 ROS 2 节点。

**Tech Stack:** Python 3.11、OpenCV、Ultralytics YOLO、ByteTrack、Depth Anything V2 Small、NumPy、Pydantic、Streamlit、pytest；第 3 周使用公开 RGB-D 数据。

## Global Constraints

- 开发目录：`D:\semanticnav-ai-ros2`；运行平台：Windows 11。
- 首版使用现有 Core Ultra 5 125H、32 GB 内存和 Intel Arc 核显，不依赖 CUDA。
- 首版不购买 RGB-D 相机、Jetson、激光雷达或移动底盘。
- 检测使用 Ultralytics 官方 nano 级预训练权重；实际模型名称写入配置和运行元数据。
- 跟踪固定使用 ByteTrack，连续视频帧必须保持 tracker state。
- 单目深度只输出相对深度与 `near/mid/far/unknown`，不得伪造米制距离。
- 俯视图必须标注“局部示意地图（非米制）”。
- 自然语言首版使用确定性规则，不依赖 LLM。
- 输出统一写入 `outputs/<run_id>/`，不同运行不得互相覆盖。
- 模型、视频、数据集、输出、密钥和虚拟环境不得提交 Git。
- 每项任务均按“失败测试 → 最小实现 → 测试通过 → 提交”执行。

---

## 1. 交付范围

### 1.1 两周必须完成

```text
MP4上传
  → YOLO检测
  → ByteTrack跟踪
  → 单目相对深度
  → 局部示意栅格
  → A*避障路径
  → 简单任务解析
  → Streamlit展示
  → MP4/JSON/PNG/CSV导出
```

### 1.2 两周内明确不做

- ROS 2、Nav2和完整Gazebo导航栈；
- 实体机器人控制；
- RGB-D硬件采集；
- TensorRT与Jetson部署；
- 自定义数据标注和训练；
- 独立语义分割模型；
- SLAM和真实米制地图；
- 本地大模型。

### 1.3 第3～4周增强

- 第3周：公开RGB-D数据、稳健深度、三维反投影和误差报告；
- 第4周：优先做轻量Gazebo演示。如果第22天半天内不能稳定启动Ubuntu、Gazebo和官方差速机器人示例，则停止环境排查，改做云端LLM结构化解析。

---

## 2. 目标目录

```text
semanticnav-ai-ros2/
├─ app/
│  └─ streamlit_app.py
├─ src/semanticnav/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ models.py
│  ├─ video.py
│  ├─ tracking.py
│  ├─ rendering.py
│  ├─ serialization.py
│  ├─ depth.py
│  ├─ mapping.py
│  ├─ planning.py
│  ├─ language.py
│  ├─ pipeline.py
│  └─ rgbd.py
├─ configs/
│  └─ default.yaml
├─ tests/
│  ├─ fixtures/
│  ├─ test_models.py
│  ├─ test_video.py
│  ├─ test_tracking.py
│  ├─ test_serialization.py
│  ├─ test_depth.py
│  ├─ test_mapping.py
│  ├─ test_planning.py
│  ├─ test_language.py
│  ├─ test_pipeline.py
│  └─ test_rgbd.py
├─ scripts/
│  ├─ make_test_video.py
│  ├─ run_video.py
│  └─ benchmark.py
├─ assets/samples/
├─ datasets/                 # Git忽略
├─ models/                   # Git忽略
├─ outputs/                  # Git忽略
├─ benchmarks/
├─ docs/
│  ├─ demo-script.md
│  ├─ limitations.md
│  └─ rgbd-report.md
├─ .gitignore
├─ pyproject.toml
├─ requirements.txt
├─ README.md
└─ FAST_TRACK_PLAN.md
```

### 模块职责

| 模块 | 职责 |
|---|---|
| `models.py` | Pydantic数据模型和跨模块字段约定 |
| `video.py` | 视频元数据、逐帧读取和视频写出 |
| `tracking.py` | YOLO与ByteTrack适配，不负责绘图 |
| `rendering.py` | 边界框、轨迹、深度热图和地图绘制 |
| `serialization.py` | JSON、CSV和运行目录 |
| `depth.py` | 单目深度推理、归一化和目标风险等级 |
| `mapping.py` | 图像观测到局部示意栅格的确定性映射 |
| `planning.py` | 障碍膨胀和A* |
| `language.py` | 简单中英文任务解析与安全检查 |
| `pipeline.py` | 编排模块和输出进度，不实现模型细节 |
| `rgbd.py` | 第3周的深度过滤与三维反投影 |
| `streamlit_app.py` | 用户交互，不承载算法实现 |

---

## 3. 固定接口

```python
from typing import Literal
from pydantic import BaseModel, Field

DepthLevel = Literal["near", "mid", "far", "unknown"]

class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class TrackedObject(BaseModel):
    track_id: int
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    depth_level: DepthLevel = "unknown"
    relative_depth: float | None = None

class FrameResult(BaseModel):
    frame_index: int
    timestamp_s: float
    inference_ms: float
    total_ms: float
    objects: list[TrackedObject]

class SemanticTask(BaseModel):
    target: str | None
    avoid_classes: list[str]
    speed_mode: Literal["slow", "normal"]
    clarification_required: bool
    clarification_reason: str | None = None

class PlannedPath(BaseModel):
    cells: list[tuple[int, int]]
    path_length_cells: float
    planning_ms: float
    success: bool
    failure_reason: str | None = None
```

结果目录：

```text
outputs/20260814-143522-a1b2c3/
├─ annotated.mp4
├─ results.json
├─ metrics.csv
├─ semantic_map.png
├─ path.csv
└─ run_metadata.json
```

---

# 第一部分：14天可演示版

## Task 1（第1天）：工程基线与数据模型

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/semanticnav/__init__.py`
- Create: `src/semanticnav/config.py`
- Create: `src/semanticnav/models.py`
- Create: `configs/default.yaml`
- Create: `tests/test_models.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `BBox`、`TrackedObject`、`FrameResult`、`SemanticTask`、`PlannedPath`。
- Produces: `load_config(path: Path) -> AppConfig`。

- [ ] 创建并激活环境：

```powershell
Set-Location "D:\semanticnav-ai-ros2"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

- [ ] 在`tests/test_models.py`先写置信度越界失败测试：

```python
def test_tracked_object_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        TrackedObject(
            track_id=1,
            class_id=0,
            class_name="person",
            confidence=1.2,
            bbox=BBox(x1=0, y1=0, x2=10, y2=10),
        )
```

- [ ] 运行`pytest tests/test_models.py -v`，确认因模块尚未实现而失败。
- [ ] 实现固定接口中的数据模型；配置包含模型名、置信度、输入尺寸、深度帧间隔、地图尺寸和输出目录。
- [ ] 安装并锁定依赖：

```powershell
pip install opencv-python numpy pydantic pyyaml streamlit ultralytics pytest
pip freeze | Set-Content requirements.txt
pip install -e .
pytest tests/test_models.py -v
```

- [ ] `.gitignore`排除`.venv/`、`.env`、`datasets/`、`models/`、`outputs/`、`*.pt`、`*.onnx`、`*.engine`、`*.mp4`、`*.avi`和`*.db3`。
- [ ] 提交：

```powershell
git add .gitignore pyproject.toml requirements.txt src configs tests/test_models.py README.md
git commit -m "chore: initialize fast-track project"
```

**验收：** 模型测试通过；配置中不存在用户绝对路径；虚拟环境未被Git跟踪。

---

## Task 2（第2天）：视频输入与输出

**Files:**
- Create: `src/semanticnav/video.py`
- Create: `scripts/make_test_video.py`
- Create: `tests/test_video.py`

**Interfaces:**
- Produces: `VideoMetadata(width, height, fps, frame_count)`。
- Produces: `read_video(path) -> Iterator[(frame_index, timestamp_s, frame)]`。
- Produces: `open_video_writer(path, metadata) -> cv2.VideoWriter`。

- [ ] `make_test_video.py`生成320×240、10 FPS、10帧测试视频，每帧包含移动矩形和帧号。
- [ ] 先写测试：

```python
def test_read_video_returns_index_timestamp_and_frame(test_video):
    frames = list(read_video(test_video))
    assert len(frames) == 10
    assert frames[0][0] == 0
    assert frames[1][1] == pytest.approx(0.1, abs=0.02)
    assert frames[0][2].shape == (240, 320, 3)
```

- [ ] 运行`pytest tests/test_video.py -v`并确认失败。
- [ ] 实现读取器与writer，显式检查文件存在、capture打开、FPS与宽高有效、writer打开成功，并在结束时释放句柄。
- [ ] 增加损坏文件测试，预期抛出包含“无法打开视频”的`ValueError`。
- [ ] 验证并提交：

```powershell
python scripts/make_test_video.py
pytest tests/test_video.py -v
git add src/semanticnav/video.py scripts/make_test_video.py tests/test_video.py
git commit -m "feat: add validated video io"
```

**验收：** 读取10帧；时间戳单调递增；结果视频可打开；损坏文件返回清晰错误。

---

## Task 3（第3～5天）：YOLO与ByteTrack

**Files:**
- Create: `src/semanticnav/tracking.py`
- Create: `tests/test_tracking.py`
- Create: `scripts/run_video.py`
- Modify: `configs/default.yaml`

**Interfaces:**
- Produces: `YOLOByteTracker.track(frame) -> tuple[list[TrackedObject], inference_ms]`。
- Produces: `reset() -> None`，切换输入视频时清空状态。

- [ ] 使用假结果对象先写转换测试：

```python
def test_convert_result_preserves_track_id(fake_result):
    objects = convert_ultralytics_result(fake_result, names={56: "chair"})
    assert objects[0].track_id == 7
    assert objects[0].class_name == "chair"
    assert objects[0].bbox.x2 == 510.0
```

- [ ] 运行`pytest tests/test_tracking.py -v`并确认失败。
- [ ] 实现结果转换：无boxes返回空列表；无track ID的框不进入跟踪输出；坐标保留浮点值。
- [ ] 每帧跟踪固定调用：

```python
results = self.model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    conf=self.confidence,
    imgsz=self.image_size,
    verbose=False,
)
```

- [ ] 同一视频只创建一次模型与tracker；切换视频时调用`reset()`。
- [ ] 增加无检测结果返回空列表测试。
- [ ] 用30帧视频冒烟测试：

```powershell
python scripts/run_video.py --input assets/samples/room.mp4 --max-frames 30
pytest tests/test_tracking.py tests/test_video.py -v
```

- [ ] 提交：

```powershell
git add src/semanticnav/tracking.py tests/test_tracking.py scripts/run_video.py configs/default.yaml
git commit -m "feat: add yolo bytetrack adapter"
```

**验收：** 连续调用保持跟踪状态；空检测帧不崩溃；每帧输出目标列表和推理耗时。

---

## Task 4（第5天）：标注与轨迹线

**Files:**
- Create: `src/semanticnav/rendering.py`
- Create: `tests/test_rendering.py`

**Interfaces:**
- Produces: `draw_tracks(frame, objects, histories) -> np.ndarray`。

- [ ] 先测试绘制函数不修改输入帧：

```python
def test_draw_tracks_returns_copy(frame, tracked_chair):
    original = frame.copy()
    rendered = draw_tracks(frame, [tracked_chair], {})
    assert np.array_equal(frame, original)
    assert not np.array_equal(rendered, original)
```

- [ ] 标签固定为`chair #7 0.91 near`；track ID映射稳定颜色；每条轨迹最多30个中心点；绘制前裁剪坐标。
- [ ] 运行测试并提交：

```powershell
pytest tests/test_rendering.py -v
git add src/semanticnav/rendering.py tests/test_rendering.py
git commit -m "feat: render tracks and labels"
```

**验收：** 原始帧未被就地修改；越界框不触发异常；ID颜色在视频内稳定。

---

## Task 5（第6天）：JSON、CSV和运行目录

**Files:**
- Create: `src/semanticnav/serialization.py`
- Create: `tests/test_serialization.py`

**Interfaces:**
- Produces: `create_run_directory(root) -> tuple[run_id, run_dir]`。
- Produces: `write_results_json(path, metadata, frames)`。
- Produces: `write_metrics_csv(path, frames)`。

- [ ] 先写空目标序列化测试：

```python
def test_write_results_keeps_empty_objects(tmp_path):
    frame = FrameResult(frame_index=0, timestamp_s=0, inference_ms=10, total_ms=12, objects=[])
    path = tmp_path / "results.json"
    write_results_json(path, {"run_id": "run-1"}, [frame])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["frames"][0]["objects"] == []
```

- [ ] JSON先写同目录临时文件，成功后替换目标；编码UTF-8、两空格缩进。
- [ ] CSV固定字段：`frame_index,timestamp_s,inference_ms,total_ms,object_count`。
- [ ] 同一秒创建两次运行目录时，随机短ID必须不同。
- [ ] 运行测试并提交：

```powershell
pytest tests/test_serialization.py -v
git add src/semanticnav/serialization.py tests/test_serialization.py
git commit -m "feat: serialize run results"
```

**验收：** 每帧有JSON记录；零目标保留空数组；不同运行不会覆盖。

---

## Task 6（第7～8天）：单目相对深度

**Files:**
- Create: `src/semanticnav/depth.py`
- Create: `tests/test_depth.py`
- Modify: `configs/default.yaml`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `RelativeDepthEstimator.infer(frame) -> np.ndarray`，尺寸与原帧一致，范围`[0,1]`，1表示更近。
- Produces: `assign_depth_levels(objects, depth_map, near_threshold, far_threshold)`。

- [ ] 先写常量深度图归一化测试：

```python
def test_normalize_constant_map_is_finite():
    raw = np.ones((4, 4), dtype=np.float32) * 3
    result = normalize_relative_depth(raw)
    assert np.all(result == 0)
    assert np.isfinite(result).all()
```

- [ ] 写中心区域稳健值测试，输入含`NaN`和`inf`，期望使用有限值中位数。
- [ ] 中心区域取目标框中间40%；有效像素少于20%返回`None`。
- [ ] 接入Depth Anything V2 Small，仅用于相对深度；CPU默认每3帧运行一次，中间帧复用最近结果。
- [ ] 权重加载失败时允许关闭深度模块，不得把全零或随机值解释为有效距离。
- [ ] 运行测试与30帧基准：

```powershell
pytest tests/test_depth.py -v
python scripts/run_video.py --input assets/samples/room.mp4 --max-frames 30 --depth
```

- [ ] 提交：

```powershell
git add src/semanticnav/depth.py tests/test_depth.py configs/default.yaml requirements.txt
git commit -m "feat: add relative monocular depth"
```

**验收：** 输出有限的`[0,1]`相对深度；目标获得四级标签；任何界面和JSON均不显示伪造米数。

---

## Task 7（第9～10天）：局部示意语义地图

**Files:**
- Create: `src/semanticnav/mapping.py`
- Create: `tests/test_mapping.py`
- Modify: `src/semanticnav/rendering.py`

**Interfaces:**
- Produces: `image_observation_to_cell(center_x_px, frame_width, depth_level, grid_shape)`。
- Produces: `build_semantic_grid(objects, frame_width, avoid_classes, grid_shape=(30,30))`。
- 代价固定：自由`0`、普通障碍`100`、应避让`200`、不可通行`255`。

- [ ] 先写确定性映射测试：

```python
def test_near_center_maps_near_robot():
    cell = image_observation_to_cell(320, 640, "near", (30, 30))
    assert cell == (25, 15)
```

- [ ] 写未知深度返回`None`测试。
- [ ] 机器人固定在`(29,15)`；near映射到第24～26行，mid到15～17行，far到6～8行；横向映射到2～27列。
- [ ] `person/cat/dog`默认代价200，`chair/sofa/table`默认100，任务中的`avoid_classes`覆盖为200。
- [ ] 绘图包含标题“局部示意地图（非米制）”、机器人、图例和标签。
- [ ] 运行测试并提交：

```powershell
pytest tests/test_mapping.py tests/test_rendering.py -v
git add src/semanticnav/mapping.py src/semanticnav/rendering.py tests/test_mapping.py
git commit -m "feat: build local semantic sketch map"
```

**验收：** 映射确定；近中远次序正确；未知深度不产生假障碍；地图明确标注非米制。

---

## Task 8（第11天）：障碍膨胀与A*

**Files:**
- Create: `src/semanticnav/planning.py`
- Create: `tests/test_planning.py`

**Interfaces:**
- Produces: `inflate_obstacles(grid, radius_cells) -> np.ndarray`。
- Produces: `plan_astar(grid, start, goal, blocked_cost=200) -> PlannedPath`。

- [ ] 写空地图路径测试：

```python
def test_astar_finds_path_in_empty_grid():
    grid = np.zeros((5, 5), dtype=np.uint8)
    result = plan_astar(grid, (4, 2), (0, 2))
    assert result.success is True
    assert result.cells[0] == (4, 2)
    assert result.cells[-1] == (0, 2)
```

- [ ] 写封闭地图返回`failure_reason="no_path"`测试。
- [ ] 使用八邻域；直移代价1、斜移`sqrt(2)`；禁止斜穿阻塞角。
- [ ] 非法起点、非法终点、起点被占据和无路径使用不同失败原因。
- [ ] 膨胀默认半径2格，只膨胀代价大于等于100的格子，且不修改输入数组。
- [ ] 运行测试并提交：

```powershell
pytest tests/test_planning.py -v
git add src/semanticnav/planning.py tests/test_planning.py
git commit -m "feat: add safe astar planning"
```

**验收：** 路径不穿障碍、不切阻塞角；封闭地图返回明确失败；记录路径长度和规划时间。

---

## Task 9（第12天）：自然语言规则解析

**Files:**
- Create: `src/semanticnav/language.py`
- Create: `tests/test_language.py`

**Interfaces:**
- Produces: `parse_task(text, visible_classes=None) -> SemanticTask`。
- 支持：沙发/sofa、椅子/chair、桌子/table、人/person、猫/cat、狗/dog、宠物/pet、鞋/shoe、电线/cable。

- [ ] 写中文解析测试：

```python
def test_parse_chinese_task():
    task = parse_task("去沙发附近，避开人和宠物")
    assert task.target == "sofa"
    assert task.avoid_classes == ["person", "cat", "dog"]
    assert task.clarification_required is False
```

- [ ] 写缺少目标返回`clarification_reason="missing_target"`测试。
- [ ] 使用最长词优先；“宠物”展开为`cat,dog`；“容易缠绕的东西”展开为`cable,shoe`；“慢慢/小心/slow”映射`slow`。
- [ ] 增加20条参数测试：10条中文、4条英文、2条中英混合、2条缺失目标、1条冲突目标、1条空输入。
- [ ] 运行测试并提交：

```powershell
pytest tests/test_language.py -v
git add src/semanticnav/language.py tests/test_language.py
git commit -m "feat: parse constrained navigation tasks"
```

**验收：** 20条测试通过；歧义或缺失目标进入澄清；输出不存在电机、轮速和`/cmd_vel`字段。

---

## Task 10（第13天）：端到端管线

**Files:**
- Create: `src/semanticnav/pipeline.py`
- Create: `tests/test_pipeline.py`
- Modify: `scripts/run_video.py`

**Interfaces:**
- Produces: `VideoPipeline.run(input_path, task_text, output_root, progress=None) -> RunSummary`。
- `RunSummary`包含运行目录、帧数、平均FPS、平均推理延迟、P95总延迟和规划结果。

- [ ] 使用假tracker和假depth先写端到端测试：

```python
def test_pipeline_writes_artifacts(test_video, tmp_path, fake_tracker, fake_depth):
    summary = VideoPipeline(fake_tracker, fake_depth).run(
        test_video, "去椅子附近，避开人", tmp_path
    )
    assert summary.frame_count == 10
    for name in ["annotated.mp4", "results.json", "metrics.csv", "semantic_map.png", "path.csv"]:
        assert (summary.run_dir / name).exists()
```

- [ ] 固定编排顺序：读帧→跟踪→按间隔算深度→深度等级→绘制→写视频→最终地图→A*→序列化。
- [ ] 平均FPS=`处理帧数/总墙钟时间`；P95来自所有`total_ms`；模型加载时间单列。
- [ ] 用户中止时释放reader/writer并写`status=cancelled`；其他异常写`status=failed`后重新抛出。
- [ ] 运行全部测试并提交：

```powershell
pytest -q
git add src/semanticnav/pipeline.py tests/test_pipeline.py scripts/run_video.py
git commit -m "feat: integrate fast-track pipeline"
```

**验收：** 测试视频生成五类产物；失败或中止不会留下无法播放的视频句柄。

---

## Task 11（第13天）：Streamlit界面

**Files:**
- Create: `app/streamlit_app.py`
- Create: `tests/test_app_helpers.py`
- Modify: `requirements.txt`

**Interfaces:**
- 页面只调用`VideoPipeline.run`，不直接实现YOLO、深度或A*。

- [ ] 为上传文件保存、输出清单和指标格式化写纯函数测试；导入页面模块时不得自动加载模型。
- [ ] 侧边栏包含视频、任务、置信度、输入尺寸、深度开关、深度帧间隔和运行按钮。
- [ ] 主区域显示标注视频、深度代表帧、示意地图、路径、任务JSON、平均FPS和P95延迟。
- [ ] 固定显示边界说明：

```text
本页面展示相对深度、局部示意地图和动作建议，尚未接入真实机器人底盘；结果不构成安全控制指令。
```

- [ ] 提供五类结果文件下载。
- [ ] 手动冒烟：

```powershell
streamlit run app/streamlit_app.py
```

- [ ] 测试并提交：

```powershell
pytest tests/test_app_helpers.py -v
git add app/streamlit_app.py tests/test_app_helpers.py requirements.txt
git commit -m "feat: add streamlit demo"
```

**验收：** 页面完成一次运行；错误显示为用户提示；边界说明始终可见；结果可以下载。

---

## Task 12（第14天）：基准、README与演示

**Files:**
- Create: `scripts/benchmark.py`
- Create: `benchmarks/README.md`
- Create: `docs/demo-script.md`
- Create: `docs/limitations.md`
- Modify: `README.md`

- [ ] 运行完整测试，要求0失败：

```powershell
pytest -q
```

- [ ] 对固定100帧分别运行深度关闭和每3帧深度一次，记录模型、帧数、平均FPS、平均/P95延迟与内存峰值。
- [ ] README包含项目定位、功能、安装、启动、CLI、输出格式、测试、性能表、目录、限制与路线图。
- [ ] 按60秒脚本录制Demo：任务→视频→检测跟踪→深度→地图→路径→JSON→性能。
- [ ] 检查大文件与密钥：

```powershell
git status --short
git ls-files | rg "\.(pt|onnx|engine|mp4|avi|db3)$"
rg -n "api[_-]?key|token|secret" . -g "!requirements.txt" -g "!FAST_TRACK_PLAN.md"
```

- [ ] 提交：

```powershell
git add scripts/benchmark.py benchmarks/README.md docs README.md
git commit -m "docs: publish fast-track demo"
```

**验收：** 新环境按README可启动；指标来自真实运行；仓库无密钥和大文件；Demo完整展示闭环。

---

# 第二部分：第3周RGB-D增强

## Task 13（第15～17天）：RGB-D输入与稳健深度

**Files:**
- Create: `src/semanticnav/rgbd.py`
- Create: `tests/test_rgbd.py`
- Create: `tests/fixtures/camera_intrinsics.json`
- Create: `docs/rgbd-report.md`

**Interfaces:**
- Produces: `CameraIntrinsics(fx, fy, cx, cy, width, height)`。
- Produces: `robust_metric_depth(depth_m, bbox, center_fraction=0.4, min_valid_ratio=0.2, min_m=0.1, max_m=10.0)`。

- [ ] 写0、NaN和异常值过滤测试：

```python
def test_robust_depth_filters_invalid_values():
    depth = np.array([[0, 2, 2], [np.nan, 2, 99], [2, 2, 2]], dtype=np.float32)
    sample = robust_metric_depth(depth, BBox(x1=0, y1=0, x2=3, y2=3), center_fraction=1.0)
    assert sample.valid is True
    assert sample.depth_m == pytest.approx(2.0)
```

- [ ] 数据加载器显式接收深度缩放因子；`uint16`乘缩放因子后统一为米，不依据数值猜单位。
- [ ] `rgbd-report.md`记录数据集名称、官方链接、许可、RGB/Depth对齐情况、深度单位、内参来源和样例序列。
- [ ] 测试并提交：

```powershell
pytest tests/test_rgbd.py -v
git add src/semanticnav/rgbd.py tests/test_rgbd.py tests/fixtures/camera_intrinsics.json docs/rgbd-report.md
git commit -m "feat: load and filter public rgbd data"
```

**验收：** 无效值被过滤；有效比例不足时`valid=False`；单位转换有明确依据。

---

## Task 14（第18～19天）：三维反投影

**Files:**
- Modify: `src/semanticnav/rgbd.py`
- Modify: `src/semanticnav/models.py`
- Modify: `tests/test_rgbd.py`
- Modify: `docs/rgbd-report.md`

**Interfaces:**
- Produces: `deproject_pixel(u, v, z_m, intrinsics) -> tuple[x,y,z]`。
- Produces: `localize_object(bbox, depth_m, intrinsics) -> Object3D`。

- [ ] 写主点测试：

```python
def test_principal_point_lies_on_optical_axis():
    intr = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
    assert deproject_pixel(320, 240, 2.0, intr) == pytest.approx((0, 0, 2))
```

- [ ] 用内参将`(0.2,-0.1,2.0)`投影到像素再反投影，三个坐标误差均小于`1e-6 m`。
- [ ] 实现公式：

```text
X=(u-cx)Z/fx
Y=(v-cy)Z/fy
Z=depth(u,v)
distance=sqrt(X²+Y²+Z²)
```

- [ ] 报告样本数、平均绝对误差、相对误差、无效比例和三个失败案例。无真值时只报告重复测量稳定性。
- [ ] 测试并提交：

```powershell
pytest tests/test_rgbd.py -v
git add src/semanticnav/rgbd.py src/semanticnav/models.py tests/test_rgbd.py docs/rgbd-report.md
git commit -m "feat: deproject rgbd objects"
```

**验收：** 合成点误差达标；无效深度不生成坐标；坐标标记为`camera_link`。

---

## Task 15（第20～21天）：RGB-D模式集成

**Files:**
- Modify: `src/semanticnav/pipeline.py`
- Modify: `app/streamlit_app.py`
- Modify: `tests/test_pipeline.py`
- Modify: `README.md`

- [ ] 新增`video_relative`与`public_rgbd`两种模式。
- [ ] 测试普通视频JSON不存在`distance_m`，RGB-D有效深度包含`X/Y/Z`、距离、有效比例和`camera_link`。
- [ ] 页面明确区分“相对深度”和“数据集米制深度”。
- [ ] 运行`pytest -q`并要求0失败。
- [ ] 提交：

```powershell
git add src/semanticnav/pipeline.py app/streamlit_app.py tests/test_pipeline.py README.md
git commit -m "feat: expose public rgbd mode"
```

**验收：** 两种模式边界清晰；普通视频不产生米制距离；RGB-D实验可复现。

---

# 第三部分：第4周作品集增强

## 第22天门禁

用半天验证：Ubuntu 24.04可启动、Gazebo可打开空场景、官方差速机器人示例可移动。

- 三项全部满足：执行Option A。
- 任意一项未满足：记录环境问题并停止继续排查，执行Option B。
- 两个选项只完成一个。

## Option A：Gazebo轻量演示

### 第22～28天任务

- [ ] 记录Ubuntu、Gazebo和机器人示例版本；
- [ ] 创建带沙发、椅子和箱子的简单室内场景；
- [ ] 将A*路径导出为世界坐标waypoint CSV；
- [ ] 编写waypoint跟随器；
- [ ] 实现到达、超时和路径阻塞状态；
- [ ] 运行10次并记录成功、碰撞和平均耗时；
- [ ] 录制仿真绕障Demo；
- [ ] README明确说明尚未完成Nav2语义代价地图插件。

### 验收

- [ ] 一个命令启动场景和跟随器；
- [ ] 固定场景中到达目标且不穿过静态障碍；
- [ ] 有10次运行数据；
- [ ] 不将waypoint跟随描述为完整Nav2导航。

## Option B：LLM结构化解析

### 第22～28天任务

- [ ] 定义与`SemanticTask`完全一致的结构化输出Schema；
- [ ] API密钥只从环境变量读取；
- [ ] 实现10秒超时和一次重试；
- [ ] LLM输出后运行本地Schema与安全验证；
- [ ] 网络失败、非法JSON、未知目标和冲突目标进入澄清状态；
- [ ] 编写30条中英文测试，模型调用使用mock；
- [ ] 页面显示原始指令、模型输出、验证结果和回退原因；
- [ ] 比较规则解析器与LLM在30条指令上的成功率；
- [ ] README说明LLM不生成电机命令。

### 验收

- [ ] 所有成功输出通过Pydantic验证；
- [ ] API密钥不进入日志、JSON和Git；
- [ ] API不可用时规则解析器仍可工作；
- [ ] 输出不存在轮速、转向角和`/cmd_vel`。

---

## 4. 每日检查

每天结束前运行：

```powershell
pytest -q
git status --short
git diff --check
```

每日记录：

```markdown
## YYYY-MM-DD

- 完成内容：
- 测试结果：
- 实测性能：
- 已知问题：
- 明日唯一最高优先级：
```

当天任务未完成时，第二天先完成原任务，不同时展开新模块。

---

## 5. 两周验收清单

- [ ] 能上传和处理本地MP4；
- [ ] YOLO能检测常见预训练类别；
- [ ] ByteTrack输出持续轨迹ID；
- [ ] 显示类别、置信度、框和轨迹；
- [ ] 显示相对深度及四级风险；
- [ ] 生成明确标注非米制的局部示意地图；
- [ ] A*不穿阻塞格或阻塞角；
- [ ] 解析不少于20条简单中英文任务；
- [ ] 导出视频、JSON、指标CSV、地图PNG和路径CSV；
- [ ] 显示平均FPS、平均推理延迟和P95总延迟；
- [ ] 全部自动化测试通过；
- [ ] README能指导新用户启动；
- [ ] Demo说明未接真实底盘和真实RGB-D相机。

## 6. 四周验收清单

- [ ] 两周要求继续满足；
- [ ] RGB-D单位、内参和许可有记录；
- [ ] 深度过滤处理0、NaN、无穷与范围异常；
- [ ] 已知三维点反投影测试通过；
- [ ] RGB-D模式输出相机坐标和真实深度距离；
- [ ] 普通视频仍只输出相对深度；
- [ ] 完成Gazebo或LLM中的一个增强项；
- [ ] 有性能表、RGB-D报告、失败案例和Demo；
- [ ] 仓库没有密钥、大型数据和模型权重。

---

## 7. 算力与费用

| 项目 | 两周版 | 四周版 |
|---|---|---|
| 现有CPU | 足够 | 足够 |
| NVIDIA GPU | 不需要 | 不需要 |
| 云GPU | 不需要 | 仅批量预计算时按小时使用 |
| LLM API | 不需要 | 仅Option B产生少量费用 |
| 新硬件 | 不购买 | 不购买 |

CPU过慢时依次降级：

1. 检测尺寸640降到480；
2. 深度每3帧改为每5帧；
3. 处理期间关闭逐帧网页预览；
4. 对演示视频预计算深度；
5. 最后才短时租用云GPU。

---

## 8. 范围停止线

第28天前不实现：自定义训练、独立分割、ROS 2节点、Nav2插件、Isaac Sim、TensorRT、Jetson、实体机器人、多摄像头、SLAM、账户系统、数据库和云部署。

新想法只记录到README路线图，不插入当前迭代。

---

## 9. 60秒展示脚本

```text
00–05秒：项目名称和自然语言任务
05–15秒：上传室内视频并处理
15–25秒：YOLO检测、ByteTrack ID与轨迹
25–35秒：相对深度和近/中/远风险
35–45秒：局部示意地图与A*路径
45–52秒：结构化任务JSON和动作建议
52–60秒：FPS、延迟、导出文件和系统限制
```

展示文案：

> SemanticNav AI 将自然语言导航任务转换为目标和避障约束，从室内视频中检测并跟踪常见障碍，利用单目相对深度生成局部示意地图，并通过A*规划避障路径。当前版本输出仿真路径和动作建议，未连接真实机器人底盘。

---

## 10. 简历描述

> **SemanticNav AI：室内机器人语义导航软件原型｜独立开发**  
> 构建“自然语言任务解析—YOLO目标检测—ByteTrack多目标跟踪—单目相对深度—局部语义地图—A*避障规划”软件闭环；使用Streamlit展示感知、地图、路径和性能指标，并导出结构化JSON、标注视频及基准结果。基于公开RGB-D数据实现稳健深度过滤与相机坐标反投影，通过合成三维点测试验证定位算法。

只有完成对应模块并取得真实结果后，才在简历中填写FPS、P95延迟、三维定位误差和规划成功率。

---

## 11. 执行顺序

```text
Task 1 工程基线
  → Task 2 视频IO
  → Task 3 YOLO+ByteTrack
  → Task 4 标注
  → Task 5 JSON/CSV
  → Task 6 单目深度
  → Task 7 示意地图
  → Task 8 A*
  → Task 9 任务解析
  → Task 10 管线
  → Task 11 Streamlit
  → Task 12 发布
  → Task 13～15 RGB-D
  → 第22天选择Gazebo或LLM
```

第一交付目标是第14天完成一个稳定、可测试、可导出结果的演示闭环。第14天验收通过后再进入RGB-D和作品集增强。
