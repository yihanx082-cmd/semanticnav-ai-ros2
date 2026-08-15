# SemanticNav AI 作品集版 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有Task 1～10命令行原型上，用7天完成一个可上传视频、展示完整闭环、复现实验并适合录制简历Demo的Streamlit作品集版本。

**Architecture:** 保留现有`VideoPipeline`作为唯一端到端业务入口。Streamlit只收集参数、调用管线和展示运行目录中的产物，不复制YOLO、深度、地图或A*逻辑，因此CLI与网页共享同一套算法和数据格式。

**Tech Stack:** Python 3.11、Streamlit、OpenCV、Ultralytics YOLO、ByteTrack、Depth Anything V2、NumPy、Pydantic、pytest。

## Global Constraints

- 使用现有Windows 11、Intel Core Ultra 5 125H和32 GB内存；本阶段不购买硬件。
- 默认使用`yolo26n.pt`、`imgsz=480`和`confidence=0.25`。
- 单目深度只能称为“相对深度”或“近/中/远”，不得输出米制距离。
- 地图必须标注“局部示意地图（非米制）”，A*结果称为“示意路径”或“动作建议”。
- 页面不得控制电机、底盘或ROS 2节点。
- 模型、视频、数据集、运行结果和密钥不得提交到Git。
- Streamlit页面只调用`VideoPipeline.run(...)`，不得复制感知与规划实现。
- 每个任务测试先行；相关测试和`pytest -q`通过后再单独提交。
- RGB-D、ROS 2、Gazebo和LLM不进入本轮实现。

---

## 1. 当前基线

已经完成：MP4读写、YOLO、ByteTrack、轨迹绘制、JSON/CSV、相对深度、30×30示意地图、障碍膨胀、A*、中英文规则解析、端到端管线、76项测试和4组真实视频实验。

剩余关键缺口：

1. 没有可交互网页；
2. 深度结果没有独立预览图；
3. 基准实验没有一键复现脚本；
4. README状态过期；
5. 没有正式Demo、统一多场景评测和失败案例页。

## 2. 七天安排

| 日期 | 工作 | 当天验收 | 你的参与 |
|---|---|---|---|
| 第1天 | 深度预览和CPU默认参数 | 生成`depth_preview.png` | 检查图像是否易懂，20分钟 |
| 第2天 | 上传、指标和下载辅助函数 | 安全测试通过 | 无 |
| 第3天 | Streamlit主页面 | 浏览器完成30帧处理 | 亲自运行，30分钟 |
| 第4天 | 错误提示、进度和下载 | 错误文件不会导致白屏 | 错误场景测试，20分钟 |
| 第5天 | 一键基准和三场景评测 | 生成真实性能表 | 准备两段补充视频，40分钟 |
| 第6天 | README、架构和限制文档 | 新用户按文档可启动 | 解释系统边界，30分钟 |
| 第7天 | 发布检查和60秒录屏 | 测试0失败并完成Demo | 录屏讲解，60分钟 |

预计你需要参与约3小时；编码、测试和文档整理可由Codex完成。

---

### Task 1：补齐网页需要的管线产物

**Files:**
- Modify: `configs/default.yaml`
- Modify: `src/semanticnav/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `VideoPipeline.run(...) -> RunSummary`。
- Produces: 深度成功推理后写入`depth_preview.png`。
- Produces: CPU默认值`model.image_size=480`、`depth.frame_interval=5`。

- [ ] **Step 1: 写入深度预览失败测试**

```python
assert (summary.run_dir / "depth_preview.png").exists()
preview = cv2.imread(str(summary.run_dir / "depth_preview.png"))
assert preview is not None
assert preview.shape[:2] == (240, 320)
```

- [ ] **Step 2: 增加无深度时不生成假图的测试**

```python
def test_pipeline_without_depth_does_not_write_preview(
    test_video: Path,
    tmp_path: Path,
) -> None:
    summary = VideoPipeline(FakeTracker(), None).run(
        test_video,
        "去椅子附近",
        tmp_path / "outputs",
    )
    assert not (summary.run_dir / "depth_preview.png").exists()
```

- [ ] **Step 3: 确认测试先失败**

```powershell
pytest tests\test_pipeline.py -v
```

- [ ] **Step 4: 写出归一化深度色图**

```python
normalized = np.clip(latest_depth, 0.0, 1.0)
gray = (normalized * 255.0).astype(np.uint8)
preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
if not cv2.imwrite(str(run_dir / "depth_preview.png"), preview):
    raise ValueError("无法写入深度预览图")
```

- [ ] **Step 5: 更新默认配置并验证**

```yaml
model:
  image_size: 480
depth:
  frame_interval: 5
```

```powershell
pytest tests\test_pipeline.py tests\test_depth.py -v
pytest -q
git add configs\default.yaml src\semanticnav\pipeline.py tests\test_pipeline.py
git commit -m "feat: export relative depth preview"
```

**验收：** 深度开启时有可打开的预览图；关闭时不生成假图；全量测试通过。

---

### Task 2：实现Streamlit辅助函数和上传安全

**Files:**
- Create: `src/semanticnav/app_support.py`
- Create: `tests/test_app_support.py`

**Interfaces:**
- Produces: `save_uploaded_video(filename: str, data: bytes, upload_root: Path) -> Path`。
- Produces: `collect_artifacts(run_dir: Path) -> dict[str, Path]`。
- Produces: `format_metrics(summary: RunSummary) -> dict[str, str]`。
- Produces: `load_json(path: Path) -> dict[str, object]`。

- [ ] **Step 1: 测试路径清理、MP4限制和空文件**

```python
def test_save_uploaded_video_strips_parent_directories(tmp_path: Path) -> None:
    saved = save_uploaded_video("../room.mp4", b"video", tmp_path)
    assert saved.parent == tmp_path
    assert saved.read_bytes() == b"video"


@pytest.mark.parametrize("name", ["room.avi", "room.txt", "room"])
def test_save_uploaded_video_rejects_non_mp4(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MP4"):
        save_uploaded_video(name, b"video", tmp_path)
```

- [ ] **Step 2: 测试输出白名单**

```python
def test_collect_artifacts_returns_only_existing_files(tmp_path: Path) -> None:
    (tmp_path / "results.json").write_text("{}", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("x", encoding="utf-8")
    artifacts = collect_artifacts(tmp_path)
    assert set(artifacts) == {"results.json"}
```

- [ ] **Step 3: 确认测试先失败**

```powershell
pytest tests\test_app_support.py -v
```

- [ ] **Step 4: 实现最小纯函数**

要求：只接受非空MP4；使用`Path(filename).name`；文件名增加随机后缀；下载仅允许七个已知产物；JSON错误包含文件名；指标显示两位小数。

- [ ] **Step 5: 验证并提交**

```powershell
pytest tests\test_app_support.py -v
pytest -q
git add src\semanticnav\app_support.py tests\test_app_support.py
git commit -m "feat: add streamlit support helpers"
```

**验收：** 上传路径不能逃出指定目录；无效文件被拒绝；输出清单不暴露其他文件。

---

### Task 3：实现Streamlit主页面

**Files:**
- Create: `app/streamlit_app.py`
- Create: `tests/test_streamlit_app.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `load_config(Path("configs/default.yaml")) -> AppConfig`。
- Consumes: `VideoPipeline.run(...) -> RunSummary`。
- Consumes: Task 2的上传、指标、JSON和输出函数。
- Produces: `main() -> None`，只在`if __name__ == "__main__"`中调用。

- [ ] **Step 1: 测试导入页面不会加载YOLO**

```python
def test_importing_streamlit_app_does_not_load_yolo(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "semanticnav.tracking.YOLOByteTracker.__init__",
        lambda *args, **kwargs: calls.append(1),
    )
    importlib.import_module("app.streamlit_app")
    assert calls == []
```

- [ ] **Step 2: 确认测试先失败**

```powershell
pytest tests\test_streamlit_app.py -v
```

- [ ] **Step 3: 创建侧边栏**

必须包含MP4上传、任务文本、置信度`0.10～0.90`、输入尺寸`320/480/640`、相对深度开关、深度间隔`1～10`、最大帧数（默认30）和运行按钮。

- [ ] **Step 4: 创建主展示区**

必须显示处理进度、标注视频、相对深度预览、示意地图、路径状态、任务JSON、平均FPS、平均推理、P95延迟和现存产物下载按钮。

- [ ] **Step 5: 固定显示系统边界**

```text
本页面展示相对深度、局部示意地图和动作建议，尚未接入真实机器人底盘；结果不构成安全控制指令。
```

- [ ] **Step 6: 将错误显示为用户提示**

失败时调用`st.error(str(error))`并在折叠区显示异常类型；不得生成成功指标或虚假下载链接。

- [ ] **Step 7: 自动测试和浏览器冒烟**

```powershell
pytest tests\test_streamlit_app.py tests\test_app_support.py -v
pytest -q
streamlit run app\streamlit_app.py
```

冒烟参数：现有`room.mp4`、任务“去椅子附近，避开人和宠物”、置信度0.25、尺寸480、深度关闭、30帧。

- [ ] **Step 8: 提交**

```powershell
git add app\streamlit_app.py tests\test_streamlit_app.py requirements.txt
git commit -m "feat: add streamlit portfolio demo"
```

**验收：** 浏览器完成一次处理和下载；导入页面不加载模型；边界说明始终可见。

---

### Task 4：建立一键性能基准

**Files:**
- Create: `scripts/benchmark.py`
- Create: `tests/test_benchmark.py`
- Create: `benchmarks/README.md`
- Create: `benchmarks/latest_results.csv`

**Interfaces:**
- Produces: `summarize_run(metadata: dict[str, object]) -> dict[str, object]`。
- Produces: 参数`--input`、`--start-frame`、`--max-frames`、`--output`。
- Produces: 四个场景组成的文本CSV结果。

- [ ] **Step 1: 测试指标只来自运行元数据**

```python
def test_summarize_run_preserves_measured_metrics() -> None:
    row = summarize_run({
        "run_id": "run-1",
        "frame_count": 30,
        "average_fps": 4.221,
        "average_inference_ms": 148.218,
        "p95_total_ms": 317.615,
    })
    assert row["frame_count"] == 30
    assert row["average_fps"] == pytest.approx(4.221)
```

- [ ] **Step 2: 确认失败后实现脚本**

```powershell
pytest tests\test_benchmark.py -v
```

固定场景：

```text
E1：480、conf=0.25、depth=off、100帧
E2：640、conf=0.25、depth=off、100帧
E3：480、conf=0.50、depth=off、100帧
E4：480、conf=0.25、depth=on、每5帧一次、20帧
```

- [ ] **Step 3: 运行并记录环境**

```powershell
python scripts\benchmark.py `
  --input assets\samples\room.mp4 `
  --start-frame 270 `
  --max-frames 100 `
  --output benchmarks\latest_results.csv
```

`benchmarks/README.md`记录CPU、内存、CUDA状态、Python、PyTorch、Ultralytics、模型、片段和复现命令。

- [ ] **Step 4: 验证并提交**

```powershell
pytest tests\test_benchmark.py -v
pytest -q
git add scripts\benchmark.py tests\test_benchmark.py benchmarks\README.md benchmarks\latest_results.csv
git commit -m "perf: add reproducible video benchmark"
```

**验收：** 一条命令重建四行性能表；数值全部来自实际运行。

---

### Task 5：完成多场景评测和失败案例

**Files:**
- Create: `assets/samples/EVALUATION.md`
- Create: `docs/failure-cases.md`
- Create: `benchmarks/scene_results.csv`

**Interfaces:**
- Consumes: 三段本地MP4，不提交视频本体。
- Produces: 视频来源、参数、检测类别、ID切换、误检、FPS和路径状态。

- [ ] **Step 1: 准备三段互补视频**

```text
scene_room.mp4：已有椅子和沙发场景
scene_person.mp4：室内人员横向走动
scene_pet_or_clutter.mp4：宠物或杂物场景
```

若使用素材网站，在`EVALUATION.md`记录页面、作者、许可证和下载日期。

- [ ] **Step 2: 用统一参数处理**

统一使用`imgsz=480`、`confidence=0.25`、`depth=off`和30帧。

```powershell
python scripts\run_video.py --input assets\samples\scene_room.mp4 --max-frames 30 --image-size 480 --confidence 0.25
python scripts\run_video.py --input assets\samples\scene_person.mp4 --max-frames 30 --image-size 480 --confidence 0.25
python scripts\run_video.py --input assets\samples\scene_pet_or_clutter.mp4 --max-frames 30 --image-size 480 --confidence 0.25
```

- [ ] **Step 3: 填写统一字段**

```text
scene,frames,expected_classes,detected_classes,id_switches,false_positives,average_fps,path_success,notes
```

- [ ] **Step 4: 解释四类失败**

必须包含COCO没有拖鞋/电线类别、相似纹理误检、遮挡导致ID变化、相对深度不能替代米制深度。每项写“现象、原因、当前处理、未来改进”。

- [ ] **Step 5: 提交文本结果**

```powershell
git add assets\samples\EVALUATION.md docs\failure-cases.md benchmarks\scene_results.csv
git commit -m "docs: add multi-scene failure analysis"
```

**验收：** 三类场景有真实结果；至少四类失败有证据和解释；视频不进入Git。

---

### Task 6：README、演示脚本和发布验收

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Create: `docs/demo-script.md`
- Create: `docs/architecture.md`
- Create: `docs/limitations.md`

**Interfaces:**
- Produces: 可复制的安装、测试、网页和CLI命令。
- Produces: 60秒录屏脚本、架构说明、限制与路线图。

- [ ] **Step 1: 更新README**

顺序固定为：定位、Demo、功能、数据流、安装、Streamlit、CLI、输出、性能、测试、限制、路线图。把“正在实现快速版工程基线”改为当前实际状态。

- [ ] **Step 2: 编写架构和限制**

说明`视频 → YOLO/ByteTrack → 相对深度 → 示意地图 → A* → 导出`，并解释`TrackedObject`、`FrameResult`、`SemanticTask`和`PlannedPath`。限制文档必须明确非米制、非实时、非机器人控制和COCO类别边界。

- [ ] **Step 3: 编写60秒演示脚本**

```text
0～8秒：输入任务
8～20秒：上传视频并启动
20～32秒：检测、轨迹ID和相对深度
32～43秒：示意地图和A*路径
43～52秒：JSON、CSV、FPS和延迟
52～60秒：系统边界和RGB-D路线图
```

- [ ] **Step 4: 干净环境验收**

先将`.release-venv/`加入`.gitignore`，再运行：

```powershell
conda create --prefix .\.release-venv python=3.11 pip -y
conda activate .\.release-venv
pip install -r requirements.txt
pip install -e .
pytest -q
streamlit run app\streamlit_app.py
```

- [ ] **Step 5: 检查大文件和密钥**

```powershell
git status --short
git ls-files | rg "\.(pt|pth|onnx|engine|mp4|avi|mov|db3)$"
rg -n "api[_-]?key|token|secret" . -g "!requirements.txt" -g "!NEXT_PHASE_PLAN.md"
```

- [ ] **Step 6: 最终测试和提交**

```powershell
pytest -q
git diff --check
git add README.md docs .gitignore
git commit -m "docs: publish semanticnav portfolio demo"
```

- [ ] **Step 7: 用户录制演示**

实际展示一次成功运行。等待过程可以加速，但画面必须标注“加速播放”，不得伪装实时速度。

**验收：** 新环境可启动；全量测试0失败；README与命令一致；Demo展示输入、感知、地图、路径、导出和边界。

---

## 3. 本阶段完成定义

- [ ] 浏览器上传MP4并填写中英文任务；
- [ ] 显示检测跟踪视频、相对深度、示意地图和路径；
- [ ] 显示平均FPS、平均推理和P95延迟；
- [ ] 下载全部现存产物；
- [ ] 三段室内视频有统一评测；
- [ ] 至少四类失败案例；
- [ ] 全量测试0失败；
- [ ] 新环境按README可启动；
- [ ] 完成30～60秒Demo；
- [ ] Git不包含模型、视频、输出和密钥。

## 4. 采购与算力

本阶段新增费用为`0元`，不购买独立显卡、RGB-D相机、Jetson、底盘或激光雷达。深度太慢时使用短片段、每5帧推理一次或预计算结果。

## 5. 下一道门禁

作品集版完成后只选择一个方向：

1. **首选：公开RGB-D数据和三维反投影。** 增加相机内参、稳健深度、`X/Y/Z`和合成点误差测试，不买相机。
2. **岗位需要机器人导航时：Gazebo/ROS 2。** 使用WSL2或Ubuntu，不在Windows原生环境硬装完整ROS 2链路。
3. **岗位偏AI应用时：LLM结构化解析。** 替换规则解析器，但大模型不得生成电机控制命令。

进入下一阶段前，你需要能够解释轨迹ID、相对深度、非米制地图、A*和主要失败原因。

## 6. 简历阶段性描述

> 构建室内机器人语义导航软件原型，实现YOLO目标检测、ByteTrack多目标跟踪、单目相对深度、局部语义栅格地图与A*避障规划；通过Streamlit集成自然语言任务输入、感知可视化、性能指标和结构化结果导出，并在多场景视频上评估ID稳定性、误检及CPU推理性能。

简历数字必须来自最新基准，不填写未测得的导航成功率、三维定位误差或TensorRT加速比。
