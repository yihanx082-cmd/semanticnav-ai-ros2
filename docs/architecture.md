# SemanticNav AI 架构

## 设计目标

系统把室内视频转换为可解释的感知、相对空间和示意规划结果。网页和CLI共享同一套`VideoPipeline`，避免出现“页面一套算法、脚本另一套算法”。

## 数据流

```text
MP4
  → video.read_video
  → YOLOByteTracker.track
  → TrackedObject[]
  → RelativeDepthEstimator（可选）
  → FrameResult[]
  → build_semantic_grid
  → inflate_obstacles
  → plan_astar
  → RunSummary + 运行目录
```

自然语言任务并行进入：

```text
中文/英文文本
  → parse_task
  → SemanticTask
  → 目标选择和avoid_classes代价设置
```

## 核心数据合同

### `TrackedObject`

保存`track_id`、类别、置信度、浮点边界框、相对深度和近/中/远等级。置信度和相对深度由Pydantic限制在`[0, 1]`。

### `FrameResult`

保存帧号、视频时间戳、YOLO推理耗时、完整帧耗时和当帧目标列表。JSON和CSV均从该对象生成。

### `SemanticTask`

保存目标类别、避让类别、速度模式和是否需要澄清。解析器只生成高层约束，不生成电机指令。

### `PlannedPath`

保存路径栅格、路径长度、规划耗时、成功状态和失败原因。失败时不会返回伪造路径。

## 模块职责

| 模块 | 职责 |
|---|---|
| `video.py` | 验证视频参数、逐帧读取、创建writer并释放句柄 |
| `tracking.py` | 加载YOLO、调用ByteTrack、转换Ultralytics结果、重置状态 |
| `rendering.py` | 绘制边界框、标签、稳定ID颜色和最多30点轨迹 |
| `depth.py` | 推理相对深度、归一化、中心区域稳健中位数和等级划分 |
| `mapping.py` | 把图像横坐标与近中远映射到30×30非米制栅格 |
| `planning.py` | 障碍膨胀、8邻域A*和禁止斜穿障碍角点 |
| `language.py` | 解析中英文受限任务、检测缺目标/冲突/危险指令 |
| `serialization.py` | 创建唯一运行目录并写出JSON、CSV和路径 |
| `pipeline.py` | 编排完整运行、性能统计、异常记录和资源释放 |
| `app_support.py` | 上传安全、下载白名单、指标格式和JSON读取 |
| `app/streamlit_app.py` | 页面参数、进度、展示和下载，不复制算法 |

## 坐标和空间假设

- OpenCV图像原点在左上，`x`向右、`y`向下；
- 机器人固定在示意地图底部中央；
- 图像横坐标决定地图左右位置；
- 相对深度等级决定地图纵向位置；
- 未知深度目标不写入虚假障碍位置；
- 地图没有米制分辨率、相机姿态或世界坐标系。

## 状态与资源

- 同一视频使用同一个YOLO/ByteTrack实例和`persist=True`；
- 切换运行后调用`tracker.reset()`；
- 视频reader和writer在正常、取消和异常路径中都会释放；
- JSON采用临时文件替换，避免写到一半留下有效外观的损坏文件；
- 每次运行创建唯一目录，避免覆盖历史结果；
- Streamlit只在点击按钮后加载模型，并通过资源缓存减少重复初始化。

## 错误处理

- 文件不存在：`FileNotFoundError`；
- 视频损坏或参数无效：包含具体原因的`ValueError`；
- writer创建失败：立即停止，不生成空壳成功结果；
- 深度模型不可用：页面显示明确错误；
- 管线异常：`run_metadata.json`记录`failed`和异常类型；
- A*无路：返回`success=False`和`failure_reason`。

## 测试边界

单元测试使用合成视频、假YOLO结果和假深度图，不依赖网络或GPU。真实视频实验单独验证模型加载、ByteTrack状态、原始分辨率写出和CPU性能。
