# SemanticNav AI 性能基准

## 环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11，10.0.26200 |
| CPU | Intel Core Ultra 5 125H |
| 内存 | 31.5 GB |
| CUDA | 不可用 |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cpu |
| OpenCV | 5.0.0 |
| Ultralytics | 8.4.120 |
| Transformers | 5.15.0 |
| 检测模型 | `yolo26n.pt` |
| 深度模型 | `depth-anything/Depth-Anything-V2-Small-hf` |

## 输入

- 视频：用户自行拍摄的室内竖屏MP4；
- 原始分辨率：540×960；
- 原始帧率：30 FPS；
- 实验起点：第270帧；
- E1～E3各处理100帧；
- E4处理20帧，并每5帧执行一次相对深度推理。

视频文件受`.gitignore`保护，不提交到仓库。来源记录见
[`assets/samples/SOURCE.md`](../assets/samples/SOURCE.md)。

## 结果

测量日期：2026-08-15。

| 场景 | 输入尺寸 | 置信度 | 深度 | 帧数 | 平均FPS | YOLO平均推理/ms | P95总延迟/ms | 路径 |
|---|---:|---:|---|---:|---:|---:|---:|---|
| E1 | 480 | 0.25 | 关闭 | 100 | 4.076 | 180.810 | 411.523 | 成功 |
| E2 | 640 | 0.25 | 关闭 | 100 | 2.519 | 319.755 | 645.510 | 成功 |
| E3 | 480 | 0.50 | 关闭 | 100 | 3.489 | 218.376 | 471.430 | 成功 |
| E4 | 480 | 0.25 | 每5帧一次 | 20 | 0.532 | 256.040 | 7995.077 | 成功 |

原始机器可读结果位于[`latest_results.csv`](latest_results.csv)。

## 结论

- 480输入的E1比640输入的E2快约61.8%；
- 480输入使YOLO平均推理延迟比640下降约43.5%；
- 置信度从0.25提高到0.50没有带来速度提升，且会增加漏检风险；
- CPU上的Depth Anything V2是主要瓶颈，开启后平均FPS从4.076降至0.532；
- 默认交互配置应使用480、置信度0.25和关闭深度；
- 展示完整闭环时使用短片段，并将深度间隔设为5。

FPS包含视频读取、检测、跟踪、绘制、结果收集和规划前处理，不等于单独YOLO算子的理论FPS。P95总延迟用于观察慢帧，不应被平均推理延迟替代。

## 复现

```powershell
Set-Location "D:\semanticnav-ai-ros2"
conda activate .\.venv
$env:HF_HUB_DISABLE_XET="1"

python scripts\benchmark.py `
  --input assets\samples\room.mp4 `
  --start-frame 270 `
  --max-frames 100 `
  --output benchmarks\latest_results.csv `
  --output-root outputs\benchmarks
```

脚本固定执行E1～E4并从`RunSummary`提取指标，不允许手工填写CSV。
