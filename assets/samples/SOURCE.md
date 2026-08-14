# Sample video source

- Local sample: `room.mp4`（1280×720、24 FPS，不提交到 Git）
- Original download: `room_original_4k.mp4`（4096×2160，不提交到 Git）
- Title: Woman Walking inside the House
- Source: Pexels
- Original page: https://www.pexels.com/video/woman-walking-inside-the-house-5085422/
- Creator: Yaroslav Shuraev
- License: https://www.pexels.com/legal-pages/license/
- Download date: 2026-08-14

## Download requirements

从原始页面下载 MP4。为了降低 CPU 推理负担，将工作副本缩放并保存为：

```text
assets/samples/room.mp4
```

当前样例时长约 15 秒。原始视频仅用于本地功能验证；仓库的
`.gitignore` 已排除 `*.mp4`，避免把素材或大文件提交到 GitHub。
