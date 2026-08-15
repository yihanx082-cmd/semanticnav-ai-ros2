# 多场景评测视频来源

评测日期：2026-08-15。

视频本体存放在`assets/samples/`并由`.gitignore`排除，仓库只提交来源和机器可读结果。公开视频均来自Pexels。Pexels许可页说明照片和视频可以免费使用、允许修改且通常不要求署名；本项目仍保留作者信息以便审计。

- 许可：[Pexels License](https://www.pexels.com/license/)
- 下载日期：2026-08-15
- 用途：本地非商业算法评测和作品集演示

## scene_room.mp4

- 本地文件：`room.mp4`，评测时映射为`scene_room.mp4`；
- 来源：用户自行拍摄并明确提供给本项目使用；
- 分辨率：540×960；
- 帧率：30 FPS；
- 总帧数：860；
- 评测片段：第270～299帧；
- 主要内容：椅子、沙发和多个装饰摆件。

## scene_person.mp4

- 本地文件：`scene_person.mp4`；
- 标题：Woman Walking inside the House；
- 作者：Yaroslav Shuraev；
- 页面：[Pexels视频5085422](https://www.pexels.com/video/woman-walking-inside-the-house-5085422/)；
- 页面状态：Free download、Free to use；
- 分辨率：4096×2160；
- 帧率：24 FPS；
- 总帧数：364；
- 评测片段：第60～89帧；
- 主要内容：一名人员在室内厨房侧向移动，背景包含瓶子和厨房设施。

第0～29帧没有人员进入画面，因此第一次运行被排除，不纳入最终三场景表。保留这一选择过程用于证明评测片段不是随意凑数。

## scene_pet_or_clutter.mp4

- 本地文件：`scene_pet_or_clutter.mp4`；
- 标题：Cat Inside a House；
- 作者：ROMAN ODINTSOV；
- 页面：[Pexels视频7592804](https://www.pexels.com/video/cat-inside-a-house-7592804/)；
- 页面状态：Free download、Free to use；
- 分辨率：2160×3744；
- 帧率：29.97 FPS；
- 总帧数：611；
- 评测片段：第0～29帧；
- 主要内容：室内橘猫、烤箱和绿色厨房柜体。

## 统一参数

```text
model=yolo26n.pt
tracker=bytetrack.yaml
image_size=480
confidence=0.25
depth=off
frames=30
```

场景统计可使用：

```powershell
python scripts\evaluate_scene.py `
  --scene person `
  --results outputs\evaluation\<run-id>\results.json `
  --metadata outputs\evaluation\<run-id>\run_metadata.json
```
