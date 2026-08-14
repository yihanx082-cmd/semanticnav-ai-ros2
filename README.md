# SemanticNav AI

基于目标检测、跟踪、相对深度和路径规划的室内机器人语义导航软件原型。

## 当前状态

正在实现快速版工程基线。

## 环境

- Windows 11
- Python 3.11

## 安装

```powershell
conda create --prefix .\.venv python=3.11 pip -y
conda activate .\.venv
pip install -r requirements.txt
pip install -e .
```

## 测试

```powershell
pytest -q
```

## 项目边界

当前版本输出相对深度、局部示意地图和动作建议，尚未接入真实机器人底盘。
