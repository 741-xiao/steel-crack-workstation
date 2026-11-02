# Steel Crack Workstation · 钢结构裂缝离线检测工作站

版本 **2.1.0**。基于 Ultralytics YOLO 的本地图片/视频裂缝辅助筛查软件，提供批次检测、结果追溯、人工复核和待标注样本回收。支持 Windows 桌面界面及命令行，暂不对接 PLC/MES。

**检测结果需要人工复核，未检出不代表工件合格。** 本项目没有证明新的检测算法或现场工业验收性能。

## 功能

- 检测单张图片、视频及文件夹；支持 ROI、视频抽帧和停止任务。
- SQLite 保存批次、模型 SHA-256、参数、运行环境、源文件信息和检测结果。
- 保存原图/标注图，按批次与复核状态查询，导出 CSV。
- 人工确认有裂缝、无裂缝或待复查，保留复核人、依据和历次修改；拒绝旧版本覆盖新意见。
- 导出包含原图、预测与复核历史的 ZIP，作为后续人工标注资料；不会自动生成正确训练标签。

## 安装与启动（Windows / PowerShell）

建议使用 Python 3.11–3.13。本机验证环境为 Windows、Python 3.13 和 CPU；其他电脑仍需安装与运行验证。

```powershell
git clone https://github.com/741-xiao/steel-crack-workstation.git
cd steel-crack-workstation
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

安装依赖需要联网。首次使用前，准备**可信且有权使用的裂缝模型**，编辑 `inspection_config.json`：

- `weights`：填写自己的模型文件路径；Windows 路径在 JSON 中可写作 `C:/models/best.pt`。
- `output`：结果保存目录，默认 `inspection_data`。
- `device`：普通电脑可保持 `cpu`。

**仓库没有附带权重和数据集。** 默认权重路径指向原开发电脑的训练输出，克隆后需自行配置。当前适配单一 `crack` 类别的 YOLO 检测/分割模型；普通通用目标检测权重不能直接替代裂缝模型。只加载可信来源的 `.pt` 文件。

```powershell
.\.venv\Scripts\python.exe app.py
```

选择模型和图片/视频或文件夹，填写批次并开始检测；检测完成后可点击历史记录，填写人工复核，再导出记录或样本。

## 命令行

配置好本地权重后：

```powershell
.\.venv\Scripts\python.exe industrial.py inspect --source "C:/inspection-input" --batch "工件001"
.\.venv\Scripts\python.exe industrial.py history
.\.venv\Scripts\python.exe industrial.py export --file "检测记录.csv"
.\.venv\Scripts\python.exe industrial.py export-samples --file "复核样本.zip"
```

输入目录和输出目录不能相互包含。导出复核样本前需在界面中保存复核；样本包仍需人工补画、修正标注后才能用于训练。

## 文档

- [软著申报准备目录](软著申报准备/00_从这里开始.md)：工业裂缝检测离线版 V1.0 的独立业务源码、设计说明、操作手册与截图清单；个人资料只保留本地。
- [新手使用与模型训练指南](新手使用与模型训练指南.md)：指标、数据、训练和更换模型。
- [工业试用说明](工业试用说明.md)：配置、运行、备份和局限。
- [软著准备与功能原创性说明](软著准备与功能原创性说明.md)：项目实现、第三方边界、复核流程与软著准备。
- [项目汇总](项目汇总.md)：早期训练历史，不能当作当前运行状态。

文档中部分路径和训练指标来自原开发环境，不代表克隆仓库后已拥有这些模型、数据或评测结果。数据处理/训练脚本依赖自行准备的数据目录，详细格式见训练指南。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest tests -q
```

GUI 测试默认跳过；设置 `RUN_GUI_TESTS=1` 且配置可用的 Qt 插件环境后可执行。2026-11-02 原开发环境执行完整测试为 28 项通过，包含无屏幕 GUI 测试。测试验证业务逻辑与存储行为，不代表模型准确率、现场可靠性或跨电脑兼容性。

## 许可

项目有权授权的源码与文档采用 [MIT License](LICENSE)，署名 `741-xiao and contributors`。第三方依赖、模型、数据各自保留其许可，详见 [第三方声明](THIRD_PARTY_NOTICES.md)。

特别注意 Ultralytics 的 AGPL-3.0 和 PyQt5 的 GPL v3 / 商业许可要求：**MIT 不消除组合应用的第三方许可义务**。发布物不包含权重、数据集和检测记录。
