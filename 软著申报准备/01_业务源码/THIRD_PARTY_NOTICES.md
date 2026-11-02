# 第三方组件与许可边界

根目录的 [MIT 协议](LICENSE) 是本项目贡献者对其有权授权的代码及文档给出的许可，不改变第三方软件、模型、数据或其他既有作品的许可证，也不是对全部代码来源或权属的法律保证。后续引入第三方代码时须保留其版权、许可及来源说明。

## 主要依赖

| 组件 | 用途 | 许可说明及上游 |
| --- | --- | --- |
| Ultralytics | YOLO 模型加载、推理、训练 | 本机 8.4.116 元数据为 AGPL-3.0；见 [上游许可](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 和 [许可选项](https://www.ultralytics.com/license) |
| PyQt5 | 桌面图形界面 | 本机 5.15.11 元数据为 GPL v3；另有商业许可，见 [Riverbank 许可说明](https://www.riverbankcomputing.com/software/pyqt/intro) |
| PyTorch | 张量与模型运行 | 以所用发行版的 [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) 及第三方声明为准 |
| OpenCV / opencv-python | 图像与视频处理 | OpenCV 与 Python wheel 内含组件分别保留许可；见 [OpenCV](https://github.com/opencv/opencv/blob/4.x/LICENSE) 和 [opencv-python](https://github.com/opencv/opencv-python) |
| NumPy | 数组计算 | BSD-3-Clause，见 [上游许可](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| PyYAML | 训练数据配置读取 | MIT，见 [上游许可](https://github.com/yaml/pyyaml/blob/main/LICENSE) |

Qt、编解码器等传递依赖也可能有独立许可；本表不是完整的软件物料清单。项目依赖版本范围允许升级，部署时应核对实际安装版本。

## 使用与分发

MIT 允许使用本项目有权授权的部分，但**不能据此把包含 AGPL/GPL 组件的组合应用整体当作仅受 MIT 约束的闭源产品分发**。分发或提供网络服务时，应根据实际组合、部署方式、各许可证条件或已取得的商业授权履行义务。软件著作权登记不替代这些授权。

此仓库不随附第三方库源码或二进制包；安装依赖时仍受其许可约束。SE/CBAM 教学示例使用已有方法，当前训练没有接入，不表示原创算法或新的 YOLO 架构。

## 数据集与模型权重

公开仓库不包含原始数据集、整理后的数据、预训练/微调权重、训练输出和实际检测记录。它们的来源授权链尚未完整确认，不能依据本项目 MIT 协议自行取得再分发权。

使用者应自行准备有权使用的裂缝模型及数据，并遵守对应条款。配置中保留的本地模型路径仅用于说明，不表示该权重已在 GitHub 发布。

## 开发记录与软著

本项目包含 AI 辅助开发内容。软著材料应如实记录实际人工设计、修改、测试和开发过程，不能把第三方算法或组件作为自创成果。MIT 开源授权与软件著作权登记是不同事项，许可发布不等于确认权属或保证登记通过。
