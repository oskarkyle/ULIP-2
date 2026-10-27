# ULIP-2

ULIP-2 代码、环境安装脚本与实验交接说明。

- `repo/`：上游 ULIP 模型、训练与评估代码，许可证见 `repo/LICENSE.txt`。
- `setup/`：安装、兼容性补丁、smoke 检查及来源记录。
- `scripts/human_ground/`：人体加地面实验的历史脚本快照。
- `docs/`：训练历程与交接文档。

本仓库不包含 outputs、模型权重、数据集、Python 环境、缓存、编译产物和大型演示 GIF。运行前需要自行准备模型和数据，安装说明见 `setup/INSTALL_NOTES.txt`。

人体加地面脚本保留历史路径与依赖，尚未改造成独立 runner；旧 runner 与实验源数据已在本地 outputs 清理时删除。复制出的脚本用于保留实现，不能直接独立重跑。交接文档中的历史 outputs 链接也不包含在本代码仓库中。安装脚本含原机器绝对路径，迁移后需要调整。
