# ULIP-2 姿态训练历程与 Agent 交接

更新日期：2026-10-27。用途：让下一位 Agent 接续实验，不重复踩坑。

本文根据对话与现存训练报告整理；数值结果已读取报告核对。服务器进程、GPU 占用和补采进度本次未实时检查，不能将下文历史状态当成当前状态。本次只新增交接文档，没有启动训练、采集或修改实验数据。

## 1. 当前目标与核心结论

目标是将人体姿态点云映射到语义空间，最终扩展到带背景的场景输入。当前主要训练方式是冻结点云、文本及可选图像编码器，只训练点云投影矩阵，不是端到端微调整个 ULIP-2。

六类固定顺序：

```text
standing
crouching_shallow
crouching_deep
kneeling
lying
supported
```

已走过：冻结 OpenCLIP 诊断 → 纯人体点云与文本对齐 → 图像辅助监督 → 监督对比 → 人体加人工地面 → BLIP-2 文本小试。

当前结论：

- OpenCLIP 人物图像能帮助区分人和非人，但零样本细粒度姿态分类不够可靠。
- SupCon 在仿真纯人体测试集上有所提升，但没有解决真实深蹲问题，也不是所有条件下都更好。
- 人体加地面、下沉、裁剪的联合处理改善了真实深蹲诊断结果，但浅蹲有所退步，不能归因于地面这一个因素。
- BLIP-2 小试存在空输出、泛化描述和错误姿态描述，用户决定继续使用固定文本。
- 下一条主要路线是同次采集配对的单人＋完整场景点云、人物 RGB 与固定文本；不得把独立采集的纯人体点云随意配到场景图片上。

## 2. 路径约定和先读文件

下文 `OUTPUTS` 指：

```text
/media/robosense/DataCenterNew/Kyle/ULIP-2/outputs
```

建议按顺序阅读：

1. 本文。
2. `OUTPUTS/ulip_trimodal_supcon_v3/training_plan.md`：两模式、损失、隔离要求。
3. `OUTPUTS/ulip_trimodal_supcon_v3/implementation_plan.md`。
4. `OUTPUTS/ulip_trimodal_supcon_v3/implementation_status.md`：输入审计与风险。
5. `OUTPUTS/ulip_supcon_v3_runner/README.md`：实际阶段命令，不从旧对话猜命令。
6. `OUTPUTS/ulip_trimodal_supcon_v3/human_text/report.md` 及 `real50_evaluation/report.md`。
7. `OUTPUTS/ulip_human_ground_v1/human_text/summary.json` 及 `real50_evaluation/report.md`。

注意：training_plan 开头仍写“尚未实现”，implementation_status 仍停留在 smoke 阶段。这些是历史快照，不能覆盖后续 human_text 完成报告。反过来也不能因为 human_text 完成，就认定 scene_image_text 已完成训练。

## 3. 模型及当前训练定义

```text
点云 XYZ → 预处理 → 冻结 PointBERT → 768 维特征
                                         ↓
                           可训练 pc_projection [768,1280]
                                         ↓ 单位化
                                1280 维点云语义特征
                                    ↙           ↘
                         固定文本原型          配对图像特征（仅场景模式）
                         冻结文本编码器         冻结 OpenCLIP 图像编码器
```

- 投影矩阵 983,040 个参数，无偏置、无非线性；不是另加一个传统分类头。
- 文本：每类三句固定英文，各句特征单位化，类内平均，再单位化。
- 推理：点云投影特征与六类文本原型计算余弦相似度，取最高类别；不需要图片。
- SupCon 改变点云投影特征，不更新文本编码器，不会直接降低文本原型之间的相似度。
- 同类不同 sample_id 为正样本，异类为负样本；排除自身及相同 pose ID 的重复项。不要求现有数据必须有多视角。

两种模式独立建清单、缓存、权重、报告，不混用输入：

| 模式 | 点云 | 图像 | 损失 |
|---|---|---|---|
| human_text | 纯人体；地面实验另立版本 | 无 | 文本 CE ＋可选 0.1×SupCon |
| scene_image_text | 单人＋场景完整点云 | 同帧 RGB＋实例 mask 处理出的人物图 | 文本 CE ＋0.1×图像一致性＋可选 0.1×SupCon |

v3 计划中的图像处理：人体外置灰128，bbox 扩边15%，裁剪并方形补边，224×224。不是将二值 mask 当 RGB 图送入编码器。改变为完整 RGB 或保留地面应另开版本。

已完成 human_text 配置：AdamW，lr=1e-4，weight_decay=1e-4，30 epochs，batch=256，seed=20261021；文本温度0.07，SupCon 温度0.1。按验证集 Macro-F1 选权重，两臂均从同一预训练投影矩阵独立初始化。

## 4. 实验历程

### 4.1 冻结 OpenCLIP 能力诊断

比较场景、人物相关裁剪等输入，尝试简单文本、物体与动物负类。用户要求正式对照使用同一批单人样本，每类100个、共600个，不用原来不平衡的114帧直接与其他组比较。

历史目录：`openclip_pose_diagnostic_v1`、`openclip_person_background_v1`、`openclip_person_animal_v1`。这些目录可能含早期非均衡版本；引用指标前必须核对清单，不能将其全部视作最终600样本对照。

结论：区分人／非人的能力优于细粒度姿态分类。不是靠改英文提示词即可解决所有问题。

### 4.2 早期纯人体 baseline 和图像辅助

相关目录：

- `ulip_single_pose_baseline_v1`：早期输入准备。
- `ulip_only_human_data_v1/balanced_split_v2`：数据划分、文本缓存。
- `ulip_only_human_projection_v1`：纯人体特征及训练。
- `ulip_only_human_projection_image_v2`、`ulip_projection_image_v2_runner`：旧图像辅助路线。

旧图像辅助使用的场景同帧抠人体点云，与后来要求的“完整场景点云”不同。不能直接复用其 paired_point_features 当完整场景特征，也不能把它当成新两模式方案已完成的证据。

真实数据诊断发现深蹲易被判为 supported；之后做了特征分数检查和真实／仿真点云对比。旧约66%的结果只作历史参考，不与新划分直接做严格消融。

### 4.3 纯人体 SupCon：已完成

目录：`ulip_trimodal_supcon_v3/human_text`。

- 总71,724，训练70,524，验证600、测试600。
- 验证／测试均为 AMASS，每类100，按动作序列隔离。
- 两臂均完成30轮；缓存特征下训练、测试总流程约74秒，不包括数据下载和编码缓存构建。

| 仿真测试指标 | baseline | SupCon |
|---|---:|---:|
| Accuracy | 69.17%（415/600） | 71.83%（431/600） |
| Macro-F1 | 69.42% | 72.04% |
| 跪姿召回 | 48% | 66% |
| 深蹲召回 | 66% | 61% |
| 浅蹲召回 | 75% | 68% |
| 撑姿召回 | 62% | 70% |

主要收益是跪姿、撑姿；不能说所有类别都提升。权重在各自 `baseline/best_projection.pt`、`supcon/best_projection.pt`。

### 4.4 人体加地面：已完成

目录：`ulip_human_ground_v1/human_text`。

用户确认处理：人体下移0.1 m，裁掉 Z<0 的点，加入 Z=0、半径1.5 m 的圆盘地面；人体8,000点＋地面2,000点，联合归一化。人体不足点数时存在重复采样，不能称为10,000个独立射线命中。

| 仿真600测试 | baseline | SupCon |
|---|---:|---:|
| Accuracy | 73.33% | 73.50% |
| Macro-F1 | 73.31% | 73.55% |

可视化：`ulip_human_ground_v1/input_preview/index.html`。
缓存：`ulip_human_ground_v1/human_text/input_chunks`。

已导出上传到 Openpai：`/data/timon.he/Datasets/SIM/human_ground`，71,724帧、2,242个TAR，通常32帧／包。每帧有PLY、label.txt、frame_manifest.json；没有 joints.json，不能假设有同步变换后的骨架。

### 4.5 真实50样本诊断：重要限制和实测

原始真实输入来自：
`/media/robosense/DataCenterNew/Kyle/RealData_v1_people_labels_c1_human_filtered`。

最初抽 standing10、crouching20、kneeling10、lying10，由用户人工将蹲分为浅蹲／深蹲；最终五类各10，无真实 supported 真值。

| 指标 | 纯人体 baseline | 纯人体 SupCon | 加地面 baseline | 加地面 SupCon |
|---|---:|---:|---:|---:|
| Accuracy | 48% | 50% | 68% | 62% |
| Macro-F1（五个有真值类） | 52.15% | 53.24% | 67.02% | 60.51% |
| standing 正确/10 | 9 | 9 | 10 | 10 |
| 浅蹲正确/10 | 4 | 4 | 2 | 1 |
| 深蹲正确/10 | 0 | 0 | 7 | 5 |
| 跪姿正确/10 | 5 | 5 | 5 | 5 |
| lying 正确/10 | 6 | 7 | 10 | 10 |

这是反复看过的小规模诊断集，不是新的盲测集，不用于提示词、损失权重、模型权重选择。单帧影响总准确率2个百分点。没有真实撑姿，不能报告其召回率。加地面同时改变下沉与裁剪，需独立消融才能确定具体原因。

### 4.6 BLIP-2 文本小试：已完成，未接入训练

目录：`ulip_v3_blip2_caption_pilot_v1`，查看 `report.md`、`review.html`、`summary.json`、`captions_ranked.jsonl`。

- v3 单人场景120个独立pose，每类20，训练侧抽样。
- Salesforce/blip2-opt-2.7b；每图3条候选，再使用冻结 CLIP 排序。
- 360条候选中91条为空，约25.3%；部分姿态描述不正确或过于泛化。
- 描述生成84.43秒；完整相关阶段约143.43秒，不含下载。
- 用户决定继续固定文本。不要无授权启动全量BLIP描述，也不要认为候选描述已加入训练。

## 5. 场景三模态的未解决问题

旧 v2 场景模式元数据审计：29,763单人候选、25,798独立pose；train29,261、validation245、test257。它是元数据候选数，不是全量payload校验后的最终有效数。

完整场景随机采样10,000点时，已检查示例的实际人体点只有15～69个（深蹲示例15，浅蹲23）。冻结物体级点云编码器可能主要编码背景。不能静默改成人体裁剪或实例优先采样，同时仍声称“完整场景基线不变”。

用户后来转向 v3 近距离数据。v3 与上述 v2 候选清单不是同一数据版本，需重新审计和冻结划分，不能只改路径继续用旧缓存。

必须核对同一 run/frame/view/instance/pose 的点云与RGB、mask；相同sample_id或类别不够。调平点云要按保存的变换处理，不可重复应用外参。配对投影在原始图像上检查，裁剪后的图像FOV并不等于原始雷达FOV。

多人帧不能随意配一个单人类别。第一步训练建议继续筛单人帧；若扩展多人，需要先明确实例级输入或场景多标签任务。

## 6. v3 数据补采的交接事项（状态需要刷新）

远端原始：`/data/timon.he/Datasets/SIM/ISAAC_three_stream_v3/multi_human_scene`。
原计划66,096，实际22,846成功、43,250跳过；旧done表示该轮执行结束，不表示数量达标。

补采目标43,250成功，最终66,096，人数分档1～3/4～5/6～7按6:3:1。新规则允许人物投影框重叠，槽位XY间距≥1m，雷达与槽位水平距离2～4m；每人实际Mesh可见比例≥50%、可见点≥100，实例图包含所有人物。

补采根目录：

```text
/media/robosense/DataCenterNew/Kyle/ISAAC_multi_human_2to4m_20261022/topup_success_20261026
```

远端独立扩展：

```text
/data/timon.he/Datasets/SIM/ISAAC_three_stream_v3/extensions/topup_20261026
```

700帧分层小试已完成且用户视觉批准。正式补采曾在切换仓库时因USD渲染进程段错误退出，后按用户要求不改配置直接断点恢复，原124帧保留并上传。最后一次会话确认是138成功，先前崩溃的warehouse view513已通过；这不是2026-10-27的实时进度，也不是底层崩溃已修复。

当前脚本没有渲染崩溃自动恢复。需要先看最新progress及进程，不要看到旧failure.json就重复启动。

```bash
task_topup=/media/robosense/DataCenterNew/Kyle/ISAAC_multi_human_2to4m_20261022/topup_success_20261026
cat "$task_topup/progress.json"
tail -n 30 "$task_topup/capture.log"
tail -n 30 "$task_topup/render.log"
```

运行使用GPU1几何、GPU0渲染，不要直接与新ULIP编码任务争抢。脚本身份包含代码hash，不能随意修改后无迁移resume。暂停恢复曾提交不足32帧的TAR，因此actual shard数可高于最初ceil(target/32)；判断完成看实际帧与配额，不只看expected_shards。

## 7. 建议下一位 Agent 的执行顺序

1. 阅读上述报告与runner README，核对现有代码、权重和配置；先报告当前状态，不自动重训。
2. 检查Isaac GPU、现有补采任务和存储；任何训练不得误杀采集或他人进程。
3. 与用户确认下一轮聚焦“单人完整场景三模态”，还是继续“人体＋地面”消融。两者不要混成一次无法解释的实验。
4. 若场景三模态：从v3已发布数据筛单人、完整配对、六类有效记录，记录版本/校验/排除原因。
5. 复用共同来源组隔离原则，AMASS sequence与pose身份不可跨split；新生成来源也要分组。先统计再承诺训练量。
6. 展示每类输入例子，统计完整点数、人体点数及下采样后人体占比，确认输入策略。未经用户同意不换成纯人体。
7. 同一场景清单上做严格对照：相同输入、划分、初始化和采样，只改变图像项或SupCon。纯人体与场景模式不是图像有无的严格消融。
8. 固定验证选择规则，输出各类precision/recall/F1、混淆矩阵、错误例子、权重/清单hash、编码与训练分段耗时。
9. 真实50样本仅诊断；如要宣称真实泛化改善，需要新增隔离真实验证/盲测集，补齐supported真值。

当前没有授权仅凭此交接文档启动新训练。若投影层达到瓶颈，可讨论点云编码器末层微调或实例/接触关系方案，但不是已批准实施内容。

## 8. 操作边界

- Isaac SSH：`robosense@10.30.62.124`；Openpai：`openpai@10.199.2.164`。连接与GPU状态以实时检查为准。
- Isaac解释器历史位置：`/home/robosense/workspace/ives/nvidia/IsaacSim/_build/linux-x86_64/release/python.sh`。
- 训练代码与结果主要在共享盘ULIP-2，不是全部在isaac_simulation Git里；阅读runner后再操作。
- 历史补采数据、ULIP训练、BLIP小试是不同任务，勿混用run_id、日志或done标志。
- 当前本地Git存在未解决冲突 `UU docs/README.md` 和其他用户修改；本交接未处理它们。不得擅自reset、stash、切分支或提交全部工作区。
- 本文不含凭据；不覆盖旧实验、原始TAR或标签，不按真实50结果反复改提示词。

## 9. 可直接转给下一位 Agent 的指令

> 请先读 `/media/robosense/DataCenterNew/Kyle/ULIP-2/docs/ulip2-training-handoff-20261027.md` 和其中列出的真实实验报告。当前只训练pc_projection，已完成纯人体SupCon及人体加地面实验；BLIP-2小试不采用。下一步准备单人完整场景＋同帧人物RGB＋固定文本，但必须先检查v3数据、配对、序列隔离和下采样人体占比。真实50样本只能诊断。先核查当前GPU和补采进程，不要覆盖旧结果或自行启动新训练，再给我下一步建议。
