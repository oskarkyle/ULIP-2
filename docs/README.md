# ULIP-2 实验文档入口

更新：2026-10-27。ULIP 相关说明统一保存在本工程下，不再放到 isaac_simulation 的文档目录。

## 交接与训练方案

- [训练历程与 Agent 交接](ulip2-training-handoff-20261027.md)
- [两种训练模式与损失设计](../outputs/ulip_trimodal_supcon_v3/training_plan.md)
- [实施计划](../outputs/ulip_trimodal_supcon_v3/implementation_plan.md)
- [历史实施进度与场景输入风险](../outputs/ulip_trimodal_supcon_v3/implementation_status.md)
- [Runner 使用说明](../outputs/ulip_supcon_v3_runner/README.md)

旧计划和进度文档的状态可能早于完成报告；实验结果以各实验的报告、summary、权重和清单为准。

## 训练与真实数据评估

- [早期输入准备](../outputs/ulip_single_pose_baseline_v1/report.md)
- [纯人体数据](../outputs/ulip_only_human_data_v1/report.md)
- [隔离划分](../outputs/ulip_only_human_data_v1/balanced_split_v2/report.md)
- [纯人体投影训练](../outputs/ulip_only_human_projection_v1/report.md)
- [旧图像辅助实验](../outputs/ulip_only_human_projection_image_v2/report.md)
- [纯人体 SupCon 对照](../outputs/ulip_trimodal_supcon_v3/human_text/report.md)
- [SupCon 真实50帧评估](../outputs/ulip_trimodal_supcon_v3/human_text/real50_evaluation/report.md)
- [加地面仿真测试指标](../outputs/ulip_human_ground_v1/human_text/summary.json)
- [加地面真实50帧评估](../outputs/ulip_human_ground_v1/human_text/real50_evaluation/report.md)
- [早期真实评估](../outputs/ulip_real_human_eval50_v1/report.md)
- [图像辅助真实评估](../outputs/ulip_real_human_eval50_image_v2/report.md)
- [真实深蹲特征诊断](../outputs/ulip_real_deep_feature_diagnostic_v1/feature_report.md)

## OpenCLIP 与文本实验

- [姿态诊断](../outputs/openclip_pose_diagnostic_v1/report.md)
- [简单文本对照](../outputs/openclip_pose_simple_v1/comparison_report.md)
- [人／背景实验](../outputs/openclip_person_background_v1/report.md)
- [人／动物早期实验](../outputs/openclip_person_animal_v1/report.md)
- [单人均衡物体对照](../outputs/openclip_single_person_objects_v1/report.md)
- [单人均衡动物对照](../outputs/openclip_single_person_animals_v1/report.md)
- [接触关系文本相似度](../outputs/openclip_contact_text_similarity_v1/report.md)
- [BLIP-2 小试](../outputs/ulip_v3_blip2_caption_pilot_v1/report.md)

## 文档放置约定

- 跨实验总结、交接放在 `docs/`。
- 实验报告、输入说明、运行说明保留在 `outputs/<实验>/`，保持与清单、图片、权重的相对链接有效。
- 不移动原项目 README 或上游模型说明，不覆盖旧实验结果。
- 查看全部实验 Markdown 文档：`find outputs -type f -name '*.md' | sort`（在 ULIP-2 根目录执行）。
