# 阶段4：分布式进阶 + 问题排查 + 综合实战

> ⏱️ 建议学习周期：4-6 周
>
> 🎯 目标：多卡分布式训练 + 全场景故障排查 + 完整工业闭环项目

## 📋 学习内容

### 01 - 多卡分布式微调
[📓 `01_distributed_training.ipynb`](01_distributed_training.ipynb)
- DeepSpeed ZeRO-1/2/3 详解
- Accelerate 多卡配置
- FSDP 基础
- 断点续训与集群调度

### 02 - 线上故障全场景排查
[📓 `02_troubleshooting.ipynb`](02_troubleshooting.ipynb)
- 训练侧：梯度爆炸、OOM、过拟合、断点恢复
- 推理侧：延迟高、并发卡死、显存泄漏、输出乱码
- 业务侧：注入攻击、上下文截断、接口超时
- 故障排查速查表

### 03 - 综合毕业大项目
[📓 `03_capstone_project.ipynb`](03_capstone_project.ipynb)
- 数据集 → QLoRA SFT → DPO 对齐
- AWQ 量化 → vLLM 部署
- FastAPI 网关 + Docker + 监控
- RAG 知识库对接
- 完整交付物清单

## 🔧 核心依赖

```bash
pip install deepspeed accelerate
# DeepSpeed 需要从源码或预编译包安装
```

## ✅ 验收标准

- [ ] 多卡环境完成大模型微调与推理
- [ ] 独立定位 90% 常见故障
- [ ] 拥有可投递简历的完整项目
