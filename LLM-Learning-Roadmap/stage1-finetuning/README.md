# Stage 1: Fine-Tuning（微调）

> **从数据工程到模型导出 — 完整的 LLM 微调流水线**

本阶段包含 **4 个 Jupyter Notebook**，覆盖从数据集构建到模型导出的完整微调链路。你将依次掌握：数据工程 → SFT 监督微调 → DPO 偏好对齐 → 模型合并与量化部署。

---

## 📋 学习路线图

```
Notebook 1: 数据工程          Notebook 2: LoRA/QLoRA SFT     Notebook 3: DPO 对齐         Notebook 4: 合并 & 导出
┌──────────────────┐       ┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│ • 加载 HuggingFace │  ──▶  │ • 4-bit 量化模型加载  │  ──▶  │ • RLHF vs DPO 理念   │  ──▶  │ • merge_and_unload  │
│   数据集           │       │ • LoRA 低秩适配     │       │ • 偏好数据集构建     │       │ • safetensors 保存   │
│ • 格式转换          │       │ • SFTTrainer 训练   │       │ • DPOTrainer 训练    │       │ • HF Hub 推送        │
│   (Alpaca/ShareGPT/ │       │ • 超参深度解析      │       │ • Beta 参数敏感性    │       │ • GGUF/GPTQ 量化     │
│    Qwen Chat)      │       │ • 训练可视化        │       │ • SFT vs DPO 对比    │       │ • 版本管理           │
│ • 数据清洗管线       │       │ • 内存优化         │       │ • 自动化评分         │       │ • 增量微调工作流     │
│ • 去重 & 质量过滤    │       │ • 问题排查指南      │       │ • 常见陷阱排查       │       │                     │
│ • 分层划分 & 可视化  │       │                    │       │                    │       │                     │
└──────────────────┘       └───────────────────┘       └───────────────────┘       └───────────────────┘
```

---

## 🗂️ 文件结构

```
stage1-finetuning/
├── README.md                              # 本文件
│
├── 01_dataset_engineering.ipynb           # Notebook 1: 数据工程
├── 02_lora_qlora_sft.ipynb                # Notebook 2: LoRA/QLoRA 监督微调
├── 03_dpo_alignment.ipynb                 # Notebook 3: DPO 偏好对齐
├── 04_model_merge_export.ipynb            # Notebook 4: 模型合并与导出
│
├── customer_service_dataset/              # Notebook 1 产物：示例数据集 (HF Dataset 格式)
│   ├── dataset_dict.json
│   ├── train/
│   └── test/
│
├── customer_service_train.json            # Notebook 1 产物：训练集 JSON
├── customer_service_test.json             # Notebook 1 产物：测试集 JSON
│
├── dataset_stats.png                      # Notebook 1 产物：数据分布可视化
└── dolly_categories.png                   # Notebook 1 产物：Dolly 类别分布
```

---

## 📓 各 Notebook 详解

### Notebook 1: Dataset Engineering（数据工程）

**目标**：掌握指令微调数据集的全生命周期 — 加载、转换、清洗、可视化、构建。

| 模块 | 内容 | 关键函数/技术 |
|------|------|---------------|
| **数据集加载** | 从 HuggingFace Hub 加载 Alpaca、Dolly、OpenOrca 三个数据集 | `datasets.load_dataset()` |
| **格式转换** | 实现 Alpaca ↔ ShareGPT ↔ Qwen Chat 三种格式互转 | `alpaca_to_sharegpt()`, `sharegpt_to_qwen_chat()`, `dolly_to_sharegpt()` |
| **格式统一** | 合并多源数据到统一的 ShareGPT 格式 | `concatenate_datasets()` |
| **质量过滤** | 基于词数、字符熵、轮次数等指标过滤低质量样本 | 字符级 Shannon 熵、长度截断 |
| **去重** | 精确去重（SHA256）+ 近似去重（MinHash 演示） | `hashlib`, MinHash 签名 |
| **分层划分** | 按对话长度分层划分 train/val，保证分布一致 | `sklearn.train_test_split(stratify=...)` |
| **可视化** | 词数分布直方图、熵分布、长度类别饼图、类别柱状图 | `matplotlib` |
| **人工审查** | 随机采样 + 格式化展示 + 质量标记牌 | `pretty_print_conversation()`, `quality_flags()` |
| **构建数据集** | 模板驱动生成 80 条客服对话，保存为 HF Dataset 和 JSON | `Dataset.from_dict()`, `save_to_disk()` |

**产物**：
- `customer_service_dataset/` — HuggingFace Dataset 格式的客服数据集
- `customer_service_train.json` / `customer_service_test.json` — JSON 导出
- `dataset_stats.png` / `dolly_categories.png` — 可视化图表

---

### Notebook 2: LoRA & QLoRA Supervised Fine-Tuning（LoRA/QLoRA 监督微调）

**目标**：从零开始完成一次 LoRA/QLoRA 微调，深入理解每个超参数。

| 模块 | 内容 | 关键技术 |
|------|------|----------|
| **LoRA 原理** | 低秩分解数学推导：$\Delta W = BA$，前向 $h = Wx + \frac{\alpha}{r}BAx$ | 秩 r、缩放因子 α、dropout |
| **模型加载** | Qwen2.5-0.5B + 4-bit NF4 量化 + 双重量化 | `BitsAndBytesConfig`, NormalFloat4 |
| **LoRA 配置** | r=16, α=32, target_modules 覆盖 attention + MLP | `LoraConfig`, `get_peft_model()` |
| **数据格式化** | 使用 Qwen2.5 Chat Template 将 Alpaca 转换为训练格式 | `tokenizer.apply_chat_template()` |
| **Loss Masking** | `DataCollatorForCompletionOnlyLM` — 仅对 assistant 回复计算 loss | `response_template="<\|im_start\|>assistant"` |
| **训练配置** | 完整的 SFT 超参指南（lr、warmup、cosine schedule、grad accumulation…） | `TrainingArguments` |
| **训练执行** | `SFTTrainer` 封装训练循环 | TRL |
| **Loss 可视化** | 绘制 train/eval loss 曲线 + 平滑处理 | `matplotlib` |
| **生成对比** | Before vs After 微调的生成效果对比 | `model.generate()` |
| **内存优化** | 4-bit vs 8-bit vs FP16 显存占用对比 + Flash Attention 检测 | 梯度检查点、SDPA |
| **日志追踪** | WandB / TensorBoard 配置指南 | `report_to` |
| **问题排查** | OOM 解决方案清单 + Loss 不收敛诊断流程 | 8 种 OOM 缓解策略 + 7 种收敛问题排查 |

**关键超参数速查**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `r` (rank) | 8–64 | 秩越大适配器容量越大，参数量随之增加 |
| `lora_alpha` | 16–32 | 缩放因子，实际学习率 = α/r |
| `learning_rate` | 1e-4 ~ 5e-4 | LoRA 可使用比全量微调更高的学习率 |
| `warmup_ratio` | 0.03–0.1 | 从 0 线性预热到峰值 LR 的步数比例 |
| `num_train_epochs` | 1–3 | LoRA 收敛快，通常 1-3 轮足够 |
| `max_seq_length` | 512–2048 | 根据显存调整，超过此值的序列会被截断 |

---

### Notebook 3: DPO Alignment（DPO 偏好对齐）

**目标**：用 Direct Preference Optimization 替代传统 RLHF，一步完成偏好对齐。

| 模块 | 内容 | 关键技术 |
|------|------|----------|
| **RLHF vs DPO** | 对比两种范式的架构差异：DPO 只需 2 个模型（vs RLHF 的 4 个） | DPO 损失函数推导 |
| **DPO 数据格式** | prompt + chosen + rejected 三元组结构 | — |
| **偏好数据集** | 构建覆盖 5 种偏好类型的合成数据集（格式化、幻觉、安全、完整性、简洁性） | 10 组对比样本 |
| **模型准备** | 加载 Qwen2.5-0.5B + 4-bit + LoRA（可在 SFT checkpoint 之上） | `prepare_model_for_kbit_training()` |
| **DPOTrainer 训练** | 用 TRL 的 DPOTrainer 执行偏好学习 | `DPOConfig` |
| **训练指标** | Loss 曲线 + 隐式奖励（chosen/rejected/margin）可视化 | `rewards/chosen`, `rewards/rejected`, `rewards/margins` |
| **SFT vs DPO 对比** | 在幻觉、格式化、安全、完整性 4 个维度对比生成效果 | 定性对比 |
| **自动评分** | 3 个启发式评分器：格式合规、长度合适、安全性 | 规则评分（生产环境应使用 LLM-as-judge） |
| **Beta 敏感性** | 详解 β 从 0.01 到 1.0 的行为差异 | 奖励边际监控 |
| **DPO 变体** | sigmoid / hinge / IPO / KTO / ORPO / SimPO 选用指南 | `loss_type` |
| **常见陷阱** | 参考模型选择、偏好数据质量、奖励过优化、能力遗忘 | 6 项排查清单 |

**DPO 损失函数**：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

**核心直觉**：增大 chosen 回复相对于 rejected 回复的概率，同时用 KL 惩罚约束模型不要偏离参考模型太远。

**β 参数选择指南**：

| β 值 | 行为 | 适用场景 |
|------|------|----------|
| 0.01 | 激进学习，奖励边际高。风险：模型退化、奖励作弊 | 偏好信号非常明确时 |
| **0.1** | **平衡（默认值）**。良好的偏好信号，不会偏离太远 | **大多数场景** |
| 0.5 | 保守。与参考模型偏离小 | 偏好差异微妙时 |
| 1.0 | 非常保守。几乎不改变行为 | 参考模型已经很强时 |

---

### Notebook 4: Model Merge & Export（模型合并与导出）

**目标**：将 LoRA 适配器合并为完整的可部署模型，并导出为多种格式。

| 模块 | 内容 | 关键技术 |
|------|------|----------|
| **加载适配器** | 从 base model + LoRA adapter 恢复模型 | `PeftModel.from_pretrained()`, `PeftConfig` |
| **合并权重** | `merge_and_unload()` 将 LoRA 矩阵融入基座权重 | 合并后为标准模型（无 PEFT 依赖） |
| **保存模型** | 以 safetensors 格式保存合并后的完整模型 | `save_pretrained(safe_serialization=True)` |
| **完整性校验** | 重新加载合并模型 + 生成测试验证 | 生成对比 |
| **HF Hub 推送** | 将模型推送到 HuggingFace Hub | `create_repo()`, `upload_folder()` |
| **GGUF 导出** | llama.cpp GGUF 转换 + Ollama 部署指南 | `convert_hf_to_gguf.py`, `Modelfile` |
| **GPTQ 量化** | AutoGPTQ 4-bit 量化（含校准数据准备） | `BaseQuantizeConfig`, `quantize()` |
| **版本管理** | 生产级目录结构规范 + 命名约定 | `{stage}-{version}-{key-hyperparams}` |
| **增量微调工作流** | Base → SFT → Merge → DPO → Merge → Quantize → Deploy | 完整 6 步流水线 |

**量化方案速查**：

| 方法 | 位宽 | 7B 模型大小 | 速度 | 质量 | 最佳场景 |
|------|------|------------|------|------|----------|
| FP16 | 16 | ~14 GB | 基准 | 完美 | 训练、评估 |
| INT8 | 8 | ~7 GB | 1.3x | 近乎完美 | CPU 推理 |
| GPTQ 4-bit | 4 | ~4 GB | 2-3x | 很好 | GPU 服务 |
| AWQ 4-bit | 4 | ~4 GB | 2-3x | 很好 | GPU 服务（比 GPTQ 更快） |
| GGUF Q4_K_M | 4 | ~4 GB | 1.5x | 好 | CPU / 边缘 / Ollama |
| GGUF Q2_K | 2 | ~2.5 GB | 2x | 有损 | 极限压缩 |

> **4-bit 是甜点区**：相对于 8-bit 体积减半，对大多数任务质量损失可忽略不计。

---

## 🔄 完整工作流

```
Base Model (Qwen2.5-7B)
  │
  ├── Step 1: SFT on domain data      →  adapters/sft-lora/
  │   [Notebook 1: 构建数据集]
  │   [Notebook 2: LoRA SFT]
  │
  ├── Step 2: Merge SFT                →  checkpoints/sft-v1/
  │   [Notebook 4: merge_and_unload]
  │
  ├── Step 3: DPO on preference data   →  adapters/dpo-lora/
  │   [Notebook 3: DPO 对齐]
  │   Reference = checkpoints/sft-v1/（而非原始 base model！）
  │
  ├── Step 4: Merge DPO                →  checkpoints/dpo-v1/
  │   [Notebook 4: merge_and_unload]
  │
  ├── Step 5: Quantize                 →  exports/gptq-4bit/ 或 .gguf
  │   [Notebook 4: GPTQ / GGUF]
  │
  └── Step 6: Push to Hub & Deploy
      [Notebook 4: HF Hub + Ollama]
```

> ⚠️ **关键规则**：始终在合并前单独保存 LoRA 适配器。合并是单向操作（适配器权重被吸收后无法恢复）。保留版本化的适配器。

---

## 🛠️ 环境依赖

所有 notebook 均使用以下核心库：

```bash
pip install transformers accelerate peft trl bitsandbytes datasets
pip install torch matplotlib scikit-learn pandas tqdm pyarrow
pip install huggingface_hub auto-gptq  # Notebook 4 额外依赖
```

| 库 | 用途 |
|----|------|
| `transformers` | 模型加载、Tokenizer、生成 |
| `peft` | LoRA 配置与适配器管理 |
| `trl` | SFTTrainer、DPOTrainer |
| `bitsandbytes` | 4-bit/8-bit 量化 |
| `datasets` | HuggingFace 数据集加载与处理 |
| `accelerate` | 分布式训练与设备映射 |
| `scikit-learn` | 分层划分、TF-IDF |
| `matplotlib` | 数据可视化 |
| `auto-gptq` | GPTQ 量化导出 |

---

## 📊 关键概念速查

### LoRA 核心思想

冻结预训练权重 W，注入可训练的低秩分解矩阵 BA：

$$\Delta W = B \cdot A, \quad B \in \mathbb{R}^{d \times r}, \ A \in \mathbb{R}^{r \times k}$$

只有 BA 参与梯度更新，参数量从 d×k 降至 r×(d+k)。当 r ≪ min(d,k) 时，训练参数可减少 99% 以上。

### DPO 核心思想

跳过显式奖励模型训练，直接从偏好对（chosen/rejected）中优化策略：

- 增大模型对 chosen 回复的似然
- 减小模型对 rejected 回复的似然
- 用 KL 散度约束模型不偏离参考模型太远

### 量化核心思想

将浮点权重映射到低精度整数，用更少的比特表示每个参数：

- **NF4 (NormalFloat4)**：信息论最优的 4-bit 数据类型
- **双重量化**：对量化常数再进行一次量化，每个参数再省约 0.4 bit
- **GPTQ**：基于近似二阶信息的逐层量化，对 GPU 推理优化最好
- **GGUF**：llama.cpp 格式，支持 CPU 上的混合精度推理

---

## ⚡ GPU 显存预估

以 Qwen2.5-7B 为例：

| 配置 | 训练显存 | 推理显存 |
|------|----------|----------|
| Full fine-tune (FP32) | ~112 GB | ~28 GB |
| Full fine-tune (FP16) | ~56 GB | ~14 GB |
| LoRA (FP16) | ~20 GB | ~14 GB |
| QLoRA (NF4, r=16) | **~10 GB** | ~4 GB |
| QLoRA (NF4, r=8) | **~8 GB** | ~4 GB |

> QLoRA 让 7B 模型在消费级 GPU（RTX 3090/4090 24GB）上训练成为可能。

---

## 📝 学习建议

1. **按顺序完成** — 每个 notebook 的产物是下一个的输入（SFT 适配器 → DPO → 合并导出）
2. **先通读再动手** — 每个 notebook 开头有 markdown cell 说明目标和知识点
3. **动手修改参数** — 改变 r、α、β、lr 等参数，观察对 loss 和生成效果的影响
4. **关注问题排查** — Notebook 2 和 3 末尾附有详细的 troubleshooting 指南，遇到问题先查
5. **保存所有中间产物** — LoRA 适配器、合并模型、量化模型分别保存，便于回溯和对比

---

## 🔗 相关资源

- [HuggingFace PEFT 文档](https://huggingface.co/docs/peft)
- [TRL (Transformer Reinforcement Learning) 文档](https://huggingface.co/docs/trl)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314) — QLoRA: Efficient Finetuning of Quantized LLMs
- [DPO 论文](https://arxiv.org/abs/2305.18290) — Direct Preference Optimization
- [bitsandbytes 文档](https://huggingface.co/docs/bitsandbytes)
- [Llama.cpp GGUF 格式说明](https://github.com/ggerganov/llama.cpp)

---

## 📦 下一阶段

完成本阶段后，进入 [Stage 2: Inference（推理部署）](../stage2-inference/)，学习：

- 高效推理引擎（vLLM、llama.cpp）
- Batch inference 与流式输出
- 推理优化（KV cache、continuous batching）
- 在线服务部署
