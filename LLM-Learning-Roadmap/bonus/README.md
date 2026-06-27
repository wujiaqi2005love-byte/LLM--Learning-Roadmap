# 🎁 求职加分拓展

> ⏱️ 主线学完后的选学内容
>
> 🎯 目标：补充前沿技术栈，提升简历竞争力

## 📋 学习内容

### 01 - LLM 安全
[📓 `01_llm_security.ipynb`](01_llm_security.ipynb)
- Prompt 注入攻击与防御
- 越狱检测与防护
- 输入/输出内容过滤
- OWASP Top 10 for LLM
- Guardrails 实现

### 02 - 自动化评测
[📓 `02_auto_evaluation.ipynb`](02_auto_evaluation.ipynb)
- Perplexity / BLEU / ROUGE
- LLM-as-Judge 评测
- C-Eval / CMMLU 中文基准
- 自建业务指标评测体系
- 回归测试 Pipeline

### 03 - Embedding 微调
[📓 `03_embedding_finetune.ipynb`](03_embedding_finetune.ipynb)
- BGE / E5 / Jina Embeddings
- Contrastive Learning 微调
- Hard Negative Mining
- RAG 检索效果提升对比
- t-SNE 可视化

## 🔧 核心依赖

```bash
pip install sentence-transformers faiss-cpu langchain
```

## 🌟 更多拓展方向

| 方向 | 关键词 | 学习资源 |
|------|--------|---------|
| MoE 模型微调 | Mixtral, DeepSeek-MoE | Hugging Face Blog |
| 长上下文优化 | YaRN, ALiBi, Ring Attention | 论文精读 |
| 多模态微调 | LLaVA, Qwen-VL | LLaVA 官方文档 |
| 模型压缩 | 知识蒸馏, 剪枝 | MIT HAN Lab |
| RLHF 进阶 | PPO, GRPO | DeepSpeed-Chat |
