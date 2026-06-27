# 阶段2：大模型推理加速专项

> ⏱️ 建议学习周期：3-4 周
>
> 🎯 目标：掌握量化、推理框架、并发优化，构建高并发线上服务

## 📋 学习内容

### 01 - 量化技术底层与实操
[📓 `01_quantization_techniques.ipynb`](01_quantization_techniques.ipynb)
- FP16/BF16/INT8/INT4 量化原理
- GPTQ / AWQ / GGUF 实操
- KV Cache 量化与 PagedAttention

### 02 - vLLM 部署实战
[📓 `02_vllm_deployment.ipynb`](02_vllm_deployment.ipynb)
- Continuous Batching 原理
- OpenAI 兼容 API 部署
- 核心参数调优
- 并发性能测试

### 03 - SGLang 基础
[📓 `03_sglang_basics.ipynb`](03_sglang_basics.ipynb)
- RadixAttention 前缀缓存
- 结构化生成
- 投机采样
- vLLM vs SGLang 对比

### 04 - 推理调优实战
[📓 `04_inference_optimization.ipynb`](04_inference_optimization.ipynb)
- 解码参数优化
- Locust 压测
- 性能优化 Checklist
- 输出压测报告

## 🔧 核心依赖

```bash
pip install vllm auto-gptq autoawq sglang locust
```

## ✅ 验收标准

- [ ] 独立完成 7B/13B 模型量化
- [ ] vLLM 部署 OpenAI 兼容 API
- [ ] 单卡 QPS 提升数倍
- [ ] 定位延迟/显存问题
