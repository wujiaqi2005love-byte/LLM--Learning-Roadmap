# 阶段3：工业级线上服务开发 & 容器化运维

> ⏱️ 建议学习周期：3-4 周
>
> 🎯 目标：搭建可上线生产级 LLM 服务，适配企业业务架构

## 📋 学习内容

### 01 - 模型网关与多模型调度
[📓 `01_model_gateway.ipynb`](01_model_gateway.ipynb)
- FastAPI 模型网关架构
- 多模型路由与分流
- 鉴权、限流、熔断、重试
- 灰度发布、模型热加载
- 全链路日志系统

### 02 - Docker 容器化部署
[📓 `02_docker_deployment.ipynb`](02_docker_deployment.ipynb)
- vLLM 推理镜像 Dockerfile
- 镜像分层优化
- GPU 资源限制
- Docker Compose 编排

### 03 - K8s 基础与 GPU 弹性扩缩容
[📓 `03_k8s_basics.ipynb`](03_k8s_basics.ipynb)
- Pod / Deployment / Service
- GPU 资源调度
- HPA 弹性扩缩容
- Prometheus + Grafana 监控

## 🔧 核心依赖

```bash
pip install fastapi uvicorn httpx slowapi tenacity structlog
# Docker & K8s 工具通过系统包管理器安装
```

## ✅ 验收标准

- [ ] 独立产出可交付 Docker 镜像
- [ ] 搭建多模型路由网关
- [ ] 具备限流、监控、弹性扩容能力
