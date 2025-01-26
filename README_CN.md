# mem1: 通用智能记忆存储层

- 受mem0启发，为LLM应用提供通用的智能化的记忆存储层

- 项目仓库:<https://github.com/Galileo2017/mem1>

## mem1 框架

- 语义感知的异构图索引，将文本块和命名实体结合在一个统一结构中，减少了对复杂语义理解的依赖
- 轻量级的拓扑增强检索，利用图结构实现高效的知识发现，而无需高级语言能力

## 环境

- python 3.10.14
- uv
- uv pip install -r requirements.txt

## 安装

* 从源码安装（推荐）

```bash
cd mem1
pip install -e .
```
