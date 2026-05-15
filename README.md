# AI ETL 流水线

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

**基于大语言模型的多智能体 ETL 自动化系统。**

解析自然语言需求 → 生成生产级 SQL 与 Python ETL 代码 → 自动化代码审查循环 → 根据业务规则验证输出 → 部署为 Airflow DAG。

---

## 系统架构

```
                          ┌─────────────────────────┐
                          │   自然语言需求            │
                          │   （通过 CLI/API 输入）   │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │   需求解析器              │
                          │   (DeepSeek Reasoner)    │
                          │   8-12 步思维链          │
                          │   结构化提取             │
                          └────────────┬────────────┘
                                       │
                          解析后的结构化需求
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │  代码生成智能体  │ │ 代码审查智能体   │ │  数据验证智能体  │
          │  (DeepSeek      │ │                 │ │                 │
          │   Coder)        │ │ 安全性 +         │ │ 行数校验、       │
          │                 │ │ 性能审查)        │ │  空值率、        │
          │  • SQL（含 CTE、 │ │                 │ │  数据分布)       │
          │    窗口函数）    │ │ • SQL 注入检测   │ │                 │
          │  • Python/pandas│ │ • 索引使用分析   │ │                 │
          └────────┬────────┘ │ • Pandas 性能优化│ └────────┬────────┘
                   │          └────────┬────────┘          │
                   │                   │                   │
                   └───────────────────┼───────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │  流水线编排器            │
                          │  状态机：                │
                          │  解析 → 生成 →           │
                          │  审查 → 验证 →           │
                          │  部署                    │
                          │  （自动修复循环 ≤3 次）   │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │   Airflow DAG 工厂       │
                          │   基于解析后的元数据      │
                          │   动态生成 DAG           │
                          └─────────────────────────┘
```

## 快速开始

```bash
# 1. 克隆并安装
git clone https://github.com/your-org/ai-etl-pipeline.git
cd ai-etl-pipeline
pip install -e ".[dev]"

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 DeepSeek API Key 和数据库连接地址

# 3. 解析自然语言需求
ai-etl parse --input "获取最近30天各地区的日活跃用户数"

# 4. 根据解析后的需求生成 ETL 代码
ai-etl generate --input examples/sample_request.json

# 5. 运行完整流水线（解析 → 生成 → 审查 → 验证 → 部署）
ai-etl run-pipeline --input "按产品类别计算月度收入"

# 6. 验证已有的输出数据
ai-etl validate --source-table raw_events --target-table agg_daily_users
```

## 命令行工具

| 命令             | 说明                                                         |
|------------------|--------------------------------------------------------------|
| `parse`          | 将自然语言 ETL 需求解析为结构化元数据                       |
| `generate`       | 根据解析后的需求生成 SQL + Python ETL 代码                  |
| `run-pipeline`   | 端到端执行完整的多智能体流水线                               |
| `validate`       | 根据源数据和业务规则验证 ETL 输出数据                       |

## 项目结构

```
ai-etl-pipeline/
├── pyproject.toml                  # 项目配置、依赖及工具设置
├── .env.example                    # 环境变量模板
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py                   # 基于 Pydantic-settings 的配置管理
│   ├── main.py                     # Click CLI 入口
│   ├── parser/
│   │   ├── __init__.py
│   │   └── requirement_parser.py   # 自然语言 → 结构化需求（思维链提取）
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # 重试逻辑、Token 追踪、回调机制
│   │   ├── code_gen_agent.py       # SQL + Python 代码生成
│   │   ├── code_review_agent.py    # 安全性与性能审查
│   │   └── data_validator.py       # 输出数据验证
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   └── pipeline_orchestrator.py # 多智能体状态机编排
│   └── airflow/
│       ├── __init__.py
│       └── etl_dag_factory.py      # 动态 Airflow DAG 生成
├── tests/
│   ├── __init__.py
│   ├── test_parser.py              # 解析器单元测试（Mock API）
│   └── test_agents.py              # 智能体单元测试（Mock 响应）
└── examples/
    ├── README.md                   # 示例说明文档
    └── sample_request.json         # 示例 ETL 请求
```

## 配置说明

所有设置均从环境变量（或 `.env` 文件）加载，并通过 `pydantic-settings` 进行验证。主要配置项如下：

| 变量名                   | 默认值                           | 说明                             |
|--------------------------|----------------------------------|----------------------------------|
| `DEEPSEEK_API_KEY`       | —                                | LLM 服务商的 API 密钥           |
| `DEEPSEEK_BASE_URL`      | `https://api.deepseek.com/v1`   | OpenAI 兼容的 API 端点地址      |
| `DEEPSEEK_MODEL`         | `deepseek-reasoner`             | 推理任务使用的模型              |
| `DEEPSEEK_CODER_MODEL`   | `deepseek-coder`                | 代码生成与审查使用的模型        |
| `SOURCE_DB_URL`          | —                                | 源数据库连接字符串              |
| `TARGET_DB_URL`          | —                                | 目标数据仓库连接字符串          |
| `MAX_RETRY_ATTEMPTS`     | `3`                              | 审查循环中的最大自动修复重试次数 |
| `OUTPUT_DIR`             | `./output`                       | 生成产物的输出目录              |

## 开发指南

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 类型检查
mypy src/

# 代码规范检查
ruff check src/ tests/
```

## 许可证

MIT
