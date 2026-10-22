# RAG + Tool Calling Agent

从 0 到 1 实现一个可以读取文档、检索知识、调用工具并回答问题的 Agent 应用（作业要求见 介绍.txt）。

## 项目结构

```
RAG+Tool_Calling_Agent/
├── 01_document_loader.py       # 要求1：导入 Markdown / PDF 文档（PDF 用 pymupdf4llm 转 Markdown），统一输出 Markdown 文本，原文件备份到 data/docs/（重名报错不覆盖）
├── 02_chunking.py              # 要求2：按最小章节切分 Markdown，每个块输出（不含标题的正文、父子标题链、原文 1-based 行号闭区间）
├── 03_embeddings.py            # 要求2：双向量化 —— 稠密向量（bge-small-zh-v1.5 本地离线，512 维归一化）+ 稀疏向量（BM25 词权重，语料级 IDF）
├── 04_vector_store.py          # 要求2：混合检索（稠密余弦点积 + BM25 min-max 归一化，各 0.5 加权融合），输出 Top-K（相似度, 拼接信息）
├── 05_tools.py                 # 要求3：2 个本地工具（当前时间查询、知识库 PDF 文件查询）+ OpenAI tool calling schema + dispatch_tool 统一入口
├── 06_trace_logging.py         # 要求6：@trace_agent 装饰器，记录每步 reasoning 摘要 / 工具调用 / 最终结果，逐行落盘 data/traces.jsonl
├── 07_rag_agent.py             # 要求4+5（综合）：Agent 主循环 —— 模型自主决策「检索知识库 / 调用工具 / 直接回答」，回答标注引用来源，检索无果时拒答
├── 08_evaluation.py            # 要求7：批量流程测试 —— 跑 eval/questions.json 的 20 条问题，只记录流程表现，不验证答案
├── data/
│   ├── docs/                   # 01 导入时的源文档备份（AI应用开发工程师——比特鹰成长计划.pdf）
│   └── traces.jsonl            # 06 运行生成的轨迹日志（不入库）
├── eval/
│   ├── questions.json          # 要求7：20 条测试问题（纯问题，覆盖知识检索 / 工具 / 超纲三类，不含期望答案）
│   └── eval_report.json        # 08 运行生成的流程测试报告（入库，供查看最近一次评估结果）
├── embedding/bge-small-zh-v1.5/  # 本地 Embedding 模型（183MB，不入库，按环境准备下载）
├── AI应用开发工程师——比特鹰成长计划.pdf  # 演示知识库源文档（01–07 的演示与知识库构建输入）
├── requirements.txt            # 依赖清单（python14 环境实测锁定版本）
├── .env.example                # 配置模板（复制为 .env 填入真实 Key）
└── .gitignore
```

## 环境准备

1. Python 3.14（conda 环境 `python14`），安装依赖：

   ```bash
   python -m pip install -r requirements.txt
   ```

2. 复制 `.env.example` 为 `.env`，填入真实配置：

   ```ini
   API_KEY=sk-xxx
   OpenAI_BASE_URL=https://xxx/compatible-mode/v1
   LLM_QWEN_MODEL=qwen3.8-max-0902
   ```

3. 下载本地 Embedding 模型到 `embedding/bge-small-zh-v1.5/`（约 180MB，离线加载不联网）：

   ```bash
   python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='embedding/bge-small-zh-v1.5')"
   ```

## 运行方式

01–05 均可独立运行演示（`python 01_document_loader.py` 等，演示输入为项目根目录的 PDF）。完整链路：

```bash
python 07_rag_agent.py          # 四类演示：知识库检索 / 当前时间 / 文档清单 / 超纲拒答
python 08_evaluation.py         # 流程测试：跑全部 20 条问题，报告写入 eval/eval_report.json
python 08_evaluation.py 5       # 只跑前 5 条（快速冒烟）
```

评估说明：08 为流程测试 —— 每条问题走一遍完整 Agent 链路（模型决策 → 工具调用 → 回答 → 06 轨迹落盘），只记录回答全文、工具调用与耗时，不验证答案内容；通过标准为正常返回非空回答。轨迹明细可在 data/traces.jsonl 中按 trace_id 回放。

## 进度

- [x] 01–08 全部脚本实现并实测通过
- [x] eval/questions.json 填充 20 条测试问题
- [x] 流程测试：20/20 通过（平均 28.07s/题，工具调用共 42 次）
- [x] GitHub 仓库创建与提交：https://github.com/DevDreamer-J/Biteagle_RAG_Tool_Calling_Agent
