# 架构说明

面向技术面试官：这份文档解释「为什么这样分层」，而不是重复 README 里的使用说明。

## 1. 链路与数据流

```
corpus/*.md
  └─ scripts/build_corpus.py ──► data/paragraphs.jsonl（p0001…，一行一段）
       └─ chunker.build_chunks ──► Chunk（同文档同标题内打包，超长段落按句切分）
            └─ embeddings.encode ──► 归一化向量（BGE 512 维 / n-gram 2048 维）
                 └─ store.add ──► index/（ChromaDB collection 或 embeddings.npy）

question ──► embeddings.encode(is_query=True) ──► store.query(Top-20)
         ──► pipeline.retrieve：0.5×稠密分 + 0.5×n-gram 字面分 ──► Top-K
         ──► generator.generate（LLM 或抽取式）
         ──► citations.verify_answer（逐断言覆盖率）
         ──► RagAnswer（hits / generation / verification / notes）
```

一次问答的产物是一个 `RagAnswer`，`to_dict()` 同时被 `demo.json` 与 `eval_details.json` 复用，避免展示层与评测层出现两套口径。

## 2. 模块职责与边界

| 模块 | 职责 | 明确不做的事 |
| --- | --- | --- |
| `chunker` | 把段落打包成块，控制长度与重叠 | 不做分词、不做摘要 |
| `embeddings` | 文本 → 向量；负责降级到 n-gram | 不读索引、不排序 |
| `store` | 向量存取与 Top-N 查询 | 不做重排 |
| `pipeline` | 串联检索、重排、生成、校验 | 不拼装展示文案 |
| `generator` | 组织答案文本，决定拒答 | 不计算检索分 |
| `citations` | 解析 `[cXXXX]`，校验断言支撑 | 不生成答案 |
| `evaluate` | 指标计算与报告 | 不改检索/生成行为 |

关键取舍：`pipeline.retrieve()` 是唯一的检索入口，基线与完整流程都走它，只改 `k` 与 `require_citation`。评测差异因此只来自这两项，不来自「两套检索代码」。

## 3. 三个容易踩的坑与处理

1. **切块把答案切两半**：段落打包时，同块必须同文档、同 `##` 标题，否则强制断开；重叠只回退「能完整放下的尾部段落」，回退段覆盖整块时不再回退（防止死循环）。
2. **稠密模型在术语问题上漂移**：纯 BGE 在本语料 Top-3 只有 90%，且分数集中在 0.5~0.67，区分度低。解决方式不是换大模型，而是在稠密召回 20 个候选后叠字面分重排。字面分用与离线兜底同一套 n-gram 哈希，加载索引时在内存算一次。
3. **引用编号存在 ≠ 结论正确**：校验器把答案拆成断言，用「双字覆盖率 0.5 + 单字覆盖率 0.5」的混合分对比被引用块，≥ 0.55 记为已支撑。纯双字对改写过严、纯单字过松，混合分是实测调出来的。越界编号单独记录，不吞掉。

## 4. 失败处理策略

- 降级判断只在「构造期」做一次（`EmbedderFactory` / `StoreFactory`），运行期不再静默换实现。
- 降级原因写入 `index/meta.json` 的 `embedder.note` / `store.note`，并沿 `RagPipeline.notes` 流到 CLI 与报告。
- LLM 调用失败抛 `RuntimeError`（包含异常类型与消息），不返回半成品答案。
- 索引缺失时，`demo` / `eval` 自动先构建；`query` 直接报错并提示命令。

## 5. 可验证性

- 单测覆盖核心逻辑：切块边界与重叠、兜底向量的排序与跨实例确定性、引用校验的正例/负例/越界/去重。
- 评测前校验每题 `gold_snippet` 能在语料中唯一命中，防止语料改动后标准答案失效。
- 报告同时保留逐题明细，平均分好看与否都能回溯。

## 6. 如果要上生产，下一步做什么

按优先级：增量索引与删除语义 → 检索缓存与批处理 → 断言校验换成 LLM 判别 + 人工抽检 → 权限隔离与审计日志 → 在线服务化（并发与超时控制）。这些没有在当前版本实现，README 的「已知限制」里已列出。
