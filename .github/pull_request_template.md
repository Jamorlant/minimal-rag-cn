## 变更内容

<!-- 一句话说明改了什么、为什么改 -->

## 变更类型

- [ ] 功能（feature）
- [ ] 修复（fix）
- [ ] 重构（refactor，不改对外行为）
- [ ] 工程化 / 文档（chore、docs）
- [ ] 评测口径变更（会改变 README 数字）

## 验证方式

```bash
python -m unittest discover -s tests
python cli.py build
python cli.py demo
python cli.py eval
```

- [ ] 单测通过
- [ ] `demo` 输出正常（含拒答示例）
- [ ] `eval` 数字与预期一致（口径变化时 README 已同步）
- [ ] 未新增硬编码密钥 / 绝对路径

## 降级路径检查

- [ ] 未破坏「BGE → n-gram」「ChromaDB → numpy」「LLM → 抽取式」三条降级路径
- [ ] 新的失败点有显式日志或 `note`，不静默吞异常

详细模板见 [docs/PR_TEMPLATE.md](../docs/PR_TEMPLATE.md)。
