# PR 描述模板

> 复制到 PR 描述中使用。仓库同时提供 `.github/pull_request_template.md`，GitHub 会自动带出。
> 这个仓库是单人项目，模板用于展示提交规范的自觉，不是形式主义。

## 变更内容

<!-- 一句话说明改了什么，为什么改 -->

## 变更类型

- [ ] 功能（feature）
- [ ] 修复（fix）
- [ ] 重构（refactor，不改对外行为）
- [ ] 工程化 / 文档（chore、docs）
- [ ] 评测口径变更（会改变 README 数字）

## 影响范围

<!-- 涉及模块、对外命令、配置项；没有影响写「无」 -->

## 验证方式

<!-- 贴真实执行的命令与关键输出，不接受「应该没问题」 -->

```bash
python -m unittest discover -s tests
python cli.py build
python cli.py demo
python cli.py eval
```

- [ ] 单测通过
- [ ] `demo` 输出正常（含拒答示例）
- [ ] `eval` 数字与预期一致（若口径变化，README 已同步）
- [ ] 无新增硬编码密钥 / 路径

## 降级路径检查

- [ ] 改动未破坏「BGE → n-gram」「ChromaDB → numpy」「LLM → 抽取式」三条降级路径
- [ ] 新的失败点有显式日志或 `note`，不静默吞异常

## 截图 / 证据

<!-- 行为变化请附 CLI 输出或截图 -->

## 关联 Issue

Closes #
