# 飞书复赛方案成稿 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将官方模板改写并发布为可评审、可复验、图文一致的飞书参赛方案。

**Architecture:** 用一个本地、无凭证硬编码的发布脚本生成飞书 Docx block 数据；先静态校验内容，再通过 OpenAPI 原子地删除模板一级块、写入新块、上传图片并回读验收。图片与定量文本分离，确保生成图不成为数字证据。

**Tech Stack:** Python 3 标准库、系统 `curl`、Feishu Docx/Drive OpenAPI、PNG/WebP。

## Global Constraints

- 不打印或提交 `APP_SECRET`、tenant token 和本地凭证文件。
- 不修改飞书分享权限，不提交比赛报名表。
- 实测结果和试点目标必须分开表达。
- Fast-WAM 不得表述为叉车控制器或通用成功率。
- 删除模板前必须核验目标文档 token 和当前标题。

---

### Task 1: 生成并校验发布包

**Files:**
- Create: `参赛方案/tools/publish_feishu_submission.py`
- Create: `参赛方案/figures/fig-A-v2-真实仓储主视觉.png`
- Test: `参赛方案/tools/test_publish_feishu_submission.py`

**Interfaces:**
- Consumes: `参赛方案/参赛方案填表草稿.md`、`参赛方案/figures/*`、本地 `.env`。
- Produces: `build_blocks() -> list[dict]`、`validate_submission(blocks) -> None`。

- [ ] **Step 1: 写内容结构测试**

断言标题、模板四部分、成员信息、实测/目标边界、核心指标、链接和图片顺序均存在；断言不存在 `xxx队名`、`TBD`、评分表标题。

- [ ] **Step 2: 运行测试确认发布器尚不存在**

Run: `python3 -m unittest 参赛方案/tools/test_publish_feishu_submission.py -v`

Expected: FAIL，原因为无法导入发布器。

- [ ] **Step 3: 实现纯数据块构建与静态校验**

使用 heading、text、bullet、ordered、image 五类一级块；链接以可点击 text run 表示，图片路径以本地元数据保存并在 API 请求前移除。

- [ ] **Step 4: 运行结构测试**

Run: `python3 -m unittest 参赛方案/tools/test_publish_feishu_submission.py -v`

Expected: PASS。

### Task 2: 安全发布到指定飞书文档

**Files:**
- Modify: `参赛方案/tools/publish_feishu_submission.py`

**Interfaces:**
- Consumes: `--apply --document HUPfdCR66o5WnqxlTu3c6Bp7ndc --env <local-env>`。
- Produces: 更新后的飞书文档和 JSON 验收摘要。

- [ ] **Step 1: 实现只读预检**

获取 tenant token 后读取文档 metadata、root block 和一级子块，要求 token、标题和 revision 均符合预期。

- [ ] **Step 2: 实现覆盖写入**

批量删除 root 的 38 个模板子块，更新 page 标题，按 30 个一批追加正文块；每次 API 调用检查 `code == 0`。

- [ ] **Step 3: 实现图片上传与绑定**

为每幅图创建 Image block，上传到 `parent_type=docx_image`，再以 `replace_image` 绑定 file token；写请求间隔至少 0.4 秒。

- [ ] **Step 4: 执行发布**

Run: `python3 参赛方案/tools/publish_feishu_submission.py --apply --document HUPfdCR66o5WnqxlTu3c6Bp7ndc --env ../origin-master-2026-08-13/demo/.env`

Expected: 输出更新后的 revision、一级块数、图片数，且不输出 token。

### Task 3: 回读验收

**Files:**
- Test: `参赛方案/tools/test_publish_feishu_submission.py`

**Interfaces:**
- Consumes: 飞书 raw content 与全量 block 列表。
- Produces: `verify_remote_document(...) -> dict`。

- [ ] **Step 1: 回读正文**

检查标题、七个一级章节、成员信息、三组运行结果、Fast-WAM 边界、Demo 和 GitHub 链接。

- [ ] **Step 2: 回读图片块**

检查计划中的全部图片块存在且 token 非空。

- [ ] **Step 3: 检查模板残留**

断言 raw content 不包含 `企业名称🤝xxx队名`、`填写内容`、`评分项占比`、`请删除`。

- [ ] **Step 4: 运行最终验收**

Run: `python3 参赛方案/tools/publish_feishu_submission.py --verify --document HUPfdCR66o5WnqxlTu3c6Bp7ndc --env ../origin-master-2026-08-13/demo/.env`

Expected: `verification=PASS`。
