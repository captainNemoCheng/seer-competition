# 复赛方案 Markdown 成稿 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把官方模板、仓库草稿和改良图片整理为一份可直接审阅和迁移的 Markdown 参赛方案。

**Architecture:** 最终成稿是唯一交付物，使用仓库相对路径引用图像；验证过程只做本地、只读检查，不调用飞书或任何外部写入接口。

**Tech Stack:** Markdown、PNG/WebP、Python 3 标准库、本地 shell 只读校验。

## Global Constraints

- 禁止调用飞书写入接口或修改在线文档。
- 不打印、复制或提交任何凭证。
- 实测结果和试点目标必须分开表达。
- Fast-WAM 不得表述为叉车控制器或通用成功率。
- 所有图片必须使用真实存在的相对路径。

---

### Task 1: 编写评审导向成稿

**Files:**
- Create: `参赛方案/【40强赛】仙工智能🤝鹰之团｜世界模型驱动的分层智能叉车卸货系统.md`
- Create: `参赛方案/figures/fig-A-v2-真实仓储主视觉.png`
- Create: `参赛方案/figures/fig-E-v2-FastWAM碗到黄盘主视觉.png`

- [ ] **Step 1: 写信息卡与一句话方案**

覆盖团队、赛题、成员分工和至少一项飞书 AI 能力。

- [ ] **Step 2: 写成果介绍**

外层严格沿用官方模板的四区块和内部标题；叉车师傅短缺、创新性、业务价值、AI 应用深度和落地性作为内容重点融入对应段落，不显示评分维度章节。

- [ ] **Step 3: 插入图文证据**

主视觉使用改良后的真实仓储图；技术图只承担可核验关系，所有数字在正文解释。

- [ ] **Step 4: 写边界与企业合作路线**

明确显式世界模型、学习型世界模型、Fast-WAM、SRC-5000 和 PLC 的当前边界，并用 Harness VLA 论文支撑固定原语、记忆和闭环验证设计。

### Task 2: 本地验收

**Files:**
- Verify: `参赛方案/【40强赛】仙工智能🤝鹰之团｜世界模型驱动的分层智能叉车卸货系统.md`

- [ ] **Step 1: 检查模板覆盖**

确认信息卡、成果介绍、自由展示、附件四部分齐全。

- [ ] **Step 2: 检查事实与禁用词**

检查关键实测指标、试点目标边界和限制声明；确认不存在模板占位词、评分说明、TBD 或 TODO。

- [ ] **Step 3: 检查图片路径与链接**

逐一解析 Markdown 图片路径并确认文件存在；确认 Demo 与 GitHub 地址正确。

- [ ] **Step 4: 提交本地成稿**

只提交 Markdown、改良主视觉和对应设计/计划更新，不推送远端。
