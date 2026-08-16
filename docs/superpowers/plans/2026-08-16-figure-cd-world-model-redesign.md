# Figure C/D World-Model Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild proposal figures C and D as precise, readable world-model infographics, synchronize the proposal copy, and publish the verified assets to the existing GitHub feature branch.

**Architecture:** Keep editable SVG files as the single design source and render stable PNG deliverables with macOS CoreGraphics through `sips`, which preserves Chinese glyph fallback. A focused unittest contract validates required terminology, forbidden legacy labels, document placement, PNG dimensions, and the 8-bit CoreGraphics export before visual QA.

**Tech Stack:** SVG 1.1, macOS `sips`/CoreGraphics, ImageMagick 7 for PNG resizing only, Python `unittest`, standard-library XML/PNG parsing, Markdown, Git.

## Global Constraints

- Figure C remains a four-layer architecture: Aily interaction, AgentOS task brain, world-model prediction, skill cerebellum and safe execution.
- Figure D has three non-crossing parallel lanes（三条平行泳道）: normal execution, automatic recovery, and safe intervention.
- Main-architecture wording `VLA / 执行层` must disappear; proper nouns `Harness VLA` and objective Fast-WAM classification remain unchanged.
- Current Demo must be labeled an explicit world model built from physics simulation, Scene Graph, state machine, collision prediction, and failure memory.
- Learning-based world models must be labeled as a future enterprise-data phase, not a completed result.
- SVG text must use system Chinese sans-serif fonts and a minimum 20 px body size.
- PNG outputs must retain the existing filenames and be at least 1600 px wide.
- Normal, recovery, completion, and intervention keep blue, amber, green, and red semantics; world-model state and prediction use cyan.
- Do not change Demo behavior, skill order, Fallback identifiers, measured terminal states, or measured durations.

---

### Task 1: Add the proposal-figure contract

**Files:**
- Create: `tests/test_proposal_figures.py`
- Test: `tests/test_proposal_figures.py`

**Interfaces:**
- Consumes: repository-root-relative paths under `参赛方案/`.
- Produces: `ProposalFigureContractTests`, which later tasks use as the acceptance gate for editable SVGs, rendered PNGs, and Markdown placement.

- [ ] **Step 1: Write the failing contract tests**

Create the test module with these exact checks:

```python
import struct
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "参赛方案" / "figures"
PROPOSAL = ROOT / "参赛方案" / "【40强赛】仙工智能🤝鹰之团｜世界模型驱动的分层智能叉车卸货系统.md"


class ProposalFigureContractTests(unittest.TestCase):
    def svg_text(self, filename: str) -> str:
        path = FIGURES / filename
        self.assertTrue(path.is_file(), path)
        ET.parse(path)
        return path.read_text(encoding="utf-8")

    def test_editable_svg_sources_exist_and_parse(self):
        self.svg_text("fig-C-四层架构总览.svg")
        self.svg_text("fig-D-九技能链与Fallback状态机.svg")

    def test_figure_c_uses_world_model_four_layer_copy(self):
        text = self.svg_text("fig-C-四层架构总览.svg")
        for label in (
            "飞书 Aily 交互入口", "AgentOS 任务大脑", "世界模型预测层",
            "技能小脑与安全执行层", "显式世界模型",
            "学习型世界模型：企业数据阶段",
        ):
            self.assertIn(label, text)

    def test_figure_d_contains_nine_skills_and_three_terminal_paths(self):
        text = self.svg_text("fig-D-九技能链与Fallback状态机.svg")
        for label in (
            "进箱导航", "精准对位", "栈板识别", "货叉插入", "门架起升",
            "货叉倾斜", "月台区导航", "传送带对接", "栈板放置",
            "FB-01 横向修正", "FB-02 切换观察位姿",
            "FB-07 箱外安全停车", "COMPLETED · 9/9", "HUMAN REQUIRED",
        ):
            self.assertIn(label, text)

    def test_legacy_vla_layer_copy_is_absent(self):
        paths = (
            FIGURES / "fig-C-四层架构总览.svg",
            FIGURES / "fig-D-九技能链与Fallback状态机.svg",
            FIGURES / "图片选用说明.md",
            ROOT / "参赛方案" / "参赛方案填表草稿.md",
        )
        for path in paths:
            if path.exists():
                self.assertNotIn("VLA / 执行层", path.read_text(encoding="utf-8"))

    def test_png_outputs_are_at_least_1600_pixels_wide(self):
        for filename in (
            "fig-C-四层架构总览.png",
            "fig-D-九技能链与Fallback状态机.png",
        ):
            data = (FIGURES / filename).read_bytes()
            self.assertEqual(data[:8], b"\\x89PNG\\r\\n\\x1a\\n")
            self.assertGreaterEqual(struct.unpack(">I", data[16:20])[0], 1600)

    def test_proposal_places_figure_c_before_figure_d(self):
        text = PROPOSAL.read_text(encoding="utf-8")
        figure_c = text.find("./figures/fig-C-四层架构总览.png")
        figure_d = text.find("./figures/fig-D-九技能链与Fallback状态机.png")
        self.assertGreaterEqual(figure_c, 0)
        self.assertGreater(figure_d, figure_c)
```

The test must require these Figure C strings: `飞书 Aily 交互入口`, `AgentOS 任务大脑`, `世界模型预测层`, `技能小脑与安全执行层`, `显式世界模型`, and `学习型世界模型：企业数据阶段`.

The test must require these Figure D strings: the nine names `进箱导航`, `精准对位`, `栈板识别`, `货叉插入`, `门架起升`, `货叉倾斜`, `月台区导航`, `传送带对接`, `栈板放置`; `FB-01 横向修正`, `FB-02 切换观察位姿`, `FB-07 箱外安全停车`, `COMPLETED · 9/9`, and `HUMAN REQUIRED`.

Parse SVG with `xml.etree.ElementTree`. Read PNG width from bytes 16–20 of the PNG IHDR header using `struct.unpack(">I", data[16:20])`, avoiding a new runtime dependency.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=demo python3 -m unittest tests.test_proposal_figures -v
```

Expected: failure because the two `.svg` sources do not exist and Figure C is not yet placed in the final proposal.

- [ ] **Step 3: Commit the contract test**

```bash
git add tests/test_proposal_figures.py
git commit -m "test: define proposal figure contracts"
```

### Task 2: Rebuild Figure C from an editable SVG source

**Files:**
- Create: `参赛方案/figures/fig-C-四层架构总览.svg`
- Modify: `参赛方案/figures/fig-C-四层架构总览.png`
- Test: `tests/test_proposal_figures.py`

**Interfaces:**
- Consumes: Figure C required strings and visual constraints from Task 1.
- Produces: a 1920×1280 editable SVG and rendered PNG with the same basename.

- [ ] **Step 1: Create the SVG source**

Use one `<svg viewBox="0 0 1920 1280">` with reusable CSS classes for title, layer cards, body copy, chips, arrows, and the unified evidence sidebar. Lay out four layer cards at fixed y positions and use these exact headings:

```text
01 飞书 Aily 交互入口
02 AgentOS 任务大脑
03 世界模型预测层
04 技能小脑与安全执行层
```

Use a single downward data-flow axis and one cyan feedback arrow from layer 04 back to layer 03. The right sidebar title is `飞书协同与单一证据链` and lists `任务 → 预测 → 技能 → Fallback → 视频 → Manifest`.

Add a bottom boundary strip with exact copy:

```text
当前：显式世界模型 = 物理仿真 + Scene Graph + 状态机 + 碰撞预测 + 失败记忆
下一阶段：学习型世界模型：企业数据阶段（未训练，不作为现有成果）
```

- [ ] **Step 2: Render the PNG**

Run:

```bash
sips -s format png '参赛方案/figures/fig-C-四层架构总览.svg' \
  --out '参赛方案/figures/fig-C-四层架构总览.png'
```

- [ ] **Step 3: Run the Figure C focused tests**

Run:

```bash
PYTHONPATH=demo python3 -m unittest \
  tests.test_proposal_figures.ProposalFigureContractTests.test_editable_svg_sources_exist_and_parse \
  tests.test_proposal_figures.ProposalFigureContractTests.test_figure_c_uses_world_model_four_layer_copy \
  tests.test_proposal_figures.ProposalFigureContractTests.test_legacy_vla_layer_copy_is_absent \
  tests.test_proposal_figures.ProposalFigureContractTests.test_png_outputs_are_at_least_1600_pixels_wide -v
```

Expected: Figure C assertions pass; the shared source-existence test may remain red until Figure D is added.

- [ ] **Step 4: Commit Figure C**

```bash
git add '参赛方案/figures/fig-C-四层架构总览.svg' \
  '参赛方案/figures/fig-C-四层架构总览.png'
git commit -m "docs: redesign world-model architecture figure"
```

### Task 3: Rebuild Figure D as non-crossing execution lanes

**Files:**
- Create: `参赛方案/figures/fig-D-九技能链与Fallback状态机.svg`
- Modify: `参赛方案/figures/fig-D-九技能链与Fallback状态机.png`
- Test: `tests/test_proposal_figures.py`

**Interfaces:**
- Consumes: Figure D skill and Fallback contract from Task 1.
- Produces: a 1920×1280 editable SVG and rendered PNG with an explicit world-model loop and three orthogonal lanes.

- [ ] **Step 1: Create the SVG source**

Use one `<svg viewBox="0 0 1920 1280">`. Place the five-step cyan world-model loop across the top:

```text
观察状态 → 预测候选后果 → 选择白名单技能 → 执行与验证 → 更新状态
```

Place nine numbered skill cards in a single normal lane, visually grouped into three phases: `进入与感知`, `叉取`, and `转运与放置`. Use only horizontal arrows between adjacent cards.

Below it, draw:

- an amber recovery lane from `首次对位失败` to `FB-01 横向修正` and an orthogonal return labeled `更新世界状态后重新验证`;
- a red intervention lane from `连续识别失败` through `FB-02 切换观察位姿` and `FB-07 箱外安全停车` to `HUMAN REQUIRED`.

Use exact terminal measurements `COMPLETED · 9/9 · 66.25 s`, `恢复后 COMPLETED · 9/9 · 77.25 s`, and `HUMAN REQUIRED · 36.0 s`.

- [ ] **Step 2: Render the PNG**

Run:

```bash
sips -s format png '参赛方案/figures/fig-D-九技能链与Fallback状态机.svg' \
  --out '参赛方案/figures/fig-D-九技能链与Fallback状态机.png'
```

- [ ] **Step 3: Run all proposal-figure tests**

Run:

```bash
PYTHONPATH=demo python3 -m unittest tests.test_proposal_figures -v
```

Expected: all SVG, terminology, skill, terminal, and PNG-dimension assertions pass except the proposal-placement test, which remains red until Task 4.

- [ ] **Step 4: Commit Figure D**

```bash
git add '参赛方案/figures/fig-D-九技能链与Fallback状态机.svg' \
  '参赛方案/figures/fig-D-九技能链与Fallback状态机.png'
git commit -m "docs: clarify world-model recovery figure"
```

### Task 4: Synchronize the proposal and image guidance

**Files:**
- Modify: `参赛方案/【40强赛】仙工智能🤝鹰之团｜世界模型驱动的分层智能叉车卸货系统.md`
- Modify: `参赛方案/figures/图片选用说明.md`
- Modify: `参赛方案/参赛方案填表草稿.md`
- Modify: `参赛方案/figures/00-总览.html`
- Test: `tests/test_proposal_figures.py`

**Interfaces:**
- Consumes: final Figure C and Figure D PNG filenames from Tasks 2–3.
- Produces: final-document placement and terminology consistent with the editable figure sources.

- [ ] **Step 1: Insert Figure C in the final proposal**

After the paragraph defining the current explicit world model and before `原创 Agent 工作流`, insert:

```markdown
![分层智能架构：业务意图、世界模型与受控执行](./figures/fig-C-四层架构总览.png)

*图 1｜世界模型位于 AgentOS 与技能小脑之间：预测候选后果，但不绕过规控与安全门禁。*
```

Renumber the current Figure D and Fast-WAM captions to Figure 2 and Figure 3.

- [ ] **Step 2: Update image guidance and legacy draft copy**

In `图片选用说明.md`, replace Figure C's legacy VLA wording with `世界模型预测候选后果，技能小脑输出受控目标参数`, update its source to `SVG 精确绘制并导出 PNG`, and update Figure D's source similarly.

In `参赛方案填表草稿.md`, replace the main-layer bullet `VLA / 执行层` with separate world-model and skill-cerebellum responsibilities without changing paper titles or Fast-WAM classification.

Update `00-总览.html` displayed file sizes after rendering, keeping the same image filenames.

- [ ] **Step 3: Run the complete proposal-figure contract**

Run:

```bash
PYTHONPATH=demo python3 -m unittest tests.test_proposal_figures -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit document synchronization**

```bash
git add \
  '参赛方案/【40强赛】仙工智能🤝鹰之团｜世界模型驱动的分层智能叉车卸货系统.md' \
  '参赛方案/figures/图片选用说明.md' \
  '参赛方案/参赛方案填表草稿.md' \
  '参赛方案/figures/00-总览.html'
git commit -m "docs: align proposal copy with world-model figures"
```

### Task 5: Perform visual and repository acceptance

**Files:**
- Verify: `参赛方案/figures/fig-C-四层架构总览.png`
- Verify: `参赛方案/figures/fig-D-九技能链与Fallback状态机.png`
- Verify: all files modified in Tasks 1–4.

**Interfaces:**
- Consumes: rendered assets and synchronized Markdown.
- Produces: verified commits on `feature/40-team-proposal` and matching remote SHA.

- [ ] **Step 1: Inspect both full-size and 50% renders**

Use the image viewer on both full-size PNGs. Create temporary half-size previews with:

```bash
mkdir -p /tmp/seer-figure-review
magick '参赛方案/figures/fig-C-四层架构总览.png' -resize 50% /tmp/seer-figure-review/fig-C-50.png
magick '参赛方案/figures/fig-D-九技能链与Fallback状态机.png' -resize 50% /tmp/seer-figure-review/fig-D-50.png
```

Reject and revise any render with clipped text, overlapping arrows, unreadable labels, or poor color contrast.

- [ ] **Step 2: Run terminology and resource checks**

Run:

```bash
rg -n 'VLA / 执行层|层次化 VLA 只出技能参数' \
  '参赛方案/figures' '参赛方案/参赛方案填表草稿.md'
git diff --check
```

Expected: no legacy main-layer hits and no whitespace errors.

- [ ] **Step 3: Run focused and full regression tests**

Run:

```bash
PYTHONPATH=demo python3 -m unittest tests.test_proposal_figures -v
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  PYTHONPATH=demo python3 -m unittest discover -s tests -q
```

Expected: all proposal-figure tests and the full repository suite pass.

- [ ] **Step 4: Push and verify the remote branch**

Run:

```bash
git push myfork HEAD:refs/heads/feature/40-team-proposal
test "$(git rev-parse HEAD)" = \
  "$(git ls-remote myfork refs/heads/feature/40-team-proposal | awk '{print $1}')"
```

Expected: push succeeds and local/remote SHA values match.
