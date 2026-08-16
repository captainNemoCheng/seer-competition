import struct
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "参赛方案" / "figures"
PROPOSAL = (
    ROOT
    / "参赛方案"
    / "【40强赛】仙工智能🤝鹰之团｜世界模型驱动的分层智能叉车卸货系统.md"
)


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
            "飞书 Aily 交互入口",
            "AgentOS 任务大脑",
            "世界模型预测层",
            "技能小脑与安全执行层",
            "显式世界模型",
            "学习型世界模型：企业数据阶段",
        ):
            self.assertIn(label, text)

    def test_figure_d_contains_nine_skills_and_three_terminal_paths(self):
        text = self.svg_text("fig-D-九技能链与Fallback状态机.svg")
        for label in (
            "进箱导航",
            "精准对位",
            "栈板识别",
            "货叉插入",
            "门架起升",
            "货叉倾斜",
            "月台区导航",
            "传送带对接",
            "栈板放置",
            "FB-01 横向修正",
            "FB-02 切换观察位姿",
            "FB-07 箱外安全停车",
            "COMPLETED · 9/9",
            "HUMAN REQUIRED",
        ):
            self.assertIn(label, text)

    def test_figure_d_feedback_route_avoids_the_subtitle_band(self):
        text = self.svg_text("fig-D-九技能链与Fallback状态机.svg")
        self.assertNotIn("V267 H255", text)
        self.assertIn("V280 H255", text)

    def test_legacy_vla_layer_copy_is_absent(self):
        paths = (
            FIGURES / "fig-C-四层架构总览.svg",
            FIGURES / "fig-D-九技能链与Fallback状态机.svg",
            FIGURES / "图片选用说明.md",
            ROOT / "参赛方案" / "参赛方案填表草稿.md",
        )
        for path in paths:
            if path.exists():
                text = path.read_text(encoding="utf-8")
                for legacy_copy in (
                    "VLA / 执行层",
                    "层次化 VLA 负责柔性决策",
                    "层次化 VLA 只出技能参数",
                    "AgentOS + 层次化 VLA",
                    "小脑：层次化 VLA",
                ):
                    self.assertNotIn(legacy_copy, text)

    def test_png_outputs_are_at_least_1600_pixels_wide(self):
        for filename in (
            "fig-C-四层架构总览.png",
            "fig-D-九技能链与Fallback状态机.png",
        ):
            data = (FIGURES / filename).read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreaterEqual(struct.unpack(">I", data[16:20])[0], 1600)

    def test_png_outputs_use_coregraphics_compatible_eight_bit_depth(self):
        for filename in (
            "fig-C-四层架构总览.png",
            "fig-D-九技能链与Fallback状态机.png",
        ):
            data = (FIGURES / filename).read_bytes()
            self.assertEqual(data[24], 8, f"{filename} must be exported with sips")

    def test_proposal_places_figure_c_before_figure_d(self):
        text = PROPOSAL.read_text(encoding="utf-8")
        figure_c = text.find("./figures/fig-C-四层架构总览.png")
        figure_d = text.find("./figures/fig-D-九技能链与Fallback状态机.png")
        self.assertGreaterEqual(figure_c, 0)
        self.assertGreater(figure_d, figure_c)


if __name__ == "__main__":
    unittest.main()
