"""A bounded empty-grasp check around the stock WALL-OSS-0.5 adapter."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from policy_space.base import PolicyModel
from policy_space.runtime.server import load_policy_model

LOG = logging.getLogger(__name__)
SHAPE = (33, 14)
CAMERA_SHAPE = (480, 640, 3)
STATE_KEYS = ("follow1_pos", "follow2_pos")
OPEN = 0.6
CLOSED = 0.25
MIN_LIFT_METERS = 0.035
MAX_WAIT_STEPS = 144
PIXEL_DRIFT = 8
MIN_MATCH = 0.88
MIN_MARGIN = 0.05
MAX_PATCH_MAE = 18.0


def _is_bottle_task(text: object) -> bool:
    if not isinstance(text, str):
        return False
    prompt = " ".join(text.casefold().split()).rstrip(".")
    return prompt in {
        "pick up the bottle and place it on the plate",
        "pick up the bottle and place it on the plate, then return the arm home",
    }


def _step(obs: dict) -> int | None:
    value = obs.get("step")
    return value if type(value) is int and value >= 0 else None


def _state(obs: dict) -> np.ndarray | None:
    state = obs.get("state") or {}
    arms = [np.asarray(state.get(key), dtype=np.float32) for key in STATE_KEYS]
    if any(arm.shape != (7,) or not np.isfinite(arm).all() for arm in arms):
        return None
    if any(not 0 <= arm[6] <= 1 for arm in arms):
        return None
    return np.concatenate(arms)


def _image(obs: dict) -> np.ndarray | None:
    frame = (obs.get("images") or {}).get("camera_front")
    if isinstance(frame, np.ndarray) and frame.shape == CAMERA_SHAPE and frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)
    return None


@dataclass(frozen=True)
class Match:
    name: str
    score: float
    x: int
    y: int
    pixels: np.ndarray

    @property
    def center(self) -> tuple[float, float]:
        h, w = self.pixels.shape[:2]
        return self.x + w / 2, self.y + h / 2


class BottleMatcher:
    def __init__(self, directory: Path):
        import cv2

        self.cv2 = cv2
        self.templates = {}
        for path in sorted(directory.glob("*.png")):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot read bottle template: {path}")
            h, w = bgr.shape[:2]
            if not (12 <= h <= 192 and 12 <= w <= 192):
                raise ValueError(f"Template dimensions must be between 12 and 192 px: {path}")
            if bgr.astype(np.float32).std(axis=(0, 1)).max() < 8:
                raise ValueError(f"Template needs visible texture or edges: {path}")
            self.templates[path.name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if not self.templates:
            raise ValueError(f"No PNG bottle crops found in {directory}")

    def _scores(self, frame: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        return self.cv2.matchTemplate(frame, pixels, self.cv2.TM_CCOEFF_NORMED)

    def locate(self, frame: np.ndarray) -> Match | None:
        candidates = []
        for name, template in self.templates.items():
            h, w = template.shape[:2]
            scores = self._scores(frame, template)
            # Keep two disjoint peaks per template so repeated objects are ambiguous.
            for _ in range(2):
                _, score, _, (x, y) = self.cv2.minMaxLoc(scores)
                if score < MIN_MATCH - MIN_MARGIN:
                    break
                candidates.append(Match(name, score, x, y, frame[y:y+h, x:x+w]))
                scores[max(0, y-h+1):y+h, max(0, x-w+1):x+w] = -1
        if not candidates:
            return None
        best = max(candidates, key=lambda candidate: candidate.score)
        if best.score < MIN_MATCH:
            return None
        bx, by = best.center
        h, w = best.pixels.shape[:2]
        for candidate in candidates:
            x, y = candidate.center
            disjoint = abs(x - bx) > w / 2 or abs(y - by) > h / 2
            if disjoint and candidate.score > best.score - MIN_MARGIN:
                return None
        return Match(best.name, best.score, best.x, best.y, best.pixels.copy())

    def remains_at_start(self, frame: np.ndarray, before: Match) -> bool:
        h, w = before.pixels.shape[:2]
        x0 = max(0, before.x - PIXEL_DRIFT)
        y0 = max(0, before.y - PIXEL_DRIFT)
        x1 = before.x + w + PIXEL_DRIFT
        y1 = before.y + h + PIXEL_DRIFT
        nearby = frame[y0:y1, x0:x1]
        scores = self._scores(nearby, before.pixels)
        _, score, _, (x, y) = self.cv2.minMaxLoc(scores)
        after = nearby[y:y+h, x:x+w]
        mae = np.abs(after.astype(np.int16) - before.pixels).mean()
        return score >= MIN_MATCH and mae <= MAX_PATCH_MAE


@dataclass(frozen=True)
class PendingGrasp:
    arm: int
    step: int
    contact_z: float
    bottle: Match


def _planned_grasp(state: np.ndarray, actions: np.ndarray) -> tuple[int, float] | None:
    if actions.shape != SHAPE or not np.isfinite(actions).all():
        return None
    choices = []
    for arm, offset in enumerate((0, 7)):
        aperture = actions[1:, offset + 6]
        closed_rows = np.flatnonzero(aperture <= CLOSED)
        if state[offset + 6] < OPEN or not closed_rows.size:
            continue
        first_close = int(closed_rows[0])
        if np.any(aperture[first_close:] >= OPEN):
            continue
        contact_z = float(actions[1 + first_close, offset + 2])
        choices.append((arm, contact_z))
    return choices[0] if len(choices) == 1 else None


def _release_at_current_pose(state: np.ndarray, arm: int) -> np.ndarray:
    actions = np.repeat(state[None, :], SHAPE[0], axis=0)
    grip_index = arm * 7 + 6
    actions[1:7, grip_index] = np.linspace(state[grip_index], 1, 7)[1:]
    actions[7:, grip_index] = 1
    return actions


class Model(PolicyModel):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.mode = os.getenv("BOTTLE_MODE", "baseline")
        if self.mode not in {"baseline", "shadow", "recover"}:
            raise ValueError("BOTTLE_MODE must be baseline, shadow, or recover")
        self.matcher = None
        if self.mode != "baseline":
            templates = os.getenv("BOTTLE_TEMPLATES")
            if not templates:
                raise ValueError("Set BOTTLE_TEMPLATES to a directory of calibrated PNG crops")
            self.matcher = BottleMatcher(Path(templates).expanduser())

        path = os.getenv("WALLX_BASE_DEPLOY")
        if not path:
            raise ValueError("Set WALLX_BASE_DEPLOY to the official Arm-6A deploy.yaml")
        self.base, metadata = load_policy_model(path)
        expected = {
            "policy_name": "wallx",
            "action_dim": SHAPE[1],
            "action_horizon": SHAPE[0],
            "action_mode": "ik_eef_abs",
            "gripper_convention": "normalized_01",
            "supported_schema_pairs": [{
                "observation_schema": "bimanual_rgb_ee@1",
                "action_schema": "bimanual_ee_absolute@1",
                "embodiment_constraints": ["ex001_6r@1"],
            }],
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            self.base.close()
            raise ValueError("WALLX_BASE_DEPLOY must point to the stock Arm-6A EEF policy")
        if metadata.get("capabilities", {}).get("wall_x_rtc", {}).get("enabled"):
            self.base.close()
            raise ValueError("Use the stock standard backend with RTC disabled")
        self._clear()

    def _clear(self) -> None:
        self.obs = None
        self.instruction = ""
        self.pending: PendingGrasp | None = None
        self.last_step: int | None = None
        self.latest_actions = None
        self.flags = 0
        self.recoveries = 0

    def initialize_episode(self, episode_info: dict | None = None) -> None:
        self._clear()
        info = episode_info or {}
        self.instruction = info.get("instruction") or ""
        self.base.initialize_episode(info)

    def ingest_observation(self, obs: dict) -> None:
        self.obs = obs
        self.base.ingest_observation(obs)

    def _emit(self, event: str, **fields) -> None:
        LOG.info("bottle_recovery %s", json.dumps({
            "event": event, "mode": self.mode, "step": _step(self.obs or {}),
            "episode_id": (self.obs or {}).get("episode_id"), **fields,
        }, ensure_ascii=False))

    def _review(self, step: int, state: np.ndarray, frame: np.ndarray) -> int | None:
        grasp = self.pending
        if grasp is None or step <= grasp.step:
            return None
        if step - grasp.step > MAX_WAIT_STEPS:
            self.pending = None
            return None
        offset = 7 * grasp.arm
        if state[offset + 6] >= OPEN:
            self.pending = None
            return None
        if state[offset + 6] > CLOSED or state[offset + 2] < grasp.contact_z + MIN_LIFT_METERS:
            return None
        self.pending = None
        if not self.matcher.remains_at_start(frame, grasp.bottle):
            self._emit("inconclusive_after_lift", arm=grasp.arm)
            return None
        self.flags += 1
        self._emit("suspected_empty_grasp", arm=grasp.arm, template=grasp.bottle.name)
        return grasp.arm

    def infer_actions(self) -> np.ndarray:
        obs = self.obs or {}
        if self.matcher is None or not _is_bottle_task(obs.get("instruction") or self.instruction):
            self.pending = None
            return self.base.infer_actions()

        step, state, frame = _step(obs), _state(obs), _image(obs)
        if step is None or state is None or frame is None:
            self.pending = None
            return self.base.infer_actions()
        if step == self.last_step and self.latest_actions is not None:
            return self.latest_actions.copy()
        if self.last_step is not None and step < self.last_step:
            self.pending = None
            return self.base.infer_actions()

        arm = self._review(step, state, frame)
        if arm is not None and self.mode == "recover" and not self.recoveries:
            actions = _release_at_current_pose(state, arm)
            self.recoveries += 1
            self._emit("release_and_replan", arm=arm)
        else:
            actions = self.base.infer_actions()
            if self.pending is None and (self.mode == "shadow" or not self.recoveries):
                grasp = _planned_grasp(state, actions)
                if grasp is not None:
                    arm, contact_z = grasp
                    bottle = self.matcher.locate(frame)
                    if bottle is not None:
                        self.pending = PendingGrasp(arm, step, contact_z, bottle)
                        self._emit("grasp_watch_started", arm=arm, template=bottle.name)
        self.last_step = step
        self.latest_actions = actions.copy()
        return actions

    def finalize_episode(self, episode_info: dict | None = None) -> None:
        try:
            self.base.finalize_episode(episode_info)
            if self.matcher is not None:
                self._emit("episode_end", flags=self.flags, recoveries=self.recoveries)
        finally:
            self._clear()

    def close(self) -> None:
        self.base.close()
