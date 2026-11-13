"""Build a local LeRobot v3 bottle dataset from pinned X2Real demos."""

import argparse
import json
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


REVISION = "5a741385f9f196073cfe163b0f25033025e383a6"
BASE = (
    "https://huggingface.co/datasets/x-square-robot/X2Real-Dataset/resolve/"
    f"{REVISION}/data/x2real-desktop-ee-lerobot-v2"
)
TASK = "Pick up the bottle and place it on the plate, then return the arm home."
VECTORS = ["observation.state", "action"]
CAMERAS = [f"observation.images.{name}Img" for name in ("face", "left", "right")]
FPS = 20


def fetch(path, cache):
    local = cache / path
    if not local.is_file():
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["curl", "-fsSL", "--retry", "3", "--max-time", "180", f"{BASE}/{path}",
             "-o", str(local.with_suffix(".part"))],
            check=True,
        )
        local.with_suffix(".part").replace(local)
    return local


def download_episode(episode, cache):
    index = episode["episode_index"]
    chunk = f"chunk-{index // 1000:03d}"
    name = f"episode_{index:06d}"
    table = fetch(f"data/{chunk}/{name}.parquet", cache)
    videos = {key: fetch(f"videos/{chunk}/{key}/{name}.mp4", cache) for key in CAMERAS}
    return table, videos


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_parquet(path, table, *, index=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=index)


def file_stats(values):
    return {**{name: fn(values, axis=0).tolist() for name, fn in (
        ("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std),
    )}, "count": [len(values)]}


def convert(output, cache, count):
    if output.exists():
        raise FileExistsError(output)
    info = json.loads(fetch("meta/info.json", cache).read_text())
    episodes = [json.loads(line) for line in fetch("meta/episodes.jsonl", cache).read_text().splitlines()]
    eligible = [episode for episode in episodes if episode["tasks"] == [TASK]]
    if not 2 <= count <= len(eligible):
        raise ValueError(f"Choose 2–{len(eligible)} episodes")
    # Deterministic order gives the trainer a disjoint episode holdout at the tail.
    random.Random(42).shuffle(eligible)
    selected = eligible[:count]
    with ThreadPoolExecutor(max_workers=8) as pool:
        sources = list(pool.map(lambda episode: download_episode(episode, cache), selected))

    staging = output.with_name(output.name + ".building")
    staging.mkdir(parents=True)
    tables = []
    metadata = []
    offset = 0
    try:
        for index, (episode, (table, videos)) in enumerate(zip(selected, sources, strict=True)):
            data = pd.read_parquet(table)
            frames = episode["length"]
            if (len(data) != frames or set(data["episode_index"]) != {episode["episode_index"]}
                    or set(data["annotation.task"]) != {TASK}
                    or not np.array_equal(data["frame_index"], np.arange(frames))
                    or not np.allclose(data["timestamp"], np.arange(frames) / FPS, atol=1e-5)):
                raise ValueError(f"Invalid episode alignment: {table}")
            stats = {}
            for key in VECTORS:
                values = np.stack(data[key])
                if (values.shape != (frames, 20) or not np.isfinite(values).all()
                        or (values[:, [9, 19]] < -1e-5).any()
                        or (values[:, [9, 19]] > 1.00001).any()):
                    raise ValueError(f"Invalid 20D normalized {key}: {table}")
                stats[key] = file_stats(values)
            data = data.drop(columns="annotation.task").assign(
                timestamp=(np.arange(frames) / FPS).astype(np.float32),
                episode_index=index, task_index=0, index=np.arange(offset, offset + frames),
            )
            tables.append(data)
            entry = {
                "episode_index": index, "tasks": [TASK], "length": frames,
                "dataset_from_index": offset, "dataset_to_index": offset + frames,
                "data/chunk_index": 0, "data/file_index": 0,
                "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
            }
            for key, metrics in stats.items():
                entry.update({f"stats/{key}/{name}": value for name, value in metrics.items()})
            for key, source in videos.items():
                dest = staging / f"videos/{key}/chunk-000/file-{index:03d}.mp4"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                prefix = f"videos/{key}"
                entry.update({
                    f"{prefix}/chunk_index": 0, f"{prefix}/file_index": index,
                    f"{prefix}/from_timestamp": 0.0, f"{prefix}/to_timestamp": frames / FPS,
                })
            metadata.append(entry)
            offset += frames

        combined = pd.concat(tables, ignore_index=True)
        write_parquet(staging / "data/chunk-000/file-000.parquet", combined)
        write_parquet(staging / "meta/episodes/chunk-000/file-000.parquet", pd.DataFrame(metadata))
        tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task"))
        write_parquet(staging / "meta/tasks.parquet", tasks, index=True)
        write_json(staging / "meta/stats.json", {key: file_stats(np.stack(combined[key])) for key in VECTORS})
        features = {key: spec for key, spec in info["features"].items() if key != "annotation.task"}
        for key, dtype in (("timestamp", "float32"), ("frame_index", "int64"),
                           ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
            features[key] = {"dtype": dtype, "shape": [1], "names": None}
        for key in CAMERAS:
            features[key]["info"] = features[key].pop("video_info")
        write_json(staging / "meta/info.json", {
            "codebase_version": "v3.0", "robot_type": "x2robot_dreamzero_dual_arm",
            "fps": FPS, "total_episodes": count, "total_frames": offset, "total_tasks": 1,
            "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200,
            "splits": {"train": f"0:{count}"}, "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        })
        write_json(staging / "source.json", {
            "revision": REVISION, "source_episode_indices": [e["episode_index"] for e in selected],
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f"Converted {count} episodes / {offset} frames into {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--episodes", default=300, type=int)
    args = parser.parse_args()
    convert(args.output, args.cache, args.episodes)
