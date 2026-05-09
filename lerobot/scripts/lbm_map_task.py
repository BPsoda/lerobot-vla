#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {i} of {path}, got {type(obj)}")
            items.append(obj)
    return items


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


_WS_RE = re.compile(r"\s+")


def _norm_task_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("\u2019", "'")
    s = _WS_RE.sub(" ", s)
    s = s.rstrip(".")
    return s


@dataclass(frozen=True)
class TaskMaps:
    simplified_to_complete: dict[str, str]


def build_task_maps(
    simplified_jsonl: Path,
    complete_jsonl: Path,
) -> TaskMaps:
    simplified = _read_jsonl(simplified_jsonl)
    complete = _read_jsonl(complete_jsonl)

    simp_tasks: list[str] = [str(r["task"]) for r in simplified if isinstance(r, dict) and "task" in r]
    comp_tasks: list[str] = [str(r["task"]) for r in complete if isinstance(r, dict) and "task" in r]
    if not simp_tasks:
        raise ValueError(f"No 'task' rows found in simplified file: {simplified_jsonl}")
    if not comp_tasks:
        raise ValueError(f"No 'task' rows found in complete file: {complete_jsonl}")
    if len(simp_tasks) != len(comp_tasks):
        raise ValueError(
            "Simplified and complete task files must have the same number of task rows "
            f"(got {len(simp_tasks)} vs {len(comp_tasks)})."
        )

    simplified_to_complete: dict[str, str] = {}

    # Assumption (per user): same task order. So pair-by-index.
    for simp, comp in zip(simp_tasks, comp_tasks, strict=True):
        simp_norm = _norm_task_text(simp)
        comp_str = str(comp)
        if simp_norm in simplified_to_complete and simplified_to_complete[simp_norm] != comp_str:
            raise ValueError(
                "Duplicate simplified task string maps to different complete tasks. "
                f"Simplified={simp!r}, existing_complete={simplified_to_complete[simp_norm]!r}, new_complete={comp_str!r}"
            )
        simplified_to_complete[simp_norm] = comp_str

    return TaskMaps(simplified_to_complete=simplified_to_complete)


def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak-{ts}")
    shutil.copy2(path, backup_path)
    return backup_path


def map_tasks_inplace(
    target_jsonl: Path,
    maps: TaskMaps,
    *,
    strict: bool,
) -> tuple[int, int, int]:
    rows = _read_jsonl(target_jsonl)

    updated = 0
    skipped = 0
    missing = 0

    for row in rows:
        if "task" not in row:
            missing += 1
            continue

        old_task = str(row["task"])
        old_norm = _norm_task_text(old_task)
        new_task = maps.simplified_to_complete.get(old_norm)

        if new_task is None:
            if strict:
                raise KeyError(
                    f"Could not map task for row with task_index={row.get('task_index')} and task={old_task!r}"
                )
            skipped += 1
            continue

        if row["task"] != new_task:
            row["task"] = new_task
            updated += 1

    _write_jsonl(target_jsonl, rows)
    return updated, skipped, missing


def map_episodes_tasks_inplace(
    episodes_jsonl: Path,
    maps: TaskMaps,
    *,
    strict: bool,
) -> tuple[int, int, int]:
    """Map each string in the ``tasks`` list (LeRobot episodes.jsonl) simplified -> complete."""
    rows = _read_jsonl(episodes_jsonl)

    updated = 0
    skipped = 0
    missing = 0

    for row in rows:
        if "tasks" not in row:
            missing += 1
            continue

        tasks_val = row["tasks"]
        if not isinstance(tasks_val, list):
            if strict:
                raise TypeError(f"Expected 'tasks' to be a list, got {type(tasks_val)}")
            missing += 1
            continue

        new_list: list[str] = []
        row_changed = False
        for old_task in tasks_val:
            if not isinstance(old_task, str):
                if strict:
                    raise TypeError(f"Expected task string in 'tasks', got {type(old_task)}")
                skipped += 1
                new_list.append(old_task)  # type: ignore[arg-type]
                continue
            old_norm = _norm_task_text(old_task)
            new_task = maps.simplified_to_complete.get(old_norm)
            if new_task is None:
                if strict:
                    raise KeyError(
                        f"Could not map episode task string {old_task!r} (episode_index={row.get('episode_index')})"
                    )
                skipped += 1
                new_list.append(old_task)
                continue
            new_list.append(new_task)
            if old_task != new_task:
                row_changed = True

        if row_changed:
            row["tasks"] = new_list
            updated += 1

    _write_jsonl(episodes_jsonl, rows)
    return updated, skipped, missing


def main() -> int:
    p = argparse.ArgumentParser(
        description="Map simplified LBM task strings to complete ones (tasks.jsonl 'task' and/or episodes.jsonl 'tasks' list), with backup."
    )
    p.add_argument(
        "--simplified",
        type=Path,
        default=Path("/cephfs/huanghaoxu/Data/LBM_lerobot_dataset/lbm-eval-iid/meta/tasks.jsonl"),
        help="Simplified tasks jsonl (task_index aligned with complete).",
    )
    p.add_argument(
        "--complete",
        type=Path,
        default=Path("/cephfs/huanghaoxu/Data/LBM_lerobot_dataset/lbm-eval-train/meta/tasks.jsonl"),
        help="Complete tasks jsonl (task_index aligned with simplified).",
    )
    p.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Optional tasks.jsonl to rewrite in-place (maps top-level 'task' field).",
    )
    p.add_argument(
        "--episodes",
        type=Path,
        default=Path("/cephfs/huanghaoxu/Data/LBM_lerobot_dataset/lbm-eval-13-iid/meta/episodes.jsonl"),
        help="Episodes jsonl to rewrite in-place (maps each string in the 'tasks' list).",
    )
    p.add_argument(
        "--no-episodes",
        action="store_true",
        help="Do not map episodes.jsonl (only use with --target).",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup (not recommended).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any row cannot be mapped.",
    )
    args = p.parse_args()

    if not args.simplified.exists():
        raise FileNotFoundError(f"Simplified file not found: {args.simplified}")
    if not args.complete.exists():
        raise FileNotFoundError(f"Complete file not found: {args.complete}")
    episodes_path: Path | None = None if args.no_episodes else args.episodes

    if args.target is not None and not args.target.exists():
        raise FileNotFoundError(f"Target file not found: {args.target}")
    if episodes_path is not None and not episodes_path.exists():
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")
    if args.target is None and episodes_path is None:
        raise ValueError("Nothing to do: pass --target and/or omit --no-episodes.")

    maps = build_task_maps(args.simplified, args.complete)

    if args.target is not None:
        backup_path: Path | None = None
        if not args.no_backup:
            backup_path = backup_file(args.target)

        updated, skipped, missing = map_tasks_inplace(args.target, maps, strict=args.strict)

        print(f"Mapped tasks written to: {args.target}")
        if backup_path is not None:
            print(f"Backup saved to:       {backup_path}")
        print(f"Rows updated: {updated}")
        print(f"Rows skipped (unmapped): {skipped}")
        print(f"Rows missing 'task' field: {missing}")

    if episodes_path is not None:
        episodes_backup_path: Path | None = None
        if not args.no_backup:
            episodes_backup_path = backup_file(episodes_path)
        ep_updated, ep_skipped, ep_missing = map_episodes_tasks_inplace(
            episodes_path, maps, strict=args.strict
        )
        if args.target is not None:
            print()
        print(f"Mapped episodes written to: {episodes_path}")
        if episodes_backup_path is not None:
            print(f"Backup saved to:           {episodes_backup_path}")
        print(f"Rows updated: {ep_updated}")
        print(f"Rows skipped (unmapped): {ep_skipped}")
        print(f"Rows missing 'tasks' field: {ep_missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
