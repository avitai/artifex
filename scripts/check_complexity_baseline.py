"""Check Radon complexity against a checked-in baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_PATH = REPO_ROOT / "quality" / "complexity_baseline.json"
DEFAULT_SCAN_DIRS = (REPO_ROOT / "src", REPO_ROOT / "examples")
EXAMPLE_ENFORCED_KINDS = frozenset({"function", "method"})
RANK_ORDER = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
BASELINE_VERSION = 1


@dataclass(frozen=True)
class ComplexityEntry:
    """A single Radon complexity finding."""

    path: str
    qualname: str
    kind: str
    rank: str
    complexity: int
    lineno: int
    endline: int | None

    @property
    def key(self) -> str:
        """Stable baseline key independent of line-number drift."""
        return f"{self.path}:{self.kind}:{self.qualname}"

    def to_baseline_record(self) -> dict[str, object]:
        """Convert entry to serializable baseline metadata."""
        return {
            "complexity": self.complexity,
            "endline": self.endline,
            "kind": self.kind,
            "lineno": self.lineno,
            "path": self.path,
            "qualname": self.qualname,
            "rank": self.rank,
        }


def _rank_at_least(rank: str, minimum: str) -> bool:
    return RANK_ORDER[rank] >= RANK_ORDER[minimum]


def _relative_path(path: str, *, root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.relative_to(root).as_posix()
        except ValueError:
            return candidate.as_posix()
    return candidate.as_posix()


def _is_example_entry(relative_path: str) -> bool:
    return "examples" in Path(relative_path).parts


def _is_enforced_entry(relative_path: str, kind: str) -> bool:
    if not _is_example_entry(relative_path):
        return True
    return kind in EXAMPLE_ENFORCED_KINDS


def _qualname(raw_entry: dict[str, Any]) -> str:
    class_name = raw_entry.get("classname")
    name = raw_entry["name"]
    return f"{class_name}.{name}" if class_name else name


def _run_radon(source_dirs: list[Path], *, minimum_rank: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "radon",
        "cc",
        *[str(source_dir) for source_dir in source_dirs],
        "-s",
        "-n",
        minimum_rank,
        "--json",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)
    return json.loads(result.stdout)


def _collect_entries(
    source_dirs: list[Path],
    *,
    minimum_rank: str,
) -> dict[str, ComplexityEntry]:
    raw_results = _run_radon(source_dirs, minimum_rank=minimum_rank)
    entries: dict[str, ComplexityEntry] = {}

    for raw_path, raw_entries in raw_results.items():
        relative_path = _relative_path(raw_path, root=REPO_ROOT)

        for raw_entry in raw_entries:
            rank = raw_entry["rank"]
            kind = raw_entry["type"]
            if not _rank_at_least(rank, minimum_rank) or not _is_enforced_entry(
                relative_path, kind
            ):
                continue
            entry = ComplexityEntry(
                path=relative_path,
                qualname=_qualname(raw_entry),
                kind=kind,
                rank=rank,
                complexity=int(raw_entry["complexity"]),
                lineno=int(raw_entry["lineno"]),
                endline=raw_entry.get("endline"),
            )
            entries[entry.key] = entry

    return dict(sorted(entries.items()))


def _baseline_payload(entries: dict[str, ComplexityEntry], *, minimum_rank: str) -> dict[str, Any]:
    return {
        "version": BASELINE_VERSION,
        "minimum_rank": minimum_rank,
        "scan_paths": ["src", "examples"],
        "example_enforced_kinds": sorted(EXAMPLE_ENFORCED_KINDS),
        "entries": {key: entry.to_baseline_record() for key, entry in entries.items()},
    }


def _write_baseline(path: Path, entries: dict[str, ComplexityEntry], *, minimum_rank: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _baseline_payload(entries, minimum_rank=minimum_rank)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_stdout(f"wrote {len(entries)} complexity baseline entries to {path}")


def _write_stdout(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"complexity baseline does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != BASELINE_VERSION:
        raise SystemExit(f"unsupported complexity baseline version in {path}")
    return payload


def _format_entry(entry: ComplexityEntry) -> str:
    return f"{entry.key} rank={entry.rank} complexity={entry.complexity}"


def _compare_to_baseline(current: dict[str, ComplexityEntry], baseline: dict[str, Any]) -> int:
    baseline_entries = baseline["entries"]
    new_entries = [entry for key, entry in current.items() if key not in baseline_entries]
    regressed_entries: list[tuple[ComplexityEntry, dict[str, Any]]] = []

    for key, entry in current.items():
        baseline_entry = baseline_entries.get(key)
        if baseline_entry is None:
            continue
        baseline_rank = baseline_entry["rank"]
        baseline_complexity = int(baseline_entry["complexity"])
        if RANK_ORDER[entry.rank] > RANK_ORDER[baseline_rank]:
            regressed_entries.append((entry, baseline_entry))
        elif entry.rank == baseline_rank and entry.complexity > baseline_complexity:
            regressed_entries.append((entry, baseline_entry))

    resolved_entries = [key for key in baseline_entries if key not in current]

    if new_entries or regressed_entries:
        _write_stdout("complexity baseline check failed")
        if new_entries:
            _write_stdout("\nNew C-or-worse complexity entries:")
            for entry in new_entries[:50]:
                _write_stdout(f"- {_format_entry(entry)}")
        if regressed_entries:
            _write_stdout("\nRegressed baseline entries:")
            for entry, baseline_entry in regressed_entries[:50]:
                _write_stdout(
                    f"- {entry.key} "
                    f"{baseline_entry['rank']}({baseline_entry['complexity']}) -> "
                    f"{entry.rank}({entry.complexity})"
                )
        return 1

    _write_stdout(
        "complexity baseline check passed: "
        f"{len(current)} current entries, {len(resolved_entries)} resolved baseline entries"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Path to the complexity baseline JSON file.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory to scan. Defaults to src/ and examples/.",
    )
    parser.add_argument(
        "--minimum-rank",
        choices=sorted(RANK_ORDER),
        default="C",
        help="Minimum Radon rank to baseline.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write the current scan as the baseline instead of checking it.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the complexity baseline checker."""
    args = _parse_args()
    source_dirs = args.source_dir or list(DEFAULT_SCAN_DIRS)
    entries = _collect_entries(
        [path.resolve() for path in source_dirs],
        minimum_rank=args.minimum_rank,
    )

    baseline_path = args.baseline.resolve()
    if args.write_baseline:
        _write_baseline(baseline_path, entries, minimum_rank=args.minimum_rank)
        return 0

    baseline = _load_baseline(baseline_path)
    return _compare_to_baseline(entries, baseline)


if __name__ == "__main__":
    raise SystemExit(main())
