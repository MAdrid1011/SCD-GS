"""Fail when the public tree contains private assets or retired architecture material."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ("hpca2027-latex-template/", "GCC.pdf", "REACT3D.pdf")
# These are retired source trees and local design inputs, not ordinary build
# outputs. Their presence is a release blocker even when .gitignore hides them.
FORBIDDEN_WORKTREE_PATHS = (
    "baselines",
    "benchmarks",
    "data",
    "logs",
    "patches",
    "scripts",
    "simulator",
    "rtl/verilog",
    "physical",
    "point_cloud",
    "related-work",
    "draft",
    "third_party/tools",
    "third_party/asap7_sram_0p0",
    ".gitmodules",
    "SCD-GS.md",
    "SCD_GS_Data.md",
    "input.ply",
)
FORBIDDEN_TERMS = (
    "UCN",
    "ICE",
    "ILSF",
    "CGS",
    "Salience",
    "ASC",
    "RSB",
    "CSS",
    "UpdateDescriptor",
    "LiveStateFabric",
    "QUEEN",
    "HiCoM",
    "3DGStream",
    "4DGC",
    "RenderBackend",
    "TrainingBackend",
    "RenderLaneArray",
    "TrainingLaneArray",
    "RenderFrontEnd",
    "ScdGsRtlSpec",
    "EmitScdGsPhysicalTop",
    "Mode 0",
    "Mode 1",
    "Mode 2",
    "Mode 3",
)
TEXT_SUFFIXES = {".md", ".scala", ".py", ".trace", ".drawio", ".svg"}
SCAN_ROOTS = ("README.md", "docs", "rtl/src", "reference_model", "tools")
SELF_REFERENTIAL_FILES = {"tools/check_public_tree.py"}
REQUIRED = (
    "README.md",
    "LICENSE",
    "NOTICE",
    "docs/mechanism-contract.md",
    "docs/traceability.md",
    "docs/trace-format.md",
    "docs/reproducibility.md",
    "docs/fixtures/six_frame.scdgs-trace",
    "docs/mechanism/00_paper_contract.md",
    "docs/mechanism/01_canonical_update_and_trace.md",
    "docs/mechanism/02_siu.md",
    "docs/mechanism/03_fru.md",
    "docs/mechanism/04_ciu.md",
    "docs/mechanism/05_tru.md",
    "docs/mechanism/06_tuu.md",
    "docs/mechanism/07_memory_and_frame_protocol.md",
    "docs/mechanism/08_verification_matrix.md",
    "docs/mechanism/09_evaluation_scope.md",
    "rtl/src/main/scala/scdgs/siu/SplatImpactUnit.scala",
    "rtl/src/main/scala/scdgs/fru/FusedResidencyUnit.scala",
    "rtl/src/main/scala/scdgs/ciu/ConditionedIssueUnit.scala",
    "rtl/src/main/scala/scdgs/tru/TieredRenderUnit.scala",
    "rtl/src/main/scala/scdgs/tuu/TrainingUpdateUnit.scala",
    "rtl/src/main/scala/scdgs/bundle/ScdGsBundles.scala",
    "rtl/src/main/scala/scdgs/config/ScdGsParams.scala",
    "rtl/src/main/scala/scdgs/fp/Fp32.scala",
    "rtl/src/main/scala/scdgs/math/SqrtF32.scala",
    "rtl/src/main/scala/scdgs/memory/StateSpm.scala",
    "rtl/src/main/scala/scdgs/memory/SplatSpm.scala",
    "rtl/src/main/scala/scdgs/memory/MetaSpm.scala",
    "rtl/src/main/scala/scdgs/memory/FrameSpm.scala",
    "rtl/src/main/scala/scdgs/top/ScdGsTop.scala",
    "rtl/src/main/scala/scdgs/trace/ReplayTrace.scala",
    "reference_model/contracts.py",
    "reference_model/siu.py",
    "reference_model/fru.py",
    "reference_model/ciu.py",
    "reference_model/tru.py",
    "reference_model/tuu.py",
    "reference_model/tests/test_mechanisms.py",
    "rtl/src/test/scala/scdgs/siu/SiuSpec.scala",
    "rtl/src/test/scala/scdgs/config/ScdGsParamsSpec.scala",
    "rtl/src/test/scala/scdgs/fp/Fp32Spec.scala",
    "rtl/src/test/scala/scdgs/fru/FruSpec.scala",
    "rtl/src/test/scala/scdgs/memory/MemorySpec.scala",
    "rtl/src/test/scala/scdgs/ciu/CiuSpec.scala",
    "rtl/src/test/scala/scdgs/tru/TruSpec.scala",
    "rtl/src/test/scala/scdgs/tuu/TuuSpec.scala",
    "rtl/src/test/scala/scdgs/top/ScdGsLoopSpec.scala",
    "rtl/src/test/scala/scdgs/top/FixtureReplaySpec.scala",
    "rtl/src/test/scala/scdgs/trace/TraceReplaySpec.scala",
    "rtl/src/test/scala/scdgs/emit/GeneratedRtlSpec.scala",
    "tools/tests/README.md",
    "tools/tests/test_export_trace.md",
    "tools/tests/test_export_trace.py",
    "rtl/src/main/scala/scdgs/README.md",
    "rtl/src/main/scala/scdgs/bundle/README.md",
    "rtl/src/main/scala/scdgs/config/README.md",
    "rtl/src/main/scala/scdgs/siu/README.md",
    "rtl/src/main/scala/scdgs/fru/README.md",
    "rtl/src/main/scala/scdgs/ciu/README.md",
    "rtl/src/main/scala/scdgs/tru/README.md",
    "rtl/src/main/scala/scdgs/tuu/README.md",
    "rtl/src/main/scala/scdgs/memory/README.md",
    "rtl/src/main/scala/scdgs/top/README.md",
    "rtl/src/main/scala/scdgs/trace/README.md",
)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)


def tracked_files() -> list[str]:
    result = run("git", "ls-files")
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return [line for line in result.stdout.splitlines() if line]


def is_tracked(path: str, files: list[str]) -> bool:
    """Return whether a file or any descendant of a directory is tracked."""

    normalized = path.rstrip("/")
    return any(file == normalized or file.startswith(f"{normalized}/") for file in files)


def exists_in_worktree(path: str) -> bool:
    """Include dangling symlinks so a forbidden path cannot hide behind one."""

    candidate = ROOT / path.rstrip("/")
    return candidate.exists() or candidate.is_symlink()


def is_ignored(path: str) -> bool:
    return run("git", "check-ignore", "-q", "--no-index", "--", path).returncode == 0


def contains_forbidden_term(value: str) -> str | None:
    """Find retired terminology in text or a path component, case-insensitively."""

    normalized = re.sub(r"[._/\\-]+", " ", value)
    for term in FORBIDDEN_TERMS:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", normalized, re.IGNORECASE):
            return term
    return None


def public_text_files() -> list[Path]:
    """Return public, human-readable files without recursing into private assets."""

    paths: list[Path] = []
    for relative_root in SCAN_ROOTS:
        root = ROOT / relative_root
        if root.is_file():
            candidates = (root,)
        elif root.is_dir():
            candidates = (path for path in root.rglob("*") if path.is_file())
        else:
            continue
        for path in candidates:
            relative = path.relative_to(ROOT).as_posix()
            if relative in SELF_REFERENTIAL_FILES:
                continue
            if path.suffix in TEXT_SUFFIXES or path.name == "six_frame.scdgs-trace":
                paths.append(path)
    return paths


def main() -> int:
    failures: list[str] = []
    files = tracked_files()
    for path in PRIVATE:
        if not is_ignored(path):
            failures.append(f"private path is not ignored: {path}")
        if is_tracked(path, files):
            failures.append(f"private path is tracked: {path}")

    for path in FORBIDDEN_WORKTREE_PATHS:
        if exists_in_worktree(path):
            failures.append(f"retired material remains in the working tree: {path}")
        if is_tracked(path, files):
            failures.append(f"retired public path remains tracked: {path}")

    private_names = {Path(path).name for path in PRIVATE if path.endswith(".pdf")}
    for path in ROOT.glob("*.pdf"):
        if path.name not in private_names:
            failures.append(f"unexpected root PDF is present: {path.name}")

    for required in REQUIRED:
        if required not in files:
            failures.append(f"required public evidence is not tracked: {required}")
        elif not (ROOT / required).is_file():
            failures.append(f"required public evidence is absent from the working tree: {required}")

    for path in public_text_files():
        relative = path.relative_to(ROOT).as_posix()
        term = contains_forbidden_term(relative)
        if term:
            failures.append(f"retired term {term!r} remains in public path {relative}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        term = contains_forbidden_term(text)
        if term:
            failures.append(f"retired term {term!r} remains in {relative}")
    if failures:
        print("public-tree audit failed:", file=sys.stderr)
        print("\n".join(f"- {failure}" for failure in failures), file=sys.stderr)
        return 1
    print("public-tree audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
