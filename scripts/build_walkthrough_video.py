"""検証済みレポートから、音声不要の字幕付き短編動画を生成する。"""

# ruff: noqa: E501 - ASSの固定フォーマット行は改行すると動画生成時の意味が変わる。

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SUBTITLES = ROOT / "docs" / "video" / "ai-support-triage-walkthrough.srt"
OUTPUT = ROOT / "docs" / "video" / "ai-support-triage-walkthrough.mp4"
SLIDE_CAPTIONS = ROOT / "docs" / "video" / "ai-support-triage-walkthrough.ass"


@dataclass(frozen=True, slots=True)
class Slide:
    eyebrow: str
    title: tuple[str, ...]
    detail: tuple[str, ...]
    proof: str
    seconds: int


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise SystemExit(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], parsed)


def _slides() -> tuple[Slide, ...]:
    retrieval = _load_json(ROOT / "reports" / "retrieval-hybrid.json")
    evaluation = _load_json(ROOT / "reports" / "eval-report.json")
    failures = _load_json(ROOT / "reports" / "failure-demo.json")
    retrieval_summary = retrieval.get("summary", retrieval)
    evaluation_summary = evaluation.get("summary", evaluation)

    if failures["summary"] != {"passed": 5, "total": 5}:
        raise SystemExit("Failure report is not 5/5; refusing to publish the video.")
    if evaluation_summary.get("passed") != 8:
        raise SystemExit("End-to-end evaluation is not 8/8; refusing to publish the video.")

    return (
        Slide(
            "PORTFOLIO WALKTHROUGH",
            ("AI SUPPORT", "TRIAGE"),
            ("Evidence-grounded automation", "with deterministic safety boundaries"),
            "No paid API key required for this demonstration",
            14,
        ),
        Slide(
            "ARCHITECTURE",
            ("TENANT BOUNDARY", "BEFORE RETRIEVAL"),
            (
                "Credential → tenant-scoped data → evidence",
                "LLM proposal → policy gate → audit trace",
            ),
            "Cross-tenant object access returns 404",
            20,
        ),
        Slide(
            "MEASURED RETRIEVAL",
            ("HYBRID IMPROVES", "THE FIXED TEST SET"),
            (
                f"Recall@4  {retrieval_summary['recall_at_k']:.2f}     "
                f"MRR  {retrieval_summary['mrr']:.2f}",
                f"Unsupported rejection  {retrieval_summary['empty_result_accuracy']:.2f}",
            ),
            "13 labelled cases · BM25 remains the default",
            20,
        ),
        Slide(
            "FAILURE PROOF",
            ("5 / 5 SCENARIOS", "PASS THROUGH THE API"),
            (
                "Outage · malformed output · unsupported query",
                "duplicate delivery · cross-tenant access",
            ),
            "Finite retry and repair budgets — no unbounded agent loop",
            22,
        ),
        Slide(
            "SAFE OUTCOMES",
            ("REUSE, REFUSE,", "OR ROUTE TO A HUMAN"),
            (
                "Duplicates reuse one persisted run",
                "Missing evidence never becomes a confident answer",
            ),
            "The model advises; deterministic code decides",
            22,
        ),
        Slide(
            "VERIFIABLE DELIVERY",
            ("46 TESTS", "TWO DATABASE ENGINES"),
            (
                f"End-to-end evaluation  {evaluation_summary['passed']}/"
                f"{evaluation_summary['cases']}",
                "Seeded defects detected  4/4",
            ),
            "Public reports and honest limits are linked in the repository",
            22,
        ),
    )


def _ass_time(seconds: int) -> str:
    minutes, remainder = divmod(seconds, 60)
    return f"0:{minutes:02d}:{remainder:02d}.00"


def _ass_text(value: str | tuple[str, ...]) -> str:
    lines = (value,) if isinstance(value, str) else value
    return r"\N".join(line.replace("{", r"\{").replace("}", r"\}") for line in lines)


def _build_ass(slides: tuple[Slide, ...]) -> str:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Eyebrow,Arial,28,&H00F9E867,&H00FFFFFF,&H0008111F,&H00000000,-1,0,0,0,100,100,3,0,1,0,0,7,0,0,0,1
Style: Title,Arial,92,&H00FCFAF8,&H00FFFFFF,&H0008111F,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Detail,Arial,36,&H00E1D5CB,&H00FFFFFF,&H0008111F,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Proof,Arial,29,&H00F0E8E2,&H00FFFFFF,&H0038210F,&H0038210F,0,0,0,0,100,100,0,0,3,1,0,7,0,0,0,1
Style: Footer,Arial,24,&H008B7464,&H00FFFFFF,&H0008111F,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    events: list[str] = []
    start = 0
    for index, slide in enumerate(slides, start=1):
        end = start + slide.seconds
        timing = f"{_ass_time(start)},{_ass_time(end)}"
        events.extend(
            (
                f"Dialogue: 0,{timing},Eyebrow,,0,0,0,,{{\\pos(140,165)}}{_ass_text(slide.eyebrow)}",
                f"Dialogue: 0,{timing},Title,,0,0,0,,{{\\pos(140,300)}}{_ass_text(slide.title)}",
                f"Dialogue: 0,{timing},Detail,,0,0,0,,{{\\pos(145,590)}}{_ass_text(slide.detail)}",
                f"Dialogue: 0,{timing},Proof,,0,0,0,,{{\\pos(170,710)}}{_ass_text(slide.proof)}",
                f"Dialogue: 0,{timing},Footer,,0,0,0,,{{\\pos(140,1000)}}github.com/yuten0901/ai-support-triage",
                f"Dialogue: 0,{timing},Footer,,0,0,0,,{{\\pos(1740,1000)}}{index:02d} / 06",
            )
        )
        start = end
    return header + "\n".join(events) + "\n"


def main() -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required to build the walkthrough video.")
    if not SUBTITLES.is_file():
        raise SystemExit(f"Missing subtitles: {SUBTITLES}")

    slides = _slides()
    if sum(slide.seconds for slide in slides) != 120:
        raise SystemExit("The walkthrough must remain exactly 120 seconds.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SLIDE_CAPTIONS.write_text(_build_ass(slides), encoding="utf-8")
    subtitle_filter = (
        "subtitles=docs/video/ai-support-triage-walkthrough.ass,"
        "subtitles=docs/video/ai-support-triage-walkthrough.srt:"
        "force_style='FontName=Arial,FontSize=9,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H80000000,BorderStyle=3,Outline=1,Shadow=0,"
        "MarginL=120,MarginR=120,MarginV=48,Alignment=2'"
    )
    subprocess.run(  # noqa: S603 - 実行ファイルと引数はすべて固定または検証済み。
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=#08111f:s=1920x1080:d=120",
                "-vf",
                subtitle_filter,
                "-r",
                "30",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(OUTPUT),
            ],
            cwd=ROOT,
            check=True,
        )
    print(f"Built {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
