"""
video_utils.py  --  Data loading & visualization helpers for shot boundary detection.

Handles:
  - Loading ground-truth JSON files
  - Extracting frames from videos via OpenCV
  - Converting ground-truth ad boundaries to frame indices
  - Plotting detected vs ground-truth boundaries
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths (adjust if your layout differs)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = PROJECT_DIR / "sample_videos" / "videos"
INFO_DIR = PROJECT_DIR / "sample_videos" / "video_info"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AdSegment:
    """A single ad insertion in the final (combined) video."""
    ad_index: int
    ad_filename: str
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass
class TimelineSegment:
    """One segment (either 'video_content' or 'ad') in the timeline."""
    seg_type: str            # "video_content" or "ad"
    start_sec: float
    end_sec: float
    duration_sec: float
    ad_index: Optional[int] = None
    ad_filename: Optional[str] = None


@dataclass
class VideoInfo:
    """All metadata for a single test video."""
    video_filename: str
    video_path: Path
    json_path: Path
    duration_sec: float
    resolution: Tuple[int, int]       # (width, height)
    ads: List[AdSegment] = field(default_factory=list)
    timeline: List[TimelineSegment] = field(default_factory=list)
    fps: Optional[float] = None       # populated on first frame extraction


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_video_info(video_name: str) -> VideoInfo:
    """Load metadata + ground truth for a video (e.g. 'test_001')."""
    json_path = INFO_DIR / f"{video_name}.json"
    video_path = VIDEOS_DIR / f"{video_name}.mp4"

    with open(json_path, "r") as f:
        data = json.load(f)

    res_str = data.get("original_video_resolution", "0x0")
    w, h = (int(x) for x in res_str.split("x"))

    ads = [
        AdSegment(
            ad_index=a["ad_index"],
            ad_filename=a["ad_filename"],
            start_sec=a["final_video_ad_start_seconds"],
            end_sec=a["final_video_ad_end_seconds"],
            duration_sec=a["ad_duration_seconds"],
        )
        for a in data.get("inserted_ads", [])
    ]

    timeline = [
        TimelineSegment(
            seg_type=s["type"],
            start_sec=s["final_video_start_seconds"],
            end_sec=s["final_video_end_seconds"],
            duration_sec=s["duration_seconds"],
            ad_index=s.get("ad_index"),
            ad_filename=s.get("ad_filename"),
        )
        for s in data.get("timeline_segments", [])
    ]

    return VideoInfo(
        video_filename=data["output_filename"],
        video_path=video_path,
        json_path=json_path,
        duration_sec=data["output_duration_seconds"],
        resolution=(w, h),
        ads=ads,
        timeline=timeline,
    )


def list_available_videos() -> List[str]:
    """Return base names like ['test_001', 'test_002', ...] for all videos with JSON."""
    names = sorted(
        p.stem for p in INFO_DIR.glob("*.json")
        if (VIDEOS_DIR / f"{p.stem}.mp4").exists()
    )
    return names


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def get_video_fps(video_path: Path | str) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_video_frame_count(video_path: Path | str) -> int:
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def extract_frames(
    video_path: Path | str,
    *,
    sample_every: int = 1,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    grayscale: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Extract frames from a video file.

    Parameters
    ----------
    video_path : path to the .mp4
    sample_every : keep every N-th frame (1 = all frames)
    max_frames : stop after this many extracted frames (None = no limit)
    start_frame : first frame index to read
    grayscale : convert to single-channel grayscale

    Returns
    -------
    frames : np.ndarray of shape (N, H, W, 3) or (N, H, W) if grayscale
    fps : frames per second of the source video
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    idx = start_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - start_frame) % sample_every == 0:
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    return np.array(frames), fps


def frame_generator(
    video_path: Path | str,
    *,
    sample_every: int = 1,
    grayscale: bool = False,
) -> "Generator[Tuple[int, np.ndarray], None, None]":
    """
    Yield (frame_index, frame_array) one at a time -- memory-friendly for long videos.
    """
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield idx, frame
        idx += 1
    cap.release()


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------
def get_ground_truth_boundaries(info: VideoInfo, fps: float) -> List[int]:
    """
    Return frame indices where the ground-truth ad boundaries occur.

    Each ad contributes TWO boundaries: the first frame of the ad segment
    and the first frame of the content segment that follows. We add +1
    because the boundary timestamp is shared between the outgoing and
    incoming segments (end of segment N == start of segment N+1), so +1
    ensures we point at the first frame of the *new* segment.
    """
    boundaries = []
    for ad in info.ads:
        boundaries.append(int(round(ad.start_sec * fps)) + 2)
        boundaries.append(int(round(ad.end_sec * fps)) + 2)
    return sorted(boundaries)


def get_ground_truth_ad_intervals(info: VideoInfo, fps: float) -> List[Tuple[int, int]]:
    """Return (start_frame, end_frame) for every ad segment.

    start_frame is the first frame of the ad (+2 offset),
    end_frame is the first frame of the resumed content (+2 offset).
    """
    return [
        (int(round(ad.start_sec * fps)) + 2, int(round(ad.end_sec * fps)) + 2)
        for ad in info.ads
    ]


def seconds_to_timestamp(sec: float) -> str:
    """Convert seconds to HH:MM:SS.mmm string."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_signal_with_boundaries(
    signal: np.ndarray,
    detected_frames: List[int],
    ground_truth_frames: List[int],
    *,
    fps: float = 30.0,
    title: str = "Shot Boundary Detection",
    ylabel: str = "Score",
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (18, 5),
) -> plt.Figure:
    """
    Plot a 1-D signal (e.g. frame-difference scores) with detected and GT boundaries
    overlaid as vertical lines.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(signal))
    ax.plot(x, signal, linewidth=0.5, color="steelblue", label="Signal")

    for i, f in enumerate(ground_truth_frames):
        ax.axvline(f, color="green", alpha=0.7, linestyle="--", linewidth=1.2,
                    label="Ground Truth" if i == 0 else None)

    for i, f in enumerate(detected_frames):
        ax.axvline(f, color="red", alpha=0.6, linestyle="-", linewidth=1.0,
                    label="Detected" if i == 0 else None)

    if threshold is not None:
        ax.axhline(threshold, color="orange", linestyle=":", linewidth=1, label="Threshold")

    ax.set_xlabel("Frame index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_boundary_comparison(
    detected_frames: List[int],
    ground_truth_frames: List[int],
    total_frames: int,
    *,
    fps: float = 30.0,
    title: str = "Detected vs Ground-Truth Boundaries",
    tolerance_frames: int = 15,
    figsize: Tuple[int, int] = (18, 3),
) -> plt.Figure:
    """
    Timeline strip showing detected (red ticks) vs ground-truth (green ticks).
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hlines(0, 0, total_frames, colors="lightgray", linewidth=6)

    for i, f in enumerate(ground_truth_frames):
        ax.plot(f, 0.15, "v", color="green", markersize=10,
                label="Ground Truth" if i == 0 else None)

    for i, f in enumerate(detected_frames):
        ax.plot(f, -0.15, "^", color="red", markersize=8,
                label="Detected" if i == 0 else None)

    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(0, total_frames)
    ax.set_xlabel("Frame index")
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def show_frames_at_indices(
    video_path: Path | str,
    frame_indices: List[int],
    *,
    label: str = "Frame",
    max_show: int = 12,
    figsize_per_frame: Tuple[float, float] = (3, 2.5),
) -> plt.Figure:
    """Display thumbnail frames at the given indices."""
    indices = frame_indices[:max_show]
    n = len(indices)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.text(0.5, 0.5, "No frames to show", ha="center", va="center")
        ax.axis("off")
        return fig

    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(figsize_per_frame[0] * cols,
                                       figsize_per_frame[1] * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    cap = cv2.VideoCapture(str(video_path))
    for i, fidx in enumerate(indices):
        r, c = divmod(i, cols)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[r, c].imshow(frame)
        axes[r, c].set_title(f"{label} {fidx}", fontsize=9)
        axes[r, c].axis("off")
    cap.release()

    # hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.tight_layout()
    return fig
