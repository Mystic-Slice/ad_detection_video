"""
shot_detection.py  --  Pluggable shot-boundary detection methods.

Every detector subclasses ``ShotBoundaryDetector`` and implements:
    compute_scores(video_path, ...) -> np.ndarray   (per-frame score)
    detect(video_path, ...)         -> DetectionResult

Add new methods by subclassing and implementing ``compute_scores``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class DetectionResult:
    """Returned by every detector."""
    method_name: str
    scores: np.ndarray                   # one value per sampled frame
    detected_frames: List[int]           # frame indices flagged as boundaries
    threshold: float                     # threshold that was used
    sample_every: int = 1               # sampling stride used


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class ShotBoundaryDetector(abc.ABC):
    """Interface every detector must satisfy."""

    name: str = "base"

    @abc.abstractmethod
    def compute_scores(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
    ) -> np.ndarray:
        """Return a 1-D array of per-frame dissimilarity scores."""
        ...

    def detect(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
        threshold: Optional[float] = None,
        adaptive: bool = True,
        adaptive_k: float = 3.0,
        min_gap_frames: int = 10,
    ) -> DetectionResult:
        """
        Full pipeline: compute scores -> threshold -> return boundaries.

        Parameters
        ----------
        threshold : fixed threshold; if None and adaptive=True, use mean + k*std
        adaptive : use adaptive thresholding when threshold is None
        adaptive_k : multiplier for std in adaptive threshold
        min_gap_frames : suppress detections closer than this many (sampled) frames
        """
        scores = self.compute_scores(video_path, sample_every=sample_every)

        if threshold is None and adaptive:
            threshold = float(np.mean(scores) + adaptive_k * np.std(scores))
        elif threshold is None:
            threshold = float(np.percentile(scores, 95))

        # Find peaks above threshold
        candidates = np.where(scores > threshold)[0]

        # Non-max suppression: keep only local maxima with min gap
        detected: List[int] = []
        for c in candidates:
            if len(detected) == 0 or (c - detected[-1]) >= min_gap_frames:
                detected.append(int(c))
            elif scores[c] > scores[detected[-1]]:
                detected[-1] = int(c)

        # Map sampled indices back to original frame indices
        detected_original = [d * sample_every for d in detected]

        return DetectionResult(
            method_name=self.name,
            scores=scores,
            detected_frames=detected_original,
            threshold=threshold,
            sample_every=sample_every,
        )


# ===================================================================
# Concrete detectors
# ===================================================================

class FrameDifferenceDetector(ShotBoundaryDetector):
    """
    Pixel-wise absolute difference between consecutive frames.

    Simple, fast, and effective for hard cuts.
    """

    name = "Frame Difference (pixel MAD)"

    def compute_scores(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        prev_frame = None
        scores = []
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if prev_frame is not None:
                    diff = np.mean(np.abs(gray - prev_frame))
                    scores.append(diff)
                else:
                    scores.append(0.0)
                prev_frame = gray
            idx += 1

        cap.release()
        return np.array(scores, dtype=np.float64)


class HistogramDifferenceDetector(ShotBoundaryDetector):
    """
    Compare colour histograms of consecutive frames using chi-squared distance.

    Robust to small camera motion and lighting changes; good for hard cuts.
    """

    name = "Histogram Difference (chi-squared)"

    def __init__(self, bins: int = 64):
        self.bins = bins

    def _hist(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Compute a normalised concatenated BGR histogram."""
        hists = []
        for ch in range(3):
            h = cv2.calcHist([frame_bgr], [ch], None, [self.bins], [0, 256])
            h = h.flatten().astype(np.float64)
            h /= (h.sum() + 1e-8)
            hists.append(h)
        return np.concatenate(hists)

    def compute_scores(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        prev_hist = None
        scores = []
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                hist = self._hist(frame)
                if prev_hist is not None:
                    # Chi-squared distance
                    chi2 = np.sum((hist - prev_hist) ** 2 / (hist + prev_hist + 1e-10))
                    scores.append(chi2)
                else:
                    scores.append(0.0)
                prev_hist = hist
            idx += 1

        cap.release()
        return np.array(scores, dtype=np.float64)


class EntropyChangeDetector(ShotBoundaryDetector):
    """
    Detect shots by large changes in per-frame Shannon entropy
    (computed on the grayscale intensity histogram).

    Scene changes often cause abrupt entropy shifts.
    """

    name = "Entropy Change"

    def __init__(self, bins: int = 256):
        self.bins = bins

    @staticmethod
    def _entropy(hist: np.ndarray) -> float:
        p = hist / (hist.sum() + 1e-10)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def compute_scores(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        prev_entropy = None
        scores = []
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [self.bins], [0, 256]).flatten()
                ent = self._entropy(hist)
                if prev_entropy is not None:
                    scores.append(abs(ent - prev_entropy))
                else:
                    scores.append(0.0)
                prev_entropy = ent
            idx += 1

        cap.release()
        return np.array(scores, dtype=np.float64)


class OpticalFlowMagnitudeDetector(ShotBoundaryDetector):
    """
    Farneback dense optical flow -- mean flow magnitude between consecutive frames.

    Hard cuts produce very large apparent motion.  Gradual transitions
    (dissolves, wipes) show elevated but smaller motion.

    NOTE: This is slower than the histogram/pixel methods.
    """

    name = "Optical Flow Magnitude"

    def compute_scores(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        prev_gray = None
        scores = []
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray,
                        None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                    )
                    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    scores.append(float(np.mean(mag)))
                else:
                    scores.append(0.0)
                prev_gray = gray
            idx += 1

        cap.release()
        return np.array(scores, dtype=np.float64)


class EdgeChangeRatioDetector(ShotBoundaryDetector):
    """
    Edge Change Ratio (ECR) -- measures the fraction of new/disappeared edge
    pixels between consecutive frames.

    Particularly useful for detecting gradual transitions (dissolves, wipes)
    that pixel-difference methods can miss.
    """

    name = "Edge Change Ratio"

    def __init__(self, dilate_size: int = 3):
        self.dilate_size = dilate_size

    def compute_scores(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        prev_edges = None
        scores = []
        idx = 0
        kernel = np.ones((self.dilate_size, self.dilate_size), np.uint8)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)

                if prev_edges is not None:
                    # Dilate previous edges to allow small shifts
                    dilated_prev = cv2.dilate(prev_edges, kernel)
                    dilated_curr = cv2.dilate(edges, kernel)

                    # New edge pixels (in current but not near previous)
                    entering = np.count_nonzero(edges & ~dilated_prev)
                    # Disappeared edge pixels
                    exiting = np.count_nonzero(prev_edges & ~dilated_curr)

                    n_curr = max(np.count_nonzero(edges), 1)
                    n_prev = max(np.count_nonzero(prev_edges), 1)

                    ecr = max(entering / n_curr, exiting / n_prev)
                    scores.append(ecr)
                else:
                    scores.append(0.0)

                prev_edges = edges
            idx += 1

        cap.release()
        return np.array(scores, dtype=np.float64)


class BlackFrameTransitionDetector(ShotBoundaryDetector):
    """
    Flag transitions into and out of black / near-monochrome frames.

    Many ad insertions transition through a brief black or near-black
    frame. Per frame we compute a binary ``is_black`` indicator (mean
    brightness low AND pixel variance low); the score is |delta| from the
    previous frame, so it spikes exactly at the entry and exit of every
    black segment.

    Suggested detection settings: fixed threshold = 0.5 (values are 0/1):
        detector.detect(path, threshold=0.5, adaptive=False)
    """
    name = "Black Frame Transition"

    def __init__(self, mean_threshold: float = 25.0, std_threshold: float = 15.0):
        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold

    def compute_scores(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        prev_is_black: Optional[float] = None
        scores: List[float] = []
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                is_black = 1.0 if (gray.mean() < self.mean_threshold
                                    and gray.std() < self.std_threshold) else 0.0
                if prev_is_black is not None:
                    scores.append(abs(is_black - prev_is_black))
                else:
                    scores.append(0.0)
                prev_is_black = is_black
            idx += 1

        cap.release()
        return np.array(scores, dtype=np.float64)


class CLIPFeatureDetector(ShotBoundaryDetector):
    """
    Cosine distance between CLIP image features of sampled frames.

    Intended to run at 1-second intervals (``sample_every = int(round(fps))``).
    CLIP features are semantic, so two talking-head shots are close while
    content-to-ad tends to be far. Much more robust than pixel / histogram
    methods at the cost of a PyTorch + OpenCLIP dependency and GPU-friendly
    runtime (~1 s per 30 s of video on CPU, much faster on CUDA).

    Install:   pip install open_clip_torch torch Pillow

    Usage:
        fps = get_video_fps(path)
        detector = CLIPFeatureDetector()
        res = detector.detect(path, sample_every=int(round(fps)))
    """
    name = "CLIP Feature Distance"

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self._model = None
        self._preprocess = None
        self._torch = None
        self._device = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as e:
            raise ImportError(
                "CLIPFeatureDetector requires 'open_clip_torch' and 'torch'. "
                "Install with: pip install open_clip_torch torch"
            ) from e
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._model = model.to(self._device).eval()
        self._preprocess = preprocess

    def embed_frames(self, rgb_frames: List[np.ndarray]) -> np.ndarray:
        """
        Encode a list of RGB uint8 frames (each shaped [H, W, 3]) to
        L2-normalised CLIP image features. Returns [N, feature_dim].
        Batches according to ``self.batch_size``.
        """
        self._load_model()
        from PIL import Image
        torch = self._torch

        if not rgb_frames:
            return np.empty((0, 0), dtype=np.float32)

        tensors = [self._preprocess(Image.fromarray(f)) for f in rgb_frames]
        feats = []
        with torch.no_grad():
            for i in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[i:i + self.batch_size]).to(self._device)
                f = self._model.encode_image(batch)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def compute_scores(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        rgb_frames: List[np.ndarray] = []
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every == 0:
                rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()

        feats = self.embed_frames(rgb_frames)
        if len(feats) == 0:
            return np.array([], dtype=np.float64)

        scores = [0.0]
        for i in range(1, len(feats)):
            sim = float(np.dot(feats[i - 1], feats[i]))
            scores.append(1.0 - sim)
        return np.array(scores, dtype=np.float64)


class TransNetV2Detector(ShotBoundaryDetector):
    """
    TransNet V2 pretrained shot-boundary detector.

    State-of-the-art CNN for shot boundary detection (BBC / RAI benchmarks).
    Outputs a calibrated per-frame boundary probability; ``sample_every`` is
    ignored -- every frame is processed. Frames are resized to 48x27 internally.

    Install:   pip install transnetv2-pytorch torch

    Suggested detection settings: fixed threshold = 0.5 (values are already
    calibrated probabilities):
        detector.detect(path, threshold=0.5, adaptive=False, min_gap_frames=15)
    """
    name = "TransNet V2"

    def __init__(self):
        self._model = None
        self._torch = None
        self._device = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transnetv2_pytorch import TransNetV2
        except ImportError as e:
            raise ImportError(
                "TransNetV2Detector requires 'transnetv2-pytorch' and 'torch'.\n"
                "  pip install transnetv2-pytorch torch\n"
                "If the weights aren't bundled, download them from\n"
                "  https://github.com/soCzech/TransNetV2\n"
                "and do: model.load_state_dict(torch.load('transnetv2-pytorch-weights.pth'))"
            ) from e
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = TransNetV2().to(self._device).eval()

    def compute_scores(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
    ) -> np.ndarray:
        """
        Return one boundary probability per original video frame. ``sample_every``
        is accepted for interface compatibility but ignored -- TransNet V2 needs
        every frame for temporal context.
        """
        self._load_model()
        torch = self._torch

        # Read every frame at TransNet's expected 48x27 resolution
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (48, 27)))
        cap.release()

        if not frames:
            return np.array([], dtype=np.float64)

        arr = np.array(frames, dtype=np.uint8)  # [N, 27, 48, 3]
        N = len(arr)

        # Reference TransNet V2 inference pattern:
        #   - Pad 25 frames of context at start and end (edge-repeat)
        #   - Slide 100-frame windows with stride 50
        #   - Take only the MIDDLE 50 predictions (indices 25:75) from each
        #     window so every real frame has 25 frames of context on each side
        #   - Pad end extra so (padded_len - 100) is a multiple of 50
        pad_begin = 25
        pad_end = 25
        total = N + pad_begin + pad_end
        if total < 100:
            pad_end += 100 - total
            total = 100
        extra = (50 - ((total - 100) % 50)) % 50
        pad_end += extra

        padded = np.concatenate([
            np.repeat(arr[:1], pad_begin, axis=0),
            arr,
            np.repeat(arr[-1:], pad_end, axis=0),
        ], axis=0)

        chunks = []
        with torch.no_grad():
            for i in range(0, len(padded) - 100 + 1, 50):
                window = padded[i:i + 100]
                t = torch.from_numpy(window).unsqueeze(0).to(self._device)
                sfp, _ = self._model(t)  # [1, 100, 1]
                p = torch.sigmoid(sfp).squeeze(-1).squeeze(0).cpu().numpy()
                chunks.append(p[25:75])  # middle 50 predictions

        probs = np.concatenate(chunks)[:N]
        return probs.astype(np.float64)

    def detect(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
        threshold: Optional[float] = 0.5,
        adaptive: bool = False,
        adaptive_k: float = 3.0,
        min_gap_frames: int = 15,
    ) -> DetectionResult:
        """
        Override: TransNet V2 returns one probability per original frame, so
        we run the thresholding loop ourselves and force ``sample_every=1``
        in the result -- otherwise the base class multiplies detections by
        ``sample_every`` and shifts them off by a factor of the stride.
        """
        scores = self.compute_scores(video_path)  # length = N frames

        if threshold is None and adaptive:
            threshold = float(np.mean(scores) + adaptive_k * np.std(scores))
        elif threshold is None:
            threshold = 0.5

        candidates = np.where(scores > threshold)[0]

        detected: List[int] = []
        for c in candidates:
            if len(detected) == 0 or (c - detected[-1]) >= min_gap_frames:
                detected.append(int(c))
            elif scores[c] > scores[detected[-1]]:
                detected[-1] = int(c)

        return DetectionResult(
            method_name=self.name,
            scores=scores,
            detected_frames=detected,   # already in original frame units
            threshold=float(threshold),
            sample_every=1,
        )


class HybridTransNetCLIPDetector(ShotBoundaryDetector):
    """
    Two-stage detector: TransNet V2 for shot-boundary candidates,
    CLIP similarity to filter out intra-content cuts.

    TransNet tends to flag every shot change, including rapid cuts inside
    the same content scene. We walk its candidates in time order and keep
    only the ones where the CLIP feature of a frame just after the
    candidate differs from the CLIP feature of a frame just after the
    **previous TransNet candidate** (whether or not it was kept) by more
    than ``clip_threshold``. The first candidate is always kept.

    Rationale: two consecutive TransNet boundaries that look similar in
    CLIP space are most likely two cuts inside the same content scene,
    so the later one isn't opening a new context. A large CLIP jump
    between consecutive TransNet boundaries signals a genuine context
    change (e.g. content -> ad).

    Requires: torch, transnetv2-pytorch, open_clip_torch, Pillow
    """
    name = "Hybrid TransNet + CLIP"

    def __init__(
        self,
        transnet_threshold: float = 0.5,
        transnet_min_gap: int = 15,
        clip_threshold: float = 0.30,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        post_offset: int = 10,
    ):
        self.transnet_threshold = transnet_threshold
        self.transnet_min_gap = transnet_min_gap
        self.clip_threshold = clip_threshold
        self.post_offset = post_offset
        self._transnet = TransNetV2Detector()
        self._clip = CLIPFeatureDetector(
            model_name=clip_model, pretrained=clip_pretrained
        )

    def compute_scores(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
    ) -> np.ndarray:
        """Return TransNet's per-frame probabilities (for plot compatibility)."""
        return self._transnet.compute_scores(video_path)

    def detect(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,          # ignored
        threshold: Optional[float] = None,   # ignored
        adaptive: bool = False,         # ignored
        adaptive_k: float = 3.0,        # ignored
        min_gap_frames: Optional[int] = None,
    ) -> DetectionResult:
        # Stage 1: TransNet candidates
        gap = min_gap_frames if min_gap_frames is not None else self.transnet_min_gap
        tn = self._transnet.detect(
            video_path,
            threshold=self.transnet_threshold,
            adaptive=False,
            min_gap_frames=gap,
        )
        candidates = list(tn.detected_frames)

        if not candidates:
            return DetectionResult(
                method_name=self.name,
                scores=tn.scores,
                detected_frames=[],
                threshold=self.clip_threshold,
                sample_every=1,
            )

        # Stage 2: read one frame just after each candidate
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rgb_frames: List[np.ndarray] = []
        valid: List[int] = []
        for c in candidates:
            target = min(c + self.post_offset, max(total - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            valid.append(c)
        cap.release()

        if not rgb_frames:
            return DetectionResult(
                method_name=self.name,
                scores=tn.scores,
                detected_frames=[],
                threshold=self.clip_threshold,
                sample_every=1,
            )

        # Stage 3: batched CLIP embedding
        feats = self._clip.embed_frames(rgb_frames)

        # Stage 4: filter -- compare each candidate to the IMMEDIATELY
        # PREVIOUS candidate (not the previous kept), so last_feat updates
        # on every iteration regardless of the keep/discard decision.
        kept: List[int] = []
        last_feat = None
        for c, f in zip(valid, feats):
            if last_feat is None:
                kept.append(c)
            else:
                dist = 1.0 - float(np.dot(last_feat, f))
                if dist > self.clip_threshold:
                    kept.append(c)
            last_feat = f

        return DetectionResult(
            method_name=self.name,
            scores=tn.scores,
            detected_frames=kept,
            threshold=self.clip_threshold,
            sample_every=1,
        )


class HybridTransNetRandomCLIPDetector(ShotBoundaryDetector):
    """
    Two-stage detector:
      1) TransNet V2 proposes shot-boundary candidates.
      2) For each candidate, compare a frame after the boundary against
         randomly sampled reference frames from the whole video in CLIP space.

    Candidates whose average cosine similarity to the random references is
    below ``ad_similarity_threshold`` are labeled as ads.

    Intuition: ad frames tend to be semantically different from the overall
    visual context of the video, so their mean similarity to random context
    frames is lower.

    Requires: torch, transnetv2-pytorch, open_clip_torch, Pillow
    """
    name = "Hybrid TransNet + Random CLIP"

    def __init__(
        self,
        transnet_threshold: float = 0.5,
        transnet_min_gap: int = 15,
        ad_similarity_threshold: float = 0.22,
        random_samples: int = 10,
        random_seed: Optional[int] = 42,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        post_offset: int = 10,
    ):
        self.transnet_threshold = transnet_threshold
        self.transnet_min_gap = transnet_min_gap
        self.ad_similarity_threshold = ad_similarity_threshold
        self.random_samples = random_samples
        self.random_seed = random_seed
        self.post_offset = post_offset
        self._transnet = TransNetV2Detector()
        self._clip = CLIPFeatureDetector(
            model_name=clip_model, pretrained=clip_pretrained
        )

    def compute_scores(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,
    ) -> np.ndarray:
        """Return TransNet per-frame probabilities for plotting compatibility."""
        return self._transnet.compute_scores(video_path)

    def detect(
        self,
        video_path: Path | str,
        *,
        sample_every: int = 1,          # ignored
        threshold: Optional[float] = None,   # ignored
        adaptive: bool = False,         # ignored
        adaptive_k: float = 3.0,        # ignored
        min_gap_frames: Optional[int] = None,
    ) -> DetectionResult:
        # Stage 1: TransNet candidates
        gap = min_gap_frames if min_gap_frames is not None else self.transnet_min_gap
        tn = self._transnet.detect(
            video_path,
            threshold=self.transnet_threshold,
            adaptive=False,
            min_gap_frames=gap,
        )
        candidates = list(tn.detected_frames)

        if not candidates:
            return DetectionResult(
                method_name=self.name,
                scores=tn.scores,
                detected_frames=[],
                threshold=self.ad_similarity_threshold,
                sample_every=1,
            )

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return DetectionResult(
                method_name=self.name,
                scores=tn.scores,
                detected_frames=[],
                threshold=self.ad_similarity_threshold,
                sample_every=1,
            )

        # Stage 2: sample random reference frames from the whole video
        n_refs = max(1, min(self.random_samples, total))
        rng = np.random.default_rng(self.random_seed)
        ref_indices = np.sort(rng.choice(total, size=n_refs, replace=False)).tolist()

        # Candidate frames are sampled just after each TransNet boundary.
        cand_targets = [min(c + self.post_offset, max(total - 1, 0)) for c in candidates]

        # Read all needed frames with one pass over unique indices.
        needed = sorted(set(ref_indices + cand_targets))
        index_to_rgb = {}
        for idx in needed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                index_to_rgb[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

        ref_frames = [index_to_rgb[i] for i in ref_indices if i in index_to_rgb]
        if not ref_frames:
            return DetectionResult(
                method_name=self.name,
                scores=tn.scores,
                detected_frames=[],
                threshold=self.ad_similarity_threshold,
                sample_every=1,
            )

        cand_frames: List[np.ndarray] = []
        valid_candidates: List[int] = []
        for c, target in zip(candidates, cand_targets):
            f = index_to_rgb.get(target)
            if f is not None:
                cand_frames.append(f)
                valid_candidates.append(c)

        if not cand_frames:
            return DetectionResult(
                method_name=self.name,
                scores=tn.scores,
                detected_frames=[],
                threshold=self.ad_similarity_threshold,
                sample_every=1,
            )

        # Stage 3: CLIP embeddings and average similarity filtering
        feats = self._clip.embed_frames(ref_frames + cand_frames)
        ref_feats = feats[:len(ref_frames)]
        cand_feats = feats[len(ref_frames):]

        kept: List[int] = []
        for c, f in zip(valid_candidates, cand_feats):
            sims = ref_feats @ f
            mean_sim = float(np.mean(sims))
            if mean_sim < self.ad_similarity_threshold:
                kept.append(c)

        return DetectionResult(
            method_name=self.name,
            scores=tn.scores,
            detected_frames=kept,
            threshold=self.ad_similarity_threshold,
            sample_every=1,
        )


# ---------------------------------------------------------------------------
# Registry -- handy for looping over all methods in the notebook
# ---------------------------------------------------------------------------
ALL_DETECTORS: List[ShotBoundaryDetector] = [
    # FrameDifferenceDetector(),
    # HistogramDifferenceDetector(bins=64),
    # EntropyChangeDetector(bins=128),
    # EdgeChangeRatioDetector(dilate_size=3),
    # BlackFrameTransitionDetector(),
    # Optional -- require extra dependencies, uncomment when you want them:
    # OpticalFlowMagnitudeDetector(),                      # slow, no extra deps
    # CLIPFeatureDetector(),                               # needs open_clip_torch + torch
    # TransNetV2Detector(),                                # needs transnetv2-pytorch + torch
    # HybridTransNetCLIPDetector(),                        # needs transnetv2-pytorch + open_clip_torch + torch
    HybridTransNetRandomCLIPDetector(),                 # needs transnetv2-pytorch + open_clip_torch + torch
]


def get_detector(name: str) -> ShotBoundaryDetector:
    """Lookup a detector by its ``name`` attribute."""
    for d in ALL_DETECTORS:
        if d.name == name:
            return d
    raise KeyError(f"No detector named '{name}'. Available: {[d.name for d in ALL_DETECTORS]}")
