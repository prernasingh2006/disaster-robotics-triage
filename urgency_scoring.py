"""
Disaster-Response Triage sub-system 3: Multi-signal urgency scoring.

Fuses three independent signals into a single triage urgency score in
[0, 1] (1 = most urgent):

  1. Eye-Aspect-Ratio (EAR) blink detection  -> consciousness/responsiveness cue.
     No blinks over a window + open/closed eye ratio outside normal range
     is treated as a higher-urgency cue (possible unresponsiveness).
  2. Optical-flow chest displacement          -> breathing rate estimate.
     Very low or highly irregular breathing rate raises urgency.
  3. Audio classification confidence          -> from audio_localization.py.
     A confident "voice_call" (calling for help, alert/distressed) raises
     urgency; confident "breathing"-only with no voice lowers it slightly;
     "background" contributes nothing.

Signals are fused with a simple weighted combination with a documented
rationale per weight, rather than a black box -- so it's easy to defend
in an interview and easy to swap out for a learned fusion model later.
"""

from dataclasses import dataclass
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# 1. EAR-based blink / responsiveness cue
# ---------------------------------------------------------------------------

def eye_aspect_ratio(eye_landmarks):
    """Standard EAR formula from 6 (x, y) eye landmarks (Soukupova & Cech).

    eye_landmarks: array of shape (6, 2), ordered
        p1 (left corner), p2, p3 (top), p4 (right corner), p5, p6 (bottom)
    """
    p = np.asarray(eye_landmarks, dtype=float)
    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-8)


def responsiveness_score(ear_sequence, blink_threshold=0.21, no_blink_penalty=0.6):
    """Score in [0, 1]: higher = more concerning (less responsive).

    ear_sequence: list/array of EAR values sampled over an observation window.
    """
    ear_sequence = np.asarray(ear_sequence, dtype=float)
    is_closed = ear_sequence < blink_threshold
    blink_count = int(np.sum(np.diff(is_closed.astype(int)) == 1))

    if blink_count == 0:
        # eyes stayed one state the whole window -- ambiguous, flag it
        return no_blink_penalty
    # normal blink rate ~15-20/min; window here is short, so any blinking
    # at all is treated as a strong "responsive" signal
    return max(0.0, 0.15 - 0.03 * blink_count)


# ---------------------------------------------------------------------------
# 2. Optical-flow breathing rate estimate
# ---------------------------------------------------------------------------

def estimate_breathing_rate(frames, roi, fs):
    """Estimate breathing rate (breaths/min) from chest-region optical flow.

    frames : list of grayscale numpy arrays (consecutive video frames)
    roi    : (x, y, w, h) bounding box over the chest region
    fs     : frame rate (Hz)
    """
    x, y, w, h = roi
    displacements = []
    prev = frames[0][y:y + h, x:x + w]
    for frame in frames[1:]:
        curr = frame[y:y + h, x:x + w]
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 2, 15, 3, 5, 1.2, 0
        )
        # vertical component captures chest rise/fall
        displacements.append(float(np.mean(flow[..., 1])))
        prev = curr

    displacements = np.array(displacements)
    if len(displacements) < 4:
        return None  # not enough frames to estimate a rate

    displacements -= displacements.mean()
    spectrum = np.abs(np.fft.rfft(displacements))
    freqs = np.fft.rfftfreq(len(displacements), d=1.0 / fs)

    # physiological breathing band: 6-40 breaths/min -> 0.1-0.67 Hz
    band = (freqs >= 0.1) & (freqs <= 0.67)
    if not np.any(band):
        return None
    peak_freq = freqs[band][np.argmax(spectrum[band])]
    return peak_freq * 60.0  # breaths/min


def breathing_urgency(rate_bpm, normal_range=(12, 20)):
    """Score in [0, 1]: higher = more concerning breathing rate."""
    if rate_bpm is None:
        return 0.5  # unknown -> moderate default, don't over/under-claim
    lo, hi = normal_range
    if lo <= rate_bpm <= hi:
        return 0.0
    if rate_bpm < lo:
        # bradypnea: more urgent the closer to zero
        return float(np.clip((lo - rate_bpm) / lo, 0, 1))
    # tachypnea: more urgent the further above hi, saturating at 2x hi
    return float(np.clip((rate_bpm - hi) / hi, 0, 1))


# ---------------------------------------------------------------------------
# 3. Fusion
# ---------------------------------------------------------------------------

@dataclass
class UrgencyWeights:
    responsiveness: float = 0.40
    breathing: float = 0.35
    audio: float = 0.25


AUDIO_URGENCY = {
    "voice_call": 1.0,   # calling for help / distress vocalization
    "tapping": 0.6,      # signaling but not vocal distress
    "breathing": 0.2,    # audible breathing only, no distress cue
    "background": 0.0,
}


def audio_urgency(label, confidence):
    base = AUDIO_URGENCY.get(label, 0.0)
    return base * confidence  # low-confidence detections contribute less


def fuse_urgency(responsiveness, breathing, audio, weights: UrgencyWeights = UrgencyWeights()):
    score = (
        weights.responsiveness * responsiveness
        + weights.breathing * breathing
        + weights.audio * audio
    )
    total_weight = weights.responsiveness + weights.breathing + weights.audio
    return float(np.clip(score / total_weight, 0.0, 1.0))