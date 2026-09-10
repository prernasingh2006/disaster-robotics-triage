"""
demo.py
-------
End-to-end walkthrough of the Disaster-Response Triage pipeline on synthetic data:

  audio clips --> audio_localization.py  --> (source position, sound label)
  video ROI   --> urgency_scoring.py     --> (responsiveness, breathing rate)
  fused       --> urgency_scoring.py     --> urgency score in [0, 1]
  detection   --> victim_reid.py         --> new or existing VictimRecord
  update      --> comm_decision.py       --> communicate or hold, ConTaCT-style

All sensor data below is synthetically generated (no hardware / real
recordings needed) purely so this runs standalone and prints a readable
trace of the pipeline's reasoning at each stage.

Run: python demo.py
"""

import numpy as np

from audio_localization import localize_source, SoundEventClassifier
from urgency_scoring import (
    eye_aspect_ratio,
    responsiveness_score,
    estimate_breathing_rate,
    breathing_urgency,
    audio_urgency,
    fuse_urgency,
)
from victim_reid import VictimMemory
from comm_decision import TeamBelief, decide

RNG = np.random.default_rng(7)
FS_AUDIO = 16000
FS_VIDEO = 15


# ---------------------------------------------------------------------------
# Synthetic sensor data generators
# ---------------------------------------------------------------------------

def synth_clip(label, duration=1.0, fs=FS_AUDIO):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    noise = RNG.normal(0, 0.05, size=t.shape)
    if label == "voice_call":
        tone = 0.6 * np.sin(2 * np.pi * 350 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        return tone + noise
    if label == "tapping":
        clip = noise.copy()
        for k in range(0, len(t), fs // 4):
            clip[k:k + 50] += 0.8
        return clip
    if label == "breathing":
        return 0.15 * np.sin(2 * np.pi * 0.3 * t) + noise
    return noise  # background


SPEED_OF_SOUND = 343.0


def fractional_delay(signal, fs, delay_s):
    """Delay a signal by a (possibly fractional) number of samples via FFT phase shift."""
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    phase_shift = np.exp(-2j * np.pi * freqs * delay_s)
    return np.fft.irfft(spectrum * phase_shift, n=n)


def synth_mic_array_signals(mic_positions, true_source, base_signal, fs=FS_AUDIO):
    """Delay a source signal to each mic based on true geometry, for localize_source()."""
    mic_positions = np.asarray(mic_positions, dtype=float)
    true_source = np.asarray(true_source, dtype=float)
    signals = []
    for mic in mic_positions:
        dist = np.linalg.norm(true_source - mic)
        delay = dist / SPEED_OF_SOUND
        delayed = fractional_delay(base_signal, fs, delay)
        noisy = delayed + RNG.normal(0, 0.01, size=delayed.shape)
        signals.append(noisy)
    return signals


def synth_ear_sequence(n=20, breathing_normally=True):
    """Synthetic EAR trace: a healthy trace blinks a couple times; a
    concerning trace stays open (or closed) the whole window."""
    if breathing_normally:
        base = 0.30 + 0.02 * RNG.normal(size=n)
        for blink_at in (5, 13):
            base[blink_at:blink_at + 2] = 0.08
        return base
    return 0.30 + 0.01 * RNG.normal(size=n)  # flat, no blink -> ambiguous/concerning


def synth_video_frames(n_frames, roi, breathing_rate_bpm, fs=FS_VIDEO, frame_size=(120, 160)):
    """Frames containing a bright blob over `roi` that oscillates vertically
    at `breathing_rate_bpm`, so optical flow on the ROI recovers that rate."""
    h, w = frame_size
    x, y, rw, rh = roi
    freq_hz = breathing_rate_bpm / 60.0
    frames = []
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = x + rw / 2, y + rh / 2
    for i in range(n_frames):
        t = i / fs
        offset = 4.0 * np.sin(2 * np.pi * freq_hz * t)  # pixels of chest rise/fall
        blob = np.exp(-(((xx - cx) ** 2 + (yy - (cy + offset)) ** 2) / (2 * (rh / 3) ** 2)))
        frame = (blob * 255).astype(np.uint8)
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Main pipeline walkthrough
# ---------------------------------------------------------------------------

def victim_embedding(victim_key, noise_scale=0.05):
    """Deterministic per-victim appearance/audio embedding + small per-
    observation noise, so repeat sightings of the same victim actually
    match in VictimMemory (real embeddings would come from a ReID net;
    this stands in for "the same victim looks/sounds similar each time")."""
    local_rng = np.random.default_rng(abs(hash(victim_key)) % (2 ** 32))
    base = local_rng.normal(size=16)
    noise = RNG.normal(0, noise_scale, size=16)
    return base + noise


def run_victim_case(case_name, victim_key, audio_label, breathing_bpm, responsive, position,
                     classifier, mic_positions, memory, team_belief, t):
    print(f"\n=== {case_name} (t={t}s) ===")

    # 1. Audio: localize + classify
    base_clip = synth_clip(audio_label)
    mic_signals = synth_mic_array_signals(mic_positions, position, base_clip)
    est_position, residual = localize_source(mic_positions, mic_signals, FS_AUDIO)
    label, confidence = classifier.predict(base_clip, FS_AUDIO)
    label = str(label)
    print(f"  audio_localization: est_pos=({est_position[0]:.2f}, {est_position[1]:.2f}) "
          f"true_pos=({position[0]:.2f}, {position[1]:.2f}) residual={residual:.4f}")
    print(f"  audio classifier:   label={label!r} confidence={confidence:.2f}")

    # 2. Video: responsiveness + breathing
    ear_seq = synth_ear_sequence(breathing_normally=responsive)
    resp_score = responsiveness_score(ear_seq)

    roi = (40, 30, 40, 40)
    frames = synth_video_frames(300, roi, breathing_bpm)  # 20s window for stable FFT resolution
    est_bpm = estimate_breathing_rate(frames, roi, FS_VIDEO)
    breath_score = breathing_urgency(est_bpm)
    print(f"  responsiveness_score={resp_score:.2f}  "
          f"est_breathing_rate={est_bpm:.1f} bpm (true={breathing_bpm}) breath_score={breath_score:.2f}")

    # 3. Fuse urgency
    a_urgency = audio_urgency(label, confidence)
    urgency = fuse_urgency(resp_score, breath_score, a_urgency)
    print(f"  fused urgency score = {urgency:.2f}")

    # 4. ReID / dedup
    embedding = victim_embedding(victim_key)
    rec, is_new = memory.observe(embedding, position, t)
    memory.update_urgency(rec.victim_id, urgency)
    print(f"  victim_reid: {'NEW victim' if is_new else 'matched existing victim'} "
          f"id={rec.victim_id} n_observations={rec.n_observations}")

    # 5. ConTaCT-style if/when-to-communicate decision
    decision = decide(team_belief, rec.victim_id, position, urgency)
    verdict = "COMMUNICATE" if decision.should_communicate else "hold (not worth interrupting team yet)"
    print(f"  comm_decision: benefit={decision.benefit:.2f} cost={decision.cost:.2f} "
          f"info_gain={decision.info_gain:.2f} -> {verdict}")

    return rec, decision


def main():
    mic_positions = [(0, 0), (2, 0), (0, 2), (2, 2)]  # 4-mic array, meters

    classifier = SoundEventClassifier()
    train_clips, train_labels = [], []
    for label in SoundEventClassifier.LABELS:
        for _ in range(15):
            train_clips.append(synth_clip(label))
            train_labels.append(label)
    classifier.fit(train_clips, train_labels, FS_AUDIO)

    memory = VictimMemory()
    team_belief = TeamBelief()

    # Case 1: first sighting, distressed victim calling out -> should communicate
    run_victim_case(
        "Victim A - first detection, calling for help", victim_key="A",
        audio_label="voice_call", breathing_bpm=28, responsive=True,
        position=(1.0, 1.0), classifier=classifier, mic_positions=mic_positions,
        memory=memory, team_belief=team_belief, t=0.0,
    )

    # Case 2: same victim, re-observed moments later with barely-changed
    # urgency/position -> low information gain, comm_decision should hold
    run_victim_case(
        "Victim A - re-observed shortly after (small change)", victim_key="A",
        audio_label="voice_call", breathing_bpm=27, responsive=True,
        position=(1.05, 1.02), classifier=classifier, mic_positions=mic_positions,
        memory=memory, team_belief=team_belief, t=4.0,
    )

    # Case 3: same victim, breathing has dropped into a concerning range and
    # responsiveness has degraded -> large urgency jump, should communicate
    run_victim_case(
        "Victim A - re-observed later, condition worsening", victim_key="A",
        audio_label="breathing", breathing_bpm=6, responsive=False,
        position=(1.1, 1.0), classifier=classifier, mic_positions=mic_positions,
        memory=memory, team_belief=team_belief, t=40.0,
    )

    # Case 4: a second, previously unseen victim elsewhere -> new record,
    # zero prior team knowledge -> should communicate regardless of urgency
    run_victim_case(
        "Victim B - new victim, different location", victim_key="B",
        audio_label="tapping", breathing_bpm=16, responsive=True,
        position=(3.0, 2.5), classifier=classifier, mic_positions=mic_positions,
        memory=memory, team_belief=team_belief, t=42.0,
    )

    print("\n=== Final VictimMemory snapshot (what this robot has recorded) ===")
    for vid, info in memory.snapshot().items():
        print(f"  victim {vid}: {info}")


if __name__ == "__main__":
    main()