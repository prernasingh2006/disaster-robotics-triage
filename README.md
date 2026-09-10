# Disaster-Response Triage-Inspired Robotics Pipeline

A small end-to-end prototype for a disaster-response robot that has to
find victims, judge how urgent each one is, and decide what's worth
telling its teammates — without flooding a bandwidth-constrained,
time-critical channel with every minor update.

## Modules

| File | What it does |
|---|---|
| `audio_localization.py` | 4-mic TDOA localization (GCC-PHAT + multilateration) and mel-spectrogram sound event classification (voice call / tapping / breathing / background). |
| `urgency_scoring.py` | Fuses three signals into one urgency score: EAR-based blink/responsiveness, optical-flow chest-displacement breathing rate, and audio classifier confidence. |
| `victim_reid.py` | Cosine-similarity embedding memory (`VictimMemory` / `VictimRecord`) that deduplicates repeat sightings of the same victim vs. flags a genuinely new one. |
| `comm_decision.py` | The piece that ties this to Prof. Shah's work — see below. |
| `demo.py` | Synthetic end-to-end run through all four modules, printing the pipeline's reasoning at each step. |

Run it with:

```
pip install numpy scipy scikit-learn opencv-python-headless
python demo.py
```

## Why `comm_decision.py`

Everything upstream of it — localization, classification, urgency fusion,
ReID — was already part of the Disaster-Response Triage project. What's new here is
the piece connecting it to **ConTaCT** (Shah et al., *Coordination and
Communication for Time-Critical Collaborative Tasks in Unknown,
Deterministic Domains*): instead of broadcasting every ReID/urgency
update as soon as it happens, the robot keeps a running belief about what
its teammates already know, and only communicates when the estimated
*benefit* of an update (how much it changes the team's picture of the
world, weighted by urgency) exceeds the *cost* of interrupting them —
ConTaCT's core online cost/benefit comparison, scaled down to a much
smaller decentralized-triage setting.

The demo run makes the effect concrete:

- **First sighting of a victim** → maximal information gain → communicate.
- **Re-observed moments later, barely changed** → low information gain →
  hold, rather than re-broadcasting the same thing.
- **Same victim, condition later worsens (breathing rate drops, becomes
  unresponsive)** → urgency jump pushes benefit back above cost →
  communicate again.
- **A second, previously unseen victim** → zero prior team knowledge →
  communicate regardless of urgency.

This is a deliberately small, legible cost/benefit model — not a
reimplementation of ConTaCT's full Dec-POMDP belief propagation (Model
Self/Other Propagate + Update). The natural next step, and the direction
I'd want to take this, is swapping `information_gain()` for that fuller
belief-propagation machinery so the robot reasons about *what it thinks
teammates think* rather than a single running belief state — closer to
what ConTaCT and CommPlan actually do.

## Known limitations (for anyone reading the code closely)

- All sensor data in `demo.py` is synthetic — no real audio/video I/O.
- The mel-spectrogram classifier uses a small sklearn MLP as a stand-in
  for the production CNN (kept dependency-light on purpose); swapping in
  a torch CNN over the same `mel_spectrogram()` features is a drop-in change.
- TDOA localization accuracy degrades for sources far relative to the
  mic array's baseline (expected behavior for small-aperture arrays, not
  a bug) — visible in the demo's Victim B case.