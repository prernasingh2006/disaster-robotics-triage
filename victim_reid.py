"""
Disaster-Response Triage sub-system 2: Victim re-identification / deduplication memory.

Multiple robots (or the same robot passing by twice) may detect the same
victim. VictimMemory keeps a bank of embeddings and, on each new detection,
decides whether it's a re-observation of a known victim (update the record)
or a genuinely new victim (create a record) using cosine similarity against
stored embeddings.

This is the piece that most directly runs into Julie Shah's "when to
communicate" framing (ConTaCT / CommPlan): every time the ReID memory is
updated, the robot has to decide whether the update is worth relaying to
teammates right now, or whether it should wait. See comm_decision.py.
"""

from dataclasses import dataclass, field
import numpy as np


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


@dataclass
class VictimRecord:
    victim_id: int
    embedding: np.ndarray          # running mean embedding (appearance/audio fused)
    last_position: tuple           # (x, y)
    urgency: float = 0.0           # set/updated by urgency_scoring.py
    n_observations: int = 1
    last_seen_t: float = 0.0
    history: list = field(default_factory=list)  # (t, position, urgency)


class VictimMemory:
    """Cosine-similarity based ReID/deduplication store."""

    def __init__(self, match_threshold=0.80, ema_alpha=0.3):
        self.match_threshold = match_threshold
        self.ema_alpha = ema_alpha
        self.records: dict[int, VictimRecord] = {}
        self._next_id = 1

    def _best_match(self, embedding):
        best_id, best_sim = None, -1.0
        for vid, rec in self.records.items():
            sim = cosine_similarity(embedding, rec.embedding)
            if sim > best_sim:
                best_id, best_sim = vid, sim
        return best_id, best_sim

    def observe(self, embedding, position, t):
        """Register a new detection. Returns (record, is_new)."""
        embedding = np.asarray(embedding, dtype=float)

        if self.records:
            best_id, best_sim = self._best_match(embedding)
        else:
            best_id, best_sim = None, -1.0

        if best_id is not None and best_sim >= self.match_threshold:
            rec = self.records[best_id]
            rec.embedding = (
                (1 - self.ema_alpha) * rec.embedding + self.ema_alpha * embedding
            )
            rec.last_position = position
            rec.n_observations += 1
            rec.last_seen_t = t
            rec.history.append((t, position, rec.urgency))
            return rec, False

        vid = self._next_id
        self._next_id += 1
        rec = VictimRecord(
            victim_id=vid,
            embedding=embedding,
            last_position=position,
            last_seen_t=t,
            history=[(t, position, 0.0)],
        )
        self.records[vid] = rec
        return rec, True

    def update_urgency(self, victim_id, urgency):
        if victim_id in self.records:
            self.records[victim_id].urgency = urgency

    def snapshot(self):
        """Compact dict view, useful for building a comm message payload."""
        return {
            vid: {
                "position": rec.last_position,
                "urgency": rec.urgency,
                "n_observations": rec.n_observations,
                "last_seen_t": rec.last_seen_t,
            }
            for vid, rec in self.records.items()
        }