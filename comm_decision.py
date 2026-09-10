"""
The piece that connects the Disaster-Response Triage pipeline to Shah et al.'s ConTaCT
(Coordination and Communication for Time-Critical Collaborative Tasks in
Unknown, Deterministic Domains): rather than broadcasting every ReID /
urgency update the instant it happens, the robot maintains a belief about
what its teammates already know, and only communicates when the estimated
value of the update to the team exceeds the cost of interrupting/using
bandwidth -- ConTaCT's core "compare expected benefit of a decision against
its cost" idea, adapted to a much smaller decentralized-triage setting.

This is a deliberately small, legible model (not a full Dec-POMDP solve):
  - benefit(update)  ~ how much this changes what the team believes about
                        the world, scaled by the victim's urgency
  - cost(comm)        ~ fixed bandwidth/interruption cost + a term that grows
                        with how much replanning the update would force
  - decide()          communicates iff benefit - cost > 0, mirroring
                        ConTaCT's online cost/benefit comparison rather than
                        a fixed "always broadcast" or "never broadcast" rule.

Swapping this module's benefit/cost estimators for the full ConTaCT belief
propagation (Model Self/Other Propagate + Update) is the natural next step
mentioned in the outreach email -- this is the "little bit of working code"
version of that idea, not a claim to have reimplemented ConTaCT in full.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


@dataclass
class TeamBelief:
    """What this robot believes its teammates currently know, per victim."""
    known_urgency: Dict[int, float] = field(default_factory=dict)
    known_position: Dict[int, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class CommCostModel:
    base_cost: float = 0.05       # fixed interruption/bandwidth cost per message
    replanning_weight: float = 0.5  # extra cost if the update forces re-tasking


def information_gain(team_belief: TeamBelief, victim_id, new_position, new_urgency):
    """How much this update changes the team's picture of the world.

    Two components:
      - position drift: has the victim moved meaningfully since teammates'
        last known estimate?
      - urgency delta: has the urgency estimate changed meaningfully
        (e.g. breathing rate crossed into a concerning range)?
    Unknown-to-the-team victims count as maximal gain (new information).
    """
    if victim_id not in team_belief.known_urgency:
        return 1.0  # brand-new victim: teammates have zero information

    old_urgency = team_belief.known_urgency[victim_id]
    old_position = np.array(team_belief.known_position[victim_id])
    new_position = np.array(new_position)

    urgency_delta = abs(new_urgency - old_urgency)
    position_drift = np.linalg.norm(new_position - old_position)
    # normalize drift against a 2m "meaningful move" scale
    position_signal = min(position_drift / 2.0, 1.0)

    return float(np.clip(0.7 * urgency_delta + 0.3 * position_signal, 0.0, 1.0))


def communication_benefit(info_gain, urgency, urgency_weight=1.5):
    """ConTaCT-style benefit: information that matters more when the
    situation is more urgent is worth more to communicate."""
    return info_gain * (1.0 + urgency_weight * urgency)


def communication_cost(info_gain, cost_model: CommCostModel = CommCostModel()):
    """Cost grows with how much re-tasking the update is likely to trigger,
    proxied here by info_gain itself (a bigger surprise -> more replanning)."""
    return cost_model.base_cost + cost_model.replanning_weight * info_gain


@dataclass
class CommDecision:
    should_communicate: bool
    benefit: float
    cost: float
    info_gain: float


def decide(
    team_belief: TeamBelief,
    victim_id,
    new_position,
    new_urgency,
    cost_model: CommCostModel = CommCostModel(),
) -> CommDecision:
    """The online if/when-to-communicate decision, run every time the ReID
    memory or urgency scorer produces an update for a given victim."""
    gain = information_gain(team_belief, victim_id, new_position, new_urgency)
    benefit = communication_benefit(gain, new_urgency)
    cost = communication_cost(gain, cost_model)
    should_communicate = benefit > cost

    if should_communicate:
        team_belief.known_urgency[victim_id] = new_urgency
        team_belief.known_position[victim_id] = tuple(new_position)

    return CommDecision(should_communicate, benefit, cost, gain)