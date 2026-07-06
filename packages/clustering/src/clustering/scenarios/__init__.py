"""S1-S5 scenarios + the shared :class:`Scenario` ABC.

Use :func:`build_scenario` to instantiate the right pipeline from a
resolved :class:`clustering.config.Config`.
"""

from __future__ import annotations

from clustering.config.schema import Config
from clustering.scenarios.base import Scenario, ScenarioResult
from clustering.scenarios.s1_unsupervised import S1Unsupervised
from clustering.scenarios.s2_partial_labels import S2PartialLabels
from clustering.scenarios.s3_partial_few_shot import S3PartialFewShot
from clustering.scenarios.s4_zero_shot import S4ZeroShot
from clustering.scenarios.s5_full_supervised import S5FullSupervised

_REGISTRY: dict[str, type[Scenario]] = {
    "s1": S1Unsupervised,
    "s2": S2PartialLabels,
    "s3": S3PartialFewShot,
    "s4": S4ZeroShot,
    "s5": S5FullSupervised,
}


def build_scenario(config: Config) -> Scenario:
    """Instantiate the scenario named by ``config.scenario.name``."""
    name = config.scenario.name
    if name not in _REGISTRY:
        raise KeyError(f"Unknown scenario {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config)


def registered_scenarios() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "S1Unsupervised",
    "S2PartialLabels",
    "S3PartialFewShot",
    "S4ZeroShot",
    "S5FullSupervised",
    "Scenario",
    "ScenarioResult",
    "build_scenario",
    "registered_scenarios",
]
