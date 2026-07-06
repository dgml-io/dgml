"""Hydra DictConfig → validated pydantic :class:`Config` + deterministic run_id."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from clustering.config.schema import Config
from clustering.utils.runid import run_id_for


def resolve(cfg: DictConfig | dict[str, Any]) -> tuple[Config, str]:
    """Validate a Hydra-shaped config into a typed :class:`Config` and hash it.

    Args:
        cfg: Either an already-resolved ``DictConfig`` or a plain ``dict``.

    Returns:
        ``(validated_config, run_id)`` where ``run_id`` is the 12-char sha256
        prefix of the canonical JSON dump of the resolved config.

    Raises:
        TypeError: If the input does not resolve to a dict.
        ValidationError: If any field violates the schema.
    """
    if isinstance(cfg, DictConfig):
        plain = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    else:
        plain = cfg

    if not isinstance(plain, dict):
        raise TypeError(f"Resolved config must be a dict, got {type(plain).__name__}.")

    validated = Config.model_validate(plain)
    rid = run_id_for(validated.model_dump(mode="json"))
    return validated, rid
