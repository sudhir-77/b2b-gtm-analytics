from pathlib import Path

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def make_config(**overrides):
    cfg = load_config(ROOT / "config.yaml", quick=True)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.validate()
    return cfg
