"""Train a reinforcement-learning overlay for position sizing."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:  # pragma: no cover - allow running as script
    from .env_freqtrade_like import EnvConfig, FreqtradeLikeEnv
except ImportError:  # type: ignore
    from env_freqtrade_like import EnvConfig, FreqtradeLikeEnv

FEATURE_COLUMNS = [
    "rsi",
    "zret",
    "atr_z",
    "adx_15m",
    "slope_15m",
    "slope_1h",
    "hour",
    "weekday",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL sizing policy")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("user_data/features/trade_features.parquet"),
        help="Feature dataset",
    )
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--model-output", type=Path, default=Path("user_data/models/rl_sizing.zip"))
    parser.add_argument("--max-position", type=float, default=1.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    return parser.parse_args()


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "stable-baselines3 is required for RL training. Install with `pip install stable-baselines3`."
        ) from exc

    args = parse_arguments()
    if not args.features.exists():
        raise SystemExit(f"Feature file not found: {args.features}")

    data = pd.read_parquet(args.features)
    if data.empty:
        raise SystemExit("Feature dataset is empty")

    missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing:
        raise SystemExit(f"Dataset missing required columns: {', '.join(missing)}")

    env_config = EnvConfig(
        feature_columns=FEATURE_COLUMNS,
        max_position_size=args.max_position,
        fee=args.fee,
        slippage=args.slippage,
    )
    env = DummyVecEnv([lambda: FreqtradeLikeEnv(data, env_config)])

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.total_timesteps)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_output))
    print(f"Saved RL model to {args.model_output}")


if __name__ == "__main__":
    main()
