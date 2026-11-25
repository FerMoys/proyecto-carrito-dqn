"""Línea base aleatoria con registro compatible con la comparación experimental."""
import argparse
import csv
from pathlib import Path
import numpy as np
from car_env import CarTrackEnv

FIELDS = ["agente", "configuracion", "semilla", "condicion", "episodio",
          "reward", "success", "collision", "steps"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--csv-output", default="comparison_results.csv")
    args = parser.parse_args()

    env = CarTrackEnv(render_mode=None)
    rows = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        env.action_space.seed(args.seed * 10000 + episode)
        env.reset(seed=episode_seed, options={"random_start": args.random_start})
        total_reward, steps, done, info = 0.0, 0, False, {}
        while not done:
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        rows.append({
            "agente": "Aleatorio", "configuracion": "Aleatorio",
            "semilla": args.seed,
            "condicion": "aleatoria" if args.random_start else "fija",
            "episodio": episode + 1, "reward": total_reward,
            "success": int(info.get("success", 0)),
            "collision": int(info.get("collision", 0)), "steps": steps,
        })
    env.close()

    output = Path(args.csv_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    print({
        "agente": "Aleatorio", "semilla": args.seed,
        "condicion": "aleatoria" if args.random_start else "fija",
        "episodes": args.episodes,
        "reward_mean": float(np.mean([r["reward"] for r in rows])),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "steps_mean": float(np.mean([r["steps"] for r in rows])),
        "csv": str(output),
    })

if __name__ == "__main__":
    main()
