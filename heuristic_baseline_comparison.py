"""Línea base heurística por reglas con registro experimental compatible."""
import argparse
import csv
from pathlib import Path
import numpy as np
from car_env import CarTrackEnv

ACTION_FROM_CONTROL = {
    (0, 0): 0, (1, 0): 1, (-1, 0): 2, (0, -1): 3, (0, 1): 4,
    (1, -1): 5, (1, 1): 6, (-1, -1): 7, (-1, 1): 8,
}
FIELDS = ["agente", "configuracion", "semilla", "condicion", "episodio",
          "reward", "success", "collision", "steps"]

def heuristic_action(observation) -> int:
    far_left, left, front, right, far_right, speed, _, _, lateral = observation
    steering = 0
    if front < 0.38:
        steering = -1 if max(left, far_left) > max(right, far_right) else 1
    else:
        opening_difference = (right + far_right) - (left + far_left)
        if opening_difference > 0.16:
            steering = 1
        elif opening_difference < -0.16:
            steering = -1
        elif lateral > 0.35:
            steering = -1
        elif lateral < -0.35:
            steering = 1
    if speed > 0.72 and (front < 0.65 or steering != 0):
        throttle = -1
    elif speed < 0.60:
        throttle = 1
    else:
        throttle = 0
    return ACTION_FROM_CONTROL[(throttle, steering)]

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
        observation, _ = env.reset(
            seed=args.seed + episode,
            options={"random_start": args.random_start},
        )
        total_reward, steps, done, info = 0.0, 0, False, {}
        while not done:
            action = heuristic_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        rows.append({
            "agente": "Heuristico", "configuracion": "Heuristico",
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
        "agente": "Heuristico", "semilla": args.seed,
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
