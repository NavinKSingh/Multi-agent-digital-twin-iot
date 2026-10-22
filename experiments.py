"""
Reproduces every number and figure in the paper.

Usage:  python experiments.py            (10 seeds, both environments)
        python experiments.py --seeds 3  (quick test)

Outputs (in results/): fig_training.png, results.json, results_table.tex
"""
import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:                                   # works as a script and inside Colab/Jupyter
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
for sub in ("environment", "agents"):
    sys.path.append(os.path.join(HERE, sub))

from iot_network import IoTNetwork
from iot_network_real import RealIoTNetwork
from dqn_agent import DQNAgent

CSV_PATH = os.path.join(HERE, "data", "energy_data.csv")
RESULTS_DIR = os.path.join(HERE, "results")
TRAIN_EPISODES = 300
EVAL_EPISODES = 30
STEPS = 30
EVAL_SEED_BASE = 10_000          # same for every policy -> identical eval episodes


def make_env(kind):
    if kind == "sim":
        return IoTNetwork(num_devices=3)
    return RealIoTNetwork(CSV_PATH)


def sv(s):
    return [s["battery"], s["queue"]]


def train(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    n = len(env.devices)
    agents = [DQNAgent(state_size=2, action_size=2) for _ in range(n)]
    returns, final_batt = [], []
    for _ in range(TRAIN_EPISODES):
        states = env.reset()
        total = 0.0
        for _ in range(STEPS):
            acts = [a.act(sv(states[i])) for i, a in enumerate(agents)]
            rew, nxt = env.step(acts)
            total += sum(rew)
            for i, a in enumerate(agents):
                a.remember(sv(states[i]), acts[i], rew[i], sv(nxt[i]), not nxt[i]["alive"])
                a.train_step(batch_size=16)
            states = nxt
            if env.is_done():
                break
        returns.append(total)
        final_batt.append(float(np.mean([d.battery for d in env.devices])))
    for a in agents:
        a.epsilon = 0.0           # greedy evaluation
    return agents, returns, final_batt


def evaluate(env, policy):
    """Every policy sees identical episodes: global RNG is re-seeded before
    each episode; the random policy uses its own private RNG."""
    B, Q, G, A = [], [], [], []
    for ep in range(EVAL_EPISODES):
        random.seed(EVAL_SEED_BASE + ep)
        states = env.reset()
        total = 0.0
        for _ in range(STEPS):
            rew, states = env.step(policy(states))
            total += sum(rew)
            if env.is_done():
                break
        B.append(np.mean([d.battery for d in env.devices]))
        Q.append(np.mean([d.data_queue for d in env.devices]))
        G.append(total)
        A.append(100.0 * np.mean([d.alive for d in env.devices]))
    return float(np.mean(B)), float(np.mean(Q)), float(np.mean(G)), float(np.mean(A))


def run_seed(args):
    kind, seed = args
    env = make_env(kind) if kind == "sim" else _quiet_real()
    agents, rets, batts = train(env, seed)
    rng = random.Random(seed + 777)
    policies = {
        "DQN (ours)": lambda s: [a.act(sv(s[i])) for i, a in enumerate(agents)],
        "Random": lambda s: [rng.choice([0, 1]) for _ in s],
        "Always-send": lambda s: [1 for _ in s],
        "Always-sleep": lambda s: [0 for _ in s],
        "Queue-threshold": lambda s: [1 if x["queue"] > 0 else 0 for x in s],
    }
    res = {name: evaluate(env, pol) for name, pol in policies.items()}
    return res, rets, batts


def _quiet_real():
    so = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        return RealIoTNetwork(CSV_PATH)
    finally:
        sys.stdout.close()
        sys.stdout = so


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args, _ = ap.parse_known_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary, curves = {}, {}
    with Pool() as pool:
        for kind in ("sim", "real"):
            out = pool.map(run_seed, [(kind, s) for s in range(args.seeds)])
            names = list(out[0][0].keys())
            summary[kind] = {
                n: {
                    "mean": np.mean([o[0][n] for o in out], axis=0).tolist(),
                    "std": np.std([o[0][n] for o in out], axis=0).tolist(),
                } for n in names
            }
            curves[kind] = (np.array([o[1] for o in out]), np.array([o[2] for o in out]))

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- console + LaTeX tables ----
    titles = {"sim": "Simulated environment (3 devices)", "real": "Real-data environment (5 devices)"}
    tex = []
    for kind in ("sim", "real"):
        print(f"\n{titles[kind]}  (mean +- std over {args.seeds} seeds)")
        print(f"{'Policy':18s}{'Battery %':>16s}{'Queue':>16s}{'Return':>18s}{'Alive %':>14s}")
        tex.append(f"% {titles[kind]}")
        for n, v in summary[kind].items():
            m, s = v["mean"], v["std"]
            print(f"{n:18s}{m[0]:9.1f}+-{s[0]:4.1f}{m[1]:9.2f}+-{s[1]:4.2f}{m[2]:10.1f}+-{s[2]:5.1f}{m[3]:9.1f}+-{s[3]:4.1f}")
            tex.append(f"{n} & {m[0]:.1f} $\\pm$ {s[0]:.1f} & {m[1]:.2f} $\\pm$ {s[1]:.2f} & {m[2]:.1f} $\\pm$ {s[2]:.1f} & {m[3]:.1f} $\\pm$ {s[3]:.1f} \\\\")
        d, r = summary[kind]["DQN (ours)"]["mean"][0], summary[kind]["Random"]["mean"][0]
        print(f"Battery gain over Random: {(d - r) / r * 100:.1f}%")
    with open(os.path.join(RESULTS_DIR, "results_table.tex"), "w") as f:
        f.write("\n".join(tex))

    # ---- Figure: 2x2 training curves ----
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for col, kind in enumerate(("sim", "real")):
        for row, (data, lab) in enumerate(zip(curves[kind], ("Episode return", "Final battery (%)"))):
            m, s = data.mean(0), data.std(0)
            x = np.arange(1, len(m) + 1)
            ax[row][col].plot(x, m, color="tab:blue")
            ax[row][col].fill_between(x, m - s, m + s, alpha=0.25, color="tab:blue")
            ax[row][col].set_ylabel(lab)
            ax[row][col].grid(alpha=0.3)
        ax[0][col].set_title(titles[kind])
        ax[1][col].set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_training.png"), dpi=200)
    print("\nSaved results/fig_training.png, results.json, results_table.tex")


if __name__ == "__main__":
    main()
