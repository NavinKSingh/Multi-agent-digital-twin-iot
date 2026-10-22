import sys
import os
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "environment"))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from iot_network import IoTNetwork
from dqn_agent import DQNAgent


NUM_DEVICES = 3
TRAIN_EPISODES = 300      # how long to train the smart agents first
EVAL_EPISODES = 30        # how many episodes to test each approach on
STEPS_PER_EPISODE = 30


def state_to_vector(state_dict):
    return [state_dict["battery"], state_dict["queue"]]


def train_agents():
    """Train fresh DQN agents, same as train.py does."""
    network = IoTNetwork(num_devices=NUM_DEVICES)
    agents = [DQNAgent(state_size=2, action_size=2) for _ in range(NUM_DEVICES)]

    for episode in range(TRAIN_EPISODES):
        states = network.reset()
        for step in range(STEPS_PER_EPISODE):
            actions = [agent.act(state_to_vector(states[i])) for i, agent in enumerate(agents)]
            rewards, next_states = network.step(actions)

            for i, agent in enumerate(agents):
                s = state_to_vector(states[i])
                ns = state_to_vector(next_states[i])
                done = not next_states[i]["alive"]
                agent.remember(s, actions[i], rewards[i], ns, done)
                agent.train_step(batch_size=16)

            states = next_states
            if network.is_done():
                break

    # freeze exploration for a fair evaluation - agents should use their
    # learned knowledge, not keep trying random actions
    for agent in agents:
        agent.epsilon = 0.0

    return agents


def run_episodes(policy_fn, episodes=EVAL_EPISODES):
    """
    policy_fn(states, agents_or_none) -> list of actions
    Runs many episodes and returns average stats.
    """
    network = IoTNetwork(num_devices=NUM_DEVICES)

    battery_left_all = []
    queue_left_all = []
    devices_alive_all = []

    for _ in range(episodes):
        states = network.reset()
        for step in range(STEPS_PER_EPISODE):
            actions = policy_fn(states)
            rewards, states = network.step(actions)
            if network.is_done():
                break

        battery_left_all.append(np.mean([d.battery for d in network.devices]))
        queue_left_all.append(np.mean([d.data_queue for d in network.devices]))
        devices_alive_all.append(sum(1 for d in network.devices if d.alive))

    return {
        "avg_battery_left": np.mean(battery_left_all),
        "avg_queue_left": np.mean(queue_left_all),
        "avg_devices_alive": np.mean(devices_alive_all),
    }


def random_policy(states):
    return [random.choice([0, 1]) for _ in states]


def make_rl_policy(agents):
    def rl_policy(states):
        return [agent.act(state_to_vector(states[i])) for i, agent in enumerate(agents)]
    return rl_policy


if __name__ == "__main__":
    print("Training RL agents...")
    agents = train_agents()

    print(f"\nEvaluating over {EVAL_EPISODES} episodes each...\n")

    random_results = run_episodes(random_policy)
    rl_results = run_episodes(make_rl_policy(agents))

    print("=" * 55)
    print(f"{'Metric':30s} {'Random':>10s} {'RL Agent':>10s}")
    print("=" * 55)
    print(f"{'Avg battery left (%)':30s} {random_results['avg_battery_left']:>10.2f} {rl_results['avg_battery_left']:>10.2f}")
    print(f"{'Avg pending data queue':30s} {random_results['avg_queue_left']:>10.2f} {rl_results['avg_queue_left']:>10.2f}")
    print(f"{'Avg devices still alive':30s} {random_results['avg_devices_alive']:>10.2f} {rl_results['avg_devices_alive']:>10.2f}")
    print("=" * 55)

    battery_improvement = (
        (rl_results["avg_battery_left"] - random_results["avg_battery_left"])
        / max(random_results["avg_battery_left"], 1e-6) * 100
    )
    print(f"\n>>> RL agents preserved {battery_improvement:.1f}% more battery than random decisions <<<")