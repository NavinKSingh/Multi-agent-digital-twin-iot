import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "environment"))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "digital_twin"))

from iot_network_real import RealIoTNetwork
from dqn_agent import DQNAgent
from twin_manager import TwinManager


CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "energy_data.csv")
NUM_EPISODES = 300
STEPS_PER_EPISODE = 30
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def state_to_vector(state_dict):
    return [state_dict["battery"], state_dict["queue"]]


def run_training():
    network = RealIoTNetwork(CSV_PATH)
    num_devices = len(network.devices)

    twin_manager = TwinManager(num_devices=num_devices)
    agents = [DQNAgent(state_size=2, action_size=2) for _ in range(num_devices)]

    episode_rewards = []
    episode_battery_left = []

    for episode in range(1, NUM_EPISODES + 1):
        states = network.reset()
        twin_manager.sync_from_network(states)
        total_reward = 0

        for step in range(STEPS_PER_EPISODE):
            actions = [agent.act(state_to_vector(states[i])) for i, agent in enumerate(agents)]
            rewards, next_states = network.step(actions)
            twin_manager.sync_from_network(next_states)
            total_reward += sum(rewards)

            for i, agent in enumerate(agents):
                s = state_to_vector(states[i])
                ns = state_to_vector(next_states[i])
                done = not next_states[i]["alive"]
                agent.remember(s, actions[i], rewards[i], ns, done)
                agent.train_step(batch_size=16)

            states = next_states
            if network.is_done():
                break

        avg_battery = np.mean([d.battery for d in network.devices])
        episode_rewards.append(total_reward)
        episode_battery_left.append(avg_battery)

        if episode % 25 == 0:
            print(f"Episode {episode:3d} | "
                  f"Total Reward: {total_reward:7.2f} | "
                  f"Avg Battery Left: {avg_battery:6.2f}% | "
                  f"Exploration (epsilon): {agents[0].epsilon:.3f}")

    print("\nTraining finished (on real dataset)!")
    print(f"First episode reward:  {episode_rewards[0]:.2f}")
    print(f"Last episode reward:   {episode_rewards[-1]:.2f}")
    print(f"First episode avg battery left: {episode_battery_left[0]:.2f}%")
    print(f"Last episode avg battery left:  {episode_battery_left[-1]:.2f}%")

    print("\nFinal Digital Twin snapshot:")
    print(twin_manager.summary())

    save_results_graph(episode_rewards, episode_battery_left)
    return episode_rewards, episode_battery_left


def save_results_graph(episode_rewards, episode_battery_left):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(episode_rewards, color="tab:blue")
    ax1.set_title("Total Reward per Episode (Real Data)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")

    ax2.plot(episode_battery_left, color="tab:green")
    ax2.set_title("Avg Battery Left per Episode (Real Data)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Battery (%)")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "training_progress_real_data.png")
    plt.savefig(out_path)
    print(f"\nGraph saved to: {out_path}")


if __name__ == "__main__":
    run_training()