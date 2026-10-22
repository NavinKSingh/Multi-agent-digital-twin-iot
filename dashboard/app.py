import sys
import os
import time

import streamlit as st
import pandas as pd

# allow importing from the other project folders
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "environment"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "digital_twin"))

from iot_network import IoTNetwork
from dqn_agent import DQNAgent
from twin_manager import TwinManager


st.set_page_config(page_title="Smart City Digital Twin", layout="wide")
st.title("🏙️ Smart City IoT — Multi-Agent RL + Digital Twin Dashboard")
st.caption(
    "Each device has its own AI agent deciding SLEEP vs SEND every step, "
    "to save energy while still clearing its data queue."
)


def state_to_vector(state_dict):
    return [state_dict["battery"], state_dict["queue"]]


# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")
num_devices = st.sidebar.slider("Number of devices", 2, 8, 3)
steps_per_click = st.sidebar.slider("Steps per click", 1, 20, 5)

if st.sidebar.button("🔄 Reset simulation"):
    st.session_state.clear()

# ---------------- Initialize simulation (only once) ----------------
if "network" not in st.session_state:
    st.session_state.network = IoTNetwork(num_devices=num_devices)
    st.session_state.agents = [DQNAgent(state_size=2, action_size=2) for _ in range(num_devices)]
    st.session_state.twin_manager = TwinManager(num_devices=num_devices)
    st.session_state.states = st.session_state.network.reset()
    st.session_state.twin_manager.sync_from_network(st.session_state.states)
    st.session_state.step_count = 0
    st.session_state.reward_history = []
    st.session_state.battery_history = []

network = st.session_state.network
agents = st.session_state.agents
twin_manager = st.session_state.twin_manager


# ---------------- Run simulation steps ----------------
def run_steps(n):
    states = st.session_state.states
    for _ in range(n):
        actions = []
        for i, agent in enumerate(agents):
            s = state_to_vector(states[i])
            actions.append(agent.act(s))

        rewards, next_states = network.step(actions)
        twin_manager.sync_from_network(next_states)

        for i, agent in enumerate(agents):
            s = state_to_vector(states[i])
            ns = state_to_vector(next_states[i])
            done = not next_states[i]["alive"]
            agent.remember(s, actions[i], rewards[i], ns, done)
            agent.train_step(batch_size=16)

        states = next_states
        st.session_state.step_count += 1
        st.session_state.reward_history.append(sum(rewards))
        avg_battery = sum(d.battery for d in network.devices) / len(network.devices)
        st.session_state.battery_history.append(avg_battery)

        if network.is_done():
            break

    st.session_state.states = states


col1, col2 = st.columns([1, 3])
with col1:
    if st.button("▶️ Step simulation"):
        run_steps(steps_per_click)

with col2:
    st.write(f"**Total steps run:** {st.session_state.step_count}")


# ---------------- Live digital twin table ----------------
st.subheader("📡 Live Digital Twin Status")

twin_states = twin_manager.get_all_states()
rows = []
for device_id, s in twin_states.items():
    rows.append({
        "Device": f"Device {device_id}",
        "Battery (%)": s["battery"],
        "Pending Data": s["queue"],
        "Status": "🟢 Alive" if s["alive"] else "🔴 Dead",
    })
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

st.progress(
    min(1.0, sum(r["Battery (%)"] for r in rows) / (100 * len(rows))),
    text="Average battery across all devices",
)


# ---------------- Charts ----------------
st.subheader("📈 Training Progress")
c1, c2 = st.columns(2)

with c1:
    st.caption("Reward per step (higher = better decisions)")
    if st.session_state.reward_history:
        st.line_chart(st.session_state.reward_history)

with c2:
    st.caption("Average battery over time")
    if st.session_state.battery_history:
        st.line_chart(st.session_state.battery_history)

st.info(
    "Click **Step simulation** repeatedly (or increase 'Steps per click') "
    "to watch the agents learn — battery should stay higher and rewards "
    "should trend upward as they train."
)