import time


class DeviceTwin:
    """A virtual copy of ONE real device."""

    def __init__(self, device_id):
        self.device_id = device_id
        self.current_state = {
            "battery": 100.0,
            "queue": 0,
            "alive": True,
        }
        self.last_updated = time.time()
        self.history = []   # list of past states, oldest first

    def update(self, new_state):
        """
        Called whenever the real device's state changes.
        new_state: dict like {"battery": 87.5, "queue": 2, "alive": True}
        """
        self.history.append(self.current_state)
        if len(self.history) > 100:      # keep history from growing forever
            self.history.pop(0)

        self.current_state = dict(new_state)
        self.last_updated = time.time()

    def get_state(self):
        return self.current_state

    def get_history(self):
        return self.history


class TwinManager:
    """Manages ALL device twins for the whole city network."""

    def __init__(self, num_devices):
        self.twins = {
            device_id: DeviceTwin(device_id)
            for device_id in range(num_devices)
        }

    def sync_from_network(self, states):
        """
        Takes the real states from IoTNetwork.step()/reset() and
        updates every twin to match — this is the "synchronization" step.
        states: list of state dicts, one per device (in order)
        """
        for device_id, state in enumerate(states):
            self.twins[device_id].update(state)

    def get_all_states(self):
        """Returns the current virtual state of every device, as a dict."""
        return {
            device_id: twin.get_state()
            for device_id, twin in self.twins.items()
        }

    def get_device_history(self, device_id):
        return self.twins[device_id].get_history()

    def summary(self):
        """A quick readable summary of the whole city right now."""
        lines = []
        for device_id, twin in self.twins.items():
            s = twin.get_state()
            status = "ALIVE" if s["alive"] else "DEAD"
            lines.append(
                f"Device {device_id}: battery={s['battery']}%, "
                f"queue={s['queue']}, status={status}"
            )
        return "\n".join(lines)


# ---- Quick test: run this file directly with "python twin_manager.py" ----
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "environment"))
    from iot_network import IoTNetwork
    import random

    network = IoTNetwork(num_devices=3)
    twin_manager = TwinManager(num_devices=3)

    states = network.reset()
    twin_manager.sync_from_network(states)   # first sync

    for t in range(5):
        actions = [random.choice([0, 1]) for _ in network.devices]
        rewards, states = network.step(actions)

        # keep the digital twin in sync with the real (simulated) network
        twin_manager.sync_from_network(states)

        print(f"\n--- After step {t + 1} ---")
        print(twin_manager.summary())