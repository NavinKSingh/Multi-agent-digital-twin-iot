import pandas as pd
import random


class RealIoTDevice:
    def __init__(self, device_id, consumption_series, initial_battery=100.0):
        self.device_id = device_id
        self.battery = initial_battery
        self.data_queue = random.randint(0, 5)
        self.alive = True

        # the real recorded consumption values for this device, in time order
        self.consumption_series = list(consumption_series)
        self.time_index = 0

        # scale real kWh readings into a reasonable "% battery per step" cost
        avg = sum(self.consumption_series) / max(len(self.consumption_series), 1)
        self.scale = 2.0 / max(avg, 1e-6)   # so an "average" reading costs ~2% battery

    def _current_real_cost(self):
        """Look up today's real consumption reading, cycling through the data if needed."""
        value = self.consumption_series[self.time_index % len(self.consumption_series)]
        self.time_index += 1
        return value * self.scale

    def step(self, action):
        reward = 0.0
        if not self.alive:
            return reward

        real_cost = self._current_real_cost()

        if action == 1:  # SEND
            cost = real_cost            # full real-world cost when actively sending
            self.battery -= cost

            if self.data_queue > 0:
                self.data_queue -= 1
                reward += 1.0 + (0.3 * self.data_queue)
            else:
                reward -= 1.0

        else:  # SLEEP
            cost = real_cost * 0.1      # sleeping cuts real consumption to ~10%
            self.battery -= cost

            if self.data_queue == 0:
                reward += 0.5
            elif self.data_queue <= 2:
                reward += 0.1
            else:
                reward -= 0.5 * self.data_queue

        if random.random() < 0.3:
            self.data_queue += 1

        if self.battery <= 0:
            self.battery = 0
            self.alive = False
            reward -= 10.0

        return reward

    def get_state(self):
        return {
            "battery": round(self.battery, 2),
            "queue": self.data_queue,
            "alive": self.alive,
        }


class RealIoTNetwork:
    def __init__(self, csv_path, home_id_col="household_id", value_col="energy_consumption_kWh"):
        """
        csv_path: path to the dataset CSV
        home_id_col: column that identifies which device/household a row belongs to
        value_col: column with the real energy consumption reading
        """
        df = pd.read_csv(csv_path)

        # group real readings by device/household, keep them in original order
        grouped = df.groupby(home_id_col)[value_col].apply(list)

        self.devices = []
        for i, (home_id, series) in enumerate(grouped.items()):
            self.devices.append(RealIoTDevice(device_id=i, consumption_series=series))

        print(f"Loaded {len(self.devices)} devices from real data "
              f"({len(df)} total readings from '{csv_path}')")

    def step(self, actions):
        rewards = []
        for device, action in zip(self.devices, actions):
            rewards.append(device.step(action))
        states = [d.get_state() for d in self.devices]
        return rewards, states

    def reset(self):
        for d in self.devices:
            d.battery = 100.0
            d.data_queue = random.randint(0, 5)
            d.alive = True
            # start each episode at a random position of the household's series
            d.time_index = random.randrange(len(d.consumption_series))
        return [d.get_state() for d in self.devices]

    def is_done(self):
        return all(not d.alive for d in self.devices)


# ---- Quick test: run this file directly with "python iot_network_real.py" ----
if __name__ == "__main__":
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "energy_data.csv")

    network = RealIoTNetwork(csv_path)
    states = network.reset()
    print("Start states:", states)

    for t in range(10):
        actions = [random.choice([0, 1]) for _ in network.devices]
        rewards, states = network.step(actions)
        print(f"\nStep {t + 1}")
        print("States:", states)

        if network.is_done():
            print("All devices ran out of battery!")
            break