import random


class IoTDevice:
    def __init__(self, device_id, initial_battery=100.0):
        self.device_id = device_id
        self.battery = initial_battery          # current energy %
        self.data_queue = random.randint(0, 5)   # pending data packets
        self.alive = True

    def step(self, action):
        """
        action: 0 -> sleep, 1 -> send data
        returns: reward (based on energy saved or data sent)
        """
        reward = 0.0

        if not self.alive:
            return reward

        if action == 1:  # SEND
            cost = 2.0
            self.battery -= cost

            if self.data_queue > 0:
                self.data_queue -= 1
                # sending is good when there IS data waiting,
                # and even better when the queue was getting long
                reward += 1.0 + (0.3 * self.data_queue)
            else:
                # sent data but there was nothing to send -> wasted energy
                reward -= 1.0

        else:  # SLEEP
            cost = 0.2
            self.battery -= cost

            if self.data_queue == 0:
                reward += 0.5   # nothing pending -> sleeping was the smart move
            elif self.data_queue <= 2:
                reward += 0.1   # a little pending, sleeping is still okay
            else:
                # queue is piling up -> sleeping was a bad call
                reward -= 0.5 * self.data_queue

        # new data occasionally arrives (like a real sensor)
        if random.random() < 0.3:
            self.data_queue += 1

        if self.battery <= 0:
            self.battery = 0
            self.alive = False
            reward -= 10.0   # device died -> big penalty

        return reward

    def get_state(self):
        """This is the state (observation) the RL agent will see"""
        return {
            "battery": round(self.battery, 2),
            "queue": self.data_queue,
            "alive": self.alive,
        }


class IoTNetwork:
    def __init__(self, num_devices=5):
        self.devices = [IoTDevice(device_id=i) for i in range(num_devices)]

    def step(self, actions):
        """
        actions: list of 0/1, one action per device
        returns: list of rewards, list of states
        """
        rewards = []
        for device, action in zip(self.devices, actions):
            r = device.step(action)
            rewards.append(r)
        states = [d.get_state() for d in self.devices]
        return rewards, states

    def reset(self):
        for d in self.devices:
            d.battery = 100.0
            d.data_queue = random.randint(0, 5)
            d.alive = True
        return [d.get_state() for d in self.devices]

    def is_done(self):
        return all(not d.alive for d in self.devices)


# ---- Quick test: run this file directly with "python iot_network.py" ----
if __name__ == "__main__":
    network = IoTNetwork(num_devices=3)
    states = network.reset()
    print("Start states:", states)

    for t in range(10):
        # random actions for now - later the RL agent will decide these
        actions = [random.choice([0, 1]) for _ in network.devices]
        rewards, states = network.step(actions)
        print(f"\nStep {t+1}")
        print("Actions:", actions)
        print("Rewards:", rewards)
        print("States:", states)

        if network.is_done():
            print("All devices ran out of battery!")
            break