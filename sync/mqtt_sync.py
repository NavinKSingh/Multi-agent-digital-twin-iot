import sys
import os
import json
import time
import random

import paho.mqtt.client as mqtt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "environment"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "digital_twin"))

from iot_network import IoTNetwork
from twin_manager import TwinManager


BROKER_HOST = "broker.hivemq.com"   # free public test broker
BROKER_PORT = 1883
TOPIC_PREFIX = "smartcity/navin_project/device"   # unique-ish prefix so we don't clash with others


def topic_for(device_id):
    return f"{TOPIC_PREFIX}/{device_id}/state"


# ==================================================================
# ROLE 1: DevicePublisher - simulates real devices sending their state
# ==================================================================
class DevicePublisher:
    def __init__(self, num_devices=3):
        self.network = IoTNetwork(num_devices=num_devices)
        self.client = mqtt.Client()
        self.client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    def run(self, steps=50, delay_seconds=1.0):
        states = self.network.reset()
        self._publish_all(states)

        for step in range(steps):
            actions = [random.choice([0, 1]) for _ in self.network.devices]
            rewards, states = self.network.step(actions)
            self._publish_all(states)

            print(f"[Publisher] Step {step + 1}: published states for {len(states)} devices")
            time.sleep(delay_seconds)

            if self.network.is_done():
                print("[Publisher] All devices died, stopping.")
                break

    def _publish_all(self, states):
        for device_id, state in enumerate(states):
            payload = json.dumps(state)
            self.client.publish(topic_for(device_id), payload)


# ==================================================================
# ROLE 2: TwinSubscriber - keeps the digital twin updated from MQTT
# ==================================================================
class TwinSubscriber:
    def __init__(self, num_devices=3):
        self.twin_manager = TwinManager(num_devices=num_devices)
        self.num_devices = num_devices

        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    def _on_connect(self, client, userdata, flags, rc):
        print(f"[Subscriber] Connected to broker (code={rc}), subscribing...")
        client.subscribe(f"{TOPIC_PREFIX}/+/state")   # '+' matches any device id

    def _on_message(self, client, userdata, msg):
        # topic looks like: smartcity/navin_project/device/2/state
        parts = msg.topic.split("/")
        device_id = int(parts[-2])
        state = json.loads(msg.payload.decode())

        self.twin_manager.twins[device_id].update(state)
        print(f"[Subscriber] Twin updated for device {device_id}: {state}")

    def run(self, listen_seconds=60):
        print(f"[Subscriber] Listening for {listen_seconds} seconds...")
        self.client.loop_start()
        time.sleep(listen_seconds)
        self.client.loop_stop()

        print("\n[Subscriber] Final digital twin snapshot:")
        print(self.twin_manager.summary())


# ==================================================================
# Run this file with a role argument:
#   python mqtt_sync.py publisher
#   python mqtt_sync.py subscriber
# Run the subscriber FIRST (in one terminal), then the publisher
# (in a second terminal) so the subscriber is ready to catch messages.
# ==================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("publisher", "subscriber"):
        print("Usage: python mqtt_sync.py [publisher|subscriber]")
        sys.exit(1)

    role = sys.argv[1]

    if role == "publisher":
        DevicePublisher(num_devices=3).run(steps=30, delay_seconds=1.0)
    else:
        TwinSubscriber(num_devices=3).run(listen_seconds=40)