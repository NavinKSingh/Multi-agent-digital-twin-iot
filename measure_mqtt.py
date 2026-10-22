"""
Measures MQTT synchronization between simulated devices and the digital twin:
  - delay from publish() until the twin object has been updated
  - message loss (sent vs. received)

Usage:  python measure_mqtt.py
        python measure_mqtt.py --host broker.hivemq.com --devices 5 --steps 200
Needs: pip install paho-mqtt     (and internet for the public broker)
Result is printed and saved to results/mqtt_results.json
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import uuid

import paho.mqtt.client as mqtt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "digital_twin"))
from twin_manager import TwinManager


def make_client():
    try:                                   # paho-mqtt >= 2.0
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:                 # paho-mqtt 1.x
        return mqtt.Client()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="broker.hivemq.com")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--devices", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--interval", type=float, default=0.1, help="seconds between steps")
    args = ap.parse_args()

    prefix = f"smartcity/measure_{uuid.uuid4().hex[:8]}/device"   # private topic
    twin = TwinManager(num_devices=args.devices)
    latencies, received = [], set()
    lock = threading.Lock()
    subscribed = threading.Event()

    sub = make_client()

    def on_connect(c, u, f, rc):
        c.subscribe(f"{prefix}/+/state")      # single-level wildcard, default QoS 0

    def on_subscribe(c, u, mid, granted):
        subscribed.set()

    def on_message(c, u, msg):
        state = json.loads(msg.payload.decode())
        dev = int(msg.topic.split("/")[-2])
        twin.twins[dev].update({k: state[k] for k in ("battery", "queue", "alive")})
        t_done = time.perf_counter()                    # twin is updated now
        with lock:
            latencies.append((t_done - state["t_send"]) * 1000.0)   # ms
            received.add((dev, state["seq"]))

    sub.on_connect, sub.on_subscribe, sub.on_message = on_connect, on_subscribe, on_message
    sub.connect(args.host, args.port, keepalive=60)
    sub.loop_start()
    if not subscribed.wait(15):
        sys.exit("Could not subscribe - check internet / broker.")

    pub = make_client()
    pub.connect(args.host, args.port, keepalive=60)
    pub.loop_start()
    time.sleep(1)

    sent = 0
    for step in range(args.steps):
        for dev in range(args.devices):
            payload = {"battery": 100.0 - step * 0.3, "queue": step % 5, "alive": True,
                       "seq": step, "t_send": time.perf_counter()}
            pub.publish(f"{prefix}/{dev}/state", json.dumps(payload))   # QoS 0
            sent += 1
        time.sleep(args.interval)
    time.sleep(3)                                       # let stragglers arrive
    pub.loop_stop(); sub.loop_stop()

    with lock:
        n_recv = len(received)
        lat = sorted(latencies)
    res = {
        "broker": args.host, "devices": args.devices, "steps": args.steps,
        "sent": sent, "received": n_recv,
        "loss_percent": round(100.0 * (sent - n_recv) / sent, 2),
        "latency_ms_mean": round(statistics.mean(lat), 1) if lat else None,
        "latency_ms_median": round(statistics.median(lat), 1) if lat else None,
        "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1) if lat else None,
        "latency_ms_max": round(lat[-1], 1) if lat else None,
    }
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "mqtt_results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
