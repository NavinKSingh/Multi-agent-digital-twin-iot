# Multi-Agent Deep RL with a Digital Twin for Energy-Efficient Smart City IoT

Each battery-powered IoT device is controlled by its own Deep Q-Network (DQN) agent that decides, from its
battery level and pending-data queue, whether to **send** or **sleep**. A digital twin keeps a live replica of
every device, synchronized over **MQTT**, and a **Streamlit** dashboard shows the twin and the learning progress.
The DQN is implemented from scratch in NumPy.

The paper describing this work is in [`paper/`](paper/) (IEEE conference format).

## Results (10 seeds, 30 evaluation episodes, mean ± std)

Real-data environment (5 devices, London Smart Meters energy costs):

| Policy | Battery (%) | Queue | Return | Alive (%) |
|---|---|---|---|---|
| DQN (ours) | 75.6 ± 0.4 | 1.05 ± 0.20 | 91.9 ± 6.8 | 99.5 |
| Random | 66.1 ± 0.3 | 1.15 ± 0.07 | 26.6 ± 4.4 | 99.1 |
| Always-send | 40.6 | 0.29 | −33.3 | 92.0 |
| Always-sleep | 93.9 | 11.55 | −508.7 | 100 |
| Queue-threshold | 73.4 | 0.31 | 109.5 | 99.3 |

The DQN keeps about 12–14 % more battery than a random policy with more than three times its return.
A simple hand-designed queue-threshold rule gets a **higher return** than the DQN, because the 30-step episodes
rarely let the battery limit behavior. The paper discusses this openly. Full tables for both environments are
in `results/results.json`.

MQTT synchronization (public broker, QoS 0, 1000 messages): 0 % loss, median delay 291 ms
(`results/mqtt_results.json`; single run, publisher and subscriber on one machine).

## Project structure

```
agents/dqn_agent.py          DQN agent (NumPy, one per device)
environment/iot_network.py   simulated IoT environment
environment/iot_network_real.py  environment driven by real energy readings
digital_twin/twin_manager.py digital twin (one twin per device)
sync/mqtt_sync.py            MQTT publisher / subscriber demo
dashboard/app.py             Streamlit dashboard
experiments.py               reproduces all paper results and figures
prepare_dataset.py           builds data/energy_data.csv from the London dataset
measure_mqtt.py              measures MQTT delay and loss
train.py, train_real.py, evaluate.py   simple training / evaluation scripts
data/energy_data.csv         5 households x 1000 half-hourly readings
results/                     figures and result files
paper/                       LaTeX source and PDF of the paper
```

## Setup

```
pip install -r requirements.txt
```

## Usage

```
python experiments.py            # 10 seeds, both environments (about 10 min on 2 cores)
python experiments.py --seeds 3  # quick test
python measure_mqtt.py           # MQTT delay/loss (needs internet)
streamlit run dashboard/app.py   # live dashboard
python sync/mqtt_sync.py subscriber    # then, in a second terminal:
python sync/mqtt_sync.py publisher
```

## Dataset

`data/energy_data.csv` is a 5-household extract of the
[London Smart Meters dataset](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
(UK Power Networks, Low Carbon London project, licence CC BY 4.0). To rebuild it, download one file of
`low-carbon-london-data-168-files`, put it in `data/`, and run:

```
python prepare_dataset.py data/LCL-June2015v2_0.csv
```

The raw file is not stored in this repository because of its size.
