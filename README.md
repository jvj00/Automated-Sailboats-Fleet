# Automated-Sailboats-Fleet

## Version
Python 3.9+

## Dependencies
- [PyQt5](https://pypi.org/project/PyQt5)
- [Pandas](https://pypi.org/project/pandas)
- [Matplotlib](https://pypi.org/project/matplotlib)
- [Numpy](https://pypi.org/project/numpy)
- [graphics.py](https://pypi.org/project/graphics.py)

## Setup
Start virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Install dependencies
```bash
pip install -r requirements.txt
```

## Run
From the root folder type
```bash
python ./src/main.py ./config.json
```
The file `config.json` has the following parameters that can be changed to do different experiments:
- `simulation`: config of simulation
  - `world`: dimensions of the world
    - `width`: in meters
    - `height`: in meters
  - `duration`: experiment max duration in seconds
  - `update_period`: the discrete-time to update world (dt)
  - `boats`: number of boats (if 4x2 boats are 8)
  - `groups`: number of groups (if 4x2 groups are 4)
  - `random_target_enabled`: if true, targets are randomly generated
  - `real_time_charts_enabled`: if true, charts are shown in real time
  - `real_time_drawings_enabled`: if true, a drawing with boats simulation (position, vectors, wind...) is shown
  - `save_folder`: relative folder where results are saved
- `filters`: config of EKF and Linear consensus
  - `ekf`
    -  `update_period`: rating with which EKF is computed
  -  `fleet`
    - `update_period`: rating with which messages are exchanged and Linear Consensus is computed
    - `update_probability`: probability to exchange a message
- `sensors`: config of sensors
  - `gnss`
    - `error`
      - `x`
        - `absolute`: absolute error of x-position of GNSS
      - `y`
        - `absolute`: absolute error of y-position of GNSS
    - `update_period`: update period of GNSS (dt_gnss)
    - `update_probability`: probability of measure GNSS
  - `compass`
    - `error`
      - `absolute`: absolute error of compass (in rad)
    - `update_period`: update period of compass (dt_compass)
    - `update_probability`: probability of measure compass
  - `anemometer`
    - `error`
      - `speed`
        - `relative`: relative error of anemometer speed
      - `angle`
        - `absolute`: absolute error of anemometer direction (in rad)
    - `update_period`: update period of anemometer
    - `update_probability`: probability of measure anemometer
  - `speedometer`
    - `error`
      - `threshold`: mixed error threshold of speedometer
      - `relative`: relative error over threshold of speedometer
    - `update_period`: update period of speedometer
    - `update_probability`: probability of measure speedometer
  - `sonar`
    - `error`
      - `relative`: relative error of sonar
    - `update_period`: update period of sonar
    - `update_probability`: probability of measure sonar
- `actuators`
  - `motor`
    - `resolution`: resolution of PWM that controls motor (power of 2)
  - `rudder`
    - `resolution`: resolution of rudder in number of step per revolution
  - `wing`
    - `resolution`: resolution of rudder in number of step per revolution

## Test
```bash
python src/test_actuator.py && python src/test_entities.py && python src/test_sensor.py && python src/test_utils.py && python src/test_simulations.py
