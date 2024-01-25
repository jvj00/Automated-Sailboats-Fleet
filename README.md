# Automated-Sailboats-Fleet

# Version
Python 3.6+

# Setup

```bash
sudo apt install python3-tk
```

## Venv
```bash

python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

## Test
```bash
python src/test_actuator.py && python src/test_entities.py && python src/test_sensor.py && python src/test_utils.py && python src/test_simulations.py
