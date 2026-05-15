# Machine Learning Models in Hospital Admission Predictions


## Requirements
Docker Compose

## Running the machine learning algorithms
Unpack both `datasets.zip`  and `datasets_encoded.zip`

Run ```docker compose up --build``` to build a docker container and run the all of the following:
- Latency Comparison Test
- Noise Test for Robustness
- Optimization Demonstration

Results for each test are stored in `params/latency.log`, `params/noise.log`, and `params/demo.log` in the docker volume respectively

To run individual tests, run the following form the root of the project folder: 

| Test          | Command                 | Option                               |
|---------------|-------------------------|--------------------------------------|
| Latency Test  | python tests/latency.py |                                      |
| Noise Test    | python tests/noise.py   |                                      |
| Optimization  | python tests/demo.py    | -c for complete dataset optimization |
| Shap analysis | python shappy.py        |                                      |




