# Machine Learning Models in Hospital Admission Predictions

## Running the machine learning algorithms
Unpack both `datasets.zip`  and `datasets_encoded.zip`

Run ```docker compose up --build``` to build a docker container and run the following:
- Latency Comparison Test
- Noise Test for Robustness
- Optimization Demonstration

Results for each test are stored in `params/latency.log`, `params/noise.log`, and `params/demo.log` in the docker volume respectively
