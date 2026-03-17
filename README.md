# Parameter-Shape-Inference-Bench

This repository includes a dataset generator for black-box neural-network parameter shape inference tasks.

## Generate dataset

```bash
python generate_dataset.py --num_tasks 1000 --overwrite
```

By default this writes tasks to `parameter_shape_inference_bench/` with this structure:

- `task_xxxx/task.json`
- `task_xxxx/forward.py`
- `task_xxxx/ground_truth.json`
- `task_xxxx/verify.py`
- `task_xxxx/example_usage.py`

## Notes

- Supported task types:
  - `relu_one_layer`
  - `relu_two_layer`
  - `leaky_relu_network`
  - `maxout_network`
  - `residual_relu_network`
- Parameter sampling follows the requested rules:
  - weights sampled from `Normal(0, 0.5)`
  - biases sampled from `Uniform(-1, 1)`
  - hidden width in `[4, 30]`
  - second-layer width in `[2, 20]`
