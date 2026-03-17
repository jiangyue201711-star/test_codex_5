#!/usr/bin/env python3
"""Generate Parameter-Shape-Inference-Bench tasks."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable


DEFAULT_INPUT_DIM = 10


@dataclass
class TaskSpec:
    task_type: str
    target_parameter: str
    description: str
    forward_source_builder: Callable[[dict, int], str]


def _array_literal(values):
    return json.dumps(values)


def build_relu_one_layer_forward(params: dict, seed: int) -> str:
    return f'''import numpy as np

np.random.seed({seed})

A1 = np.array({_array_literal(params["A1"])}, dtype=float)
b1 = np.array({_array_literal(params["b1"])}, dtype=float)
A2 = np.array({_array_literal(params["A2"])}, dtype=float)
b2 = np.array({_array_literal(params["b2"])}, dtype=float)


def relu(x):
    return np.maximum(0.0, x)


def forward(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    h = relu(A1 @ x + b1)
    y = A2 @ h + b2
    return float(y)
'''


def build_relu_two_layer_forward(params: dict, seed: int) -> str:
    return f'''import numpy as np

np.random.seed({seed})

W1 = np.array({_array_literal(params["W1"])}, dtype=float)
b1 = np.array({_array_literal(params["b1"])}, dtype=float)
W2 = np.array({_array_literal(params["W2"])}, dtype=float)
b2 = np.array({_array_literal(params["b2"])}, dtype=float)
W3 = np.array({_array_literal(params["W3"])}, dtype=float)
b3 = np.array({_array_literal(params["b3"])}, dtype=float)


def relu(x):
    return np.maximum(0.0, x)


def forward(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    h1 = relu(W1 @ x + b1)
    h2 = relu(W2 @ h1 + b2)
    y = W3 @ h2 + b3
    return float(y)
'''


def build_leaky_relu_forward(params: dict, seed: int) -> str:
    return f'''import numpy as np

np.random.seed({seed})

A1 = np.array({_array_literal(params["A1"])}, dtype=float)
b1 = np.array({_array_literal(params["b1"])}, dtype=float)
A2 = np.array({_array_literal(params["A2"])}, dtype=float)
b2 = np.array({_array_literal(params["b2"])}, dtype=float)


def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0.0, x, alpha * x)


def forward(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    h = leaky_relu(A1 @ x + b1)
    y = A2 @ h + b2
    return float(y)
'''


def build_maxout_forward(params: dict, seed: int) -> str:
    return f'''import numpy as np

np.random.seed({seed})

A1 = np.array({_array_literal(params["A1"])}, dtype=float)
b1 = np.array({_array_literal(params["b1"])}, dtype=float)
A3 = np.array({_array_literal(params["A3"])}, dtype=float)
b3 = np.array({_array_literal(params["b3"])}, dtype=float)
A2 = np.array({_array_literal(params["A2"])}, dtype=float)
b2 = np.array({_array_literal(params["b2"])}, dtype=float)


def forward(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    z1 = A1 @ x + b1
    z3 = A3 @ x + b3
    h = np.maximum(z1, z3)
    y = A2 @ h + b2
    return float(y)
'''


def build_residual_relu_forward(params: dict, seed: int) -> str:
    return f'''import numpy as np

np.random.seed({seed})

W1 = np.array({_array_literal(params["W1"])}, dtype=float)
b1 = np.array({_array_literal(params["b1"])}, dtype=float)
W2 = np.array({_array_literal(params["W2"])}, dtype=float)
P = np.array({_array_literal(params["P"])}, dtype=float)
c = np.array({_array_literal(params["c"])}, dtype=float)


def relu(x):
    return np.maximum(0.0, x)


def forward(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    residual = P @ x + c
    hidden = relu(W1 @ x + b1)
    y = residual + W2 @ hidden
    return float(y)
'''


def sample_normal_matrix(rng: random.Random, rows: int, cols: int) -> list[list[float]]:
    return [[rng.gauss(0.0, 0.5) for _ in range(cols)] for _ in range(rows)]


def sample_uniform_vector(rng: random.Random, size: int) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(size)]


def sample_task_parameters(task_type: str, input_dim: int, rng: random.Random) -> dict:
    hidden = rng.randint(4, 30)
    second = rng.randint(2, 20)

    if task_type == "relu_one_layer":
        return {
            "A1": sample_normal_matrix(rng, hidden, input_dim),
            "b1": sample_uniform_vector(rng, hidden),
            "A2": sample_normal_matrix(rng, 1, hidden),
            "b2": sample_uniform_vector(rng, 1),
        }
    if task_type == "relu_two_layer":
        return {
            "W1": sample_normal_matrix(rng, hidden, input_dim),
            "b1": sample_uniform_vector(rng, hidden),
            "W2": sample_normal_matrix(rng, second, hidden),
            "b2": sample_uniform_vector(rng, second),
            "W3": sample_normal_matrix(rng, 1, second),
            "b3": sample_uniform_vector(rng, 1),
        }
    if task_type == "leaky_relu_network":
        return {
            "A1": sample_normal_matrix(rng, hidden, input_dim),
            "b1": sample_uniform_vector(rng, hidden),
            "A2": sample_normal_matrix(rng, 1, hidden),
            "b2": sample_uniform_vector(rng, 1),
        }
    if task_type == "maxout_network":
        return {
            "A1": sample_normal_matrix(rng, hidden, input_dim),
            "b1": sample_uniform_vector(rng, hidden),
            "A3": sample_normal_matrix(rng, hidden, input_dim),
            "b3": sample_uniform_vector(rng, hidden),
            "A2": sample_normal_matrix(rng, 1, hidden),
            "b2": sample_uniform_vector(rng, 1),
        }
    if task_type == "residual_relu_network":
        return {
            "W1": sample_normal_matrix(rng, hidden, input_dim),
            "b1": sample_uniform_vector(rng, hidden),
            "W2": sample_normal_matrix(rng, 1, hidden),
            "P": sample_normal_matrix(rng, 1, input_dim),
            "c": sample_uniform_vector(rng, 1),
        }
    raise ValueError(f"Unsupported task type: {task_type}")


def parameter_shapes(params: dict) -> dict:
    shapes = {}
    for name, value in params.items():
        if value and isinstance(value[0], list):
            shapes[f"{name}_shape"] = [len(value), len(value[0])]
        else:
            shapes[f"{name}_shape"] = [len(value)]
    return shapes


TASK_SPECS: list[TaskSpec] = [
    TaskSpec(
        task_type="relu_one_layer",
        target_parameter="A1",
        description="You can query forward(x). The hidden model is a one-layer ReLU network. Infer the shape of matrix A1.",
        forward_source_builder=build_relu_one_layer_forward,
    ),
    TaskSpec(
        task_type="relu_two_layer",
        target_parameter="W1",
        description="You can query forward(x). The hidden model is a two-layer ReLU network. Infer the shape of matrix W1.",
        forward_source_builder=build_relu_two_layer_forward,
    ),
    TaskSpec(
        task_type="leaky_relu_network",
        target_parameter="A1",
        description="You can query forward(x). The hidden model is a one-layer LeakyReLU network. Infer the shape of matrix A1.",
        forward_source_builder=build_leaky_relu_forward,
    ),
    TaskSpec(
        task_type="maxout_network",
        target_parameter="A1",
        description="You can query forward(x). The hidden model is a maxout network. Infer the shape of matrix A1.",
        forward_source_builder=build_maxout_forward,
    ),
    TaskSpec(
        task_type="residual_relu_network",
        target_parameter="W1",
        description="You can query forward(x). The hidden model is a residual ReLU network. Infer the shape of matrix W1.",
        forward_source_builder=build_residual_relu_forward,
    ),
]


def write_verify_script(task_dir: Path) -> None:
    source = '''import json
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    solution_path = base / "solution.json"
    if not solution_path.exists():
        print("FAIL: solution.json not found")
        return

    with open(base / "task.json", "r", encoding="utf-8") as f:
        task = json.load(f)
    with open(base / "ground_truth.json", "r", encoding="utf-8") as f:
        truth = json.load(f)
    with open(solution_path, "r", encoding="utf-8") as f:
        pred = json.load(f)

    key = f"{task['target_parameter']}_shape"
    if pred.get(key) == truth.get(key):
        print("SUCCESS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
'''
    (task_dir / "verify.py").write_text(source, encoding="utf-8")


def write_example_usage(task_dir: Path, input_dim: int) -> None:
    source = f'''import numpy as np
from forward import forward

x = np.random.randn({input_dim})
y = forward(x)
print(y)
'''
    (task_dir / "example_usage.py").write_text(source, encoding="utf-8")


def generate_dataset(num_tasks: int, output_dir: Path, input_dim: int, query_limit: int, seed: int, overwrite: bool) -> None:
    rng = random.Random(seed)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, num_tasks + 1):
        task_id = f"task_{i:04d}"
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        spec = rng.choice(TASK_SPECS)
        task_seed = rng.randint(0, 2**31 - 1)
        local_rng = random.Random(task_seed)

        params = sample_task_parameters(spec.task_type, input_dim, local_rng)
        truth = parameter_shapes(params)

        task_json = {
            "task_id": task_id,
            "task_type": spec.task_type,
            "input_dimension": input_dim,
            "target_parameter": spec.target_parameter,
            "query_limit": query_limit,
            "description": spec.description,
        }

        (task_dir / "task.json").write_text(json.dumps(task_json, indent=2), encoding="utf-8")
        (task_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
        (task_dir / "forward.py").write_text(spec.forward_source_builder(params, task_seed), encoding="utf-8")

        write_verify_script(task_dir)
        write_example_usage(task_dir, input_dim)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Parameter-Shape-Inference-Bench dataset")
    parser.add_argument("--num_tasks", type=int, required=True, help="Number of tasks to generate")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("parameter_shape_inference_bench"),
        help="Path to dataset root directory",
    )
    parser.add_argument("--input_dimension", type=int, default=DEFAULT_INPUT_DIM)
    parser.add_argument("--query_limit", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Remove output_dir before generating")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        num_tasks=args.num_tasks,
        output_dir=args.output_dir,
        input_dim=args.input_dimension,
        query_limit=args.query_limit,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
