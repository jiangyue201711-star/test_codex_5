import json
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
