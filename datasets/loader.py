"""Dataset loading utilities."""

import json
from pathlib import Path

from eval.runner import EvalSample


def load_json_dataset(path: str | Path) -> list[EvalSample]:
    """Load a JSON dataset file into a list of EvalSample objects.

    The JSON file should contain a list of objects, each with at minimum
    'query' and 'expected_answer' fields.

    Args:
        path: Path to the JSON file.

    Returns:
        A list of EvalSample objects.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    samples: list[EvalSample] = []
    for item in data:
        sample = EvalSample(
            query=item["query"],
            expected_answer=item["expected_answer"],
            context=item.get("context", ""),
            metadata=item.get("metadata", {}),
        )
        samples.append(sample)

    return samples
