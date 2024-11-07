import random
from typing import Any, Callable, Dict, List, Optional

from datasets import load_dataset, Dataset


def load_data_batches(
    dataset_name: str,
    batch_size: int,
    num_batches: int,
    filter_function: Optional[Callable[[Dict[str, Any]], bool]] = None,
    success_probability: float = 0.1,
) -> List[List[str]]:
    dataset = load_dataset_from_hf(dataset_name)
    dataset = dataset.select(random.sample(range(len(dataset)), int(num_batches * batch_size / success_probability)))
    dataset = dataset.filter(filter_function, num_proc=32)
    dataset = dataset.select(random.sample(range(len(dataset)), num_batches * batch_size))
    iterable_dataset = iter(dataset)
    batches = []
    for i in range(num_batches):
        batch = []
        while len(batch) < batch_size:
            text = next(iterable_dataset)["text"]
            batch.append(text)
        batches.append(batch)
    return batches


def load_dataset_from_hf(dataset_name: str)-> Dataset:
    assert dataset_name in {
        "arxiv",
        "c4",
        "github",
        "stackexchange",
        "wikipedia"
    }, "other datasets not implemented"
    dataset = load_dataset(
        "togethercomputer/RedPajama-Data-1T",
        dataset_name,
        split="train",
        trust_remote_code=True,
    )
    return dataset


__all__ = ["load_data_batches", "load_dataset_from_hf"]
