from typing import *
from pathlib import Path
import argparse
import os
import random
import torch
import tyro
from src.dataset import load_data_batches
from src.models import load_model_tokenizer
from src.model_wrapper import ModelWrapper


def log_internals(
		experiment_root: Path,
		model_name: str,
		dataset_name: str,
		batch_size: int,
		num_batches: int,
		truncation_length: int,
		device_map: str = "auto",
		random_seed: int = 1234
):
	experiment_root.mkdir(parents=True, exist_ok=True)
	experiment_path = experiment_root / model_name / dataset_name / f"batch_size_{batch_size}_num_batches_{num_batches}_truncation_length_{truncation_length}"

	random.seed(random_seed)
	torch.random.manual_seed(random_seed)

	model, tokenizer = load_model_tokenizer(model_name, eval_mode=True, device_map=device_map)
	model_wrapper = ModelWrapper(model)

	dataset = load_data_batches(
		dataset_name, batch_size, num_batches,
		filter_function=lambda s: len(s) >= 10 * truncation_length
	)
	for batch_idx, batch in enumerate(dataset):
		batch_path = experiment_path / f"batch_{batch_idx}"
		batch_path.mkdir(parents=True, exist_ok=True)

		input_ids = tokenizer(batch, return_tensors="pt", truncation=True, max_length=truncation_length).input_ids
		decoded_batch = tokenizer.batch_decode(input_ids)
		tokenized_batch = [tokenizer.convert_ids_to_tokens(input_ids[i]) for i in range(len(batch))]

		text_path = batch_path / "text.pt"
		with open(text_path, "wb") as f:
			torch.save({"text": batch, "decoded_text": decoded_batch, "tokenized_text": tokenized_batch}, f)

		input_ids = input_ids.to(device)
		_, model_internals = model_wrapper.forward(input_ids, log_attn=log_attn, log_mlp=log_mlp, log_layer=log_latents, log_model=log_latents)

		internals_path = batch_path / "internals.pt"
		with open(internals_path, "wb") as f:
			torch.save(model_internals, f)


if __name__ == "__main__":
	tyro.cli(log_internals)
