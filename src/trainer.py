from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics
from .model import SASRecModel


def evaluation(
    dataloader: DataLoader,
    model: SASRecModel,
    device: str = "cpu",
    num_batches: Optional[int] = None,
) -> Dict[str, float]:
    eval_metrics = defaultdict(list)

    model.eval()

    for idx, batch in tqdm(enumerate(dataloader), desc="Evaluating"):
        for key in batch:
            if isinstance(batch[key], dict):
                for sub_key in batch[key]:
                    batch[key][sub_key] = batch[key][sub_key].to(device)
            else:
                assert isinstance(batch[key], torch.Tensor)
                batch[key] = batch[key].to(device)

        with torch.inference_mode():
            model_output = model(batch)
            batch_metrics = compute_metrics(model_output["all_scores"], model_output["positive_scores"])
        for key, values in batch_metrics.items():
            eval_metrics[key].extend(values)

        if num_batches is not None and idx + 1 >= num_batches:
            break

    final_metrics = {}
    for key, values in eval_metrics.items():
        final_metrics[key] = np.mean(values)

    return final_metrics


def train(
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int | None = None,
    device: str = "cpu",
    num_valid_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:

    step_num = 0
    epoch_num = 0

    best_checkpoint = None
    best_metric_name = "dcg@1000"
    best_metric_value = float("-inf")

    while num_epochs is None or epoch_num < num_epochs:
        print(f"Start epoch {epoch_num + 1}")
        running_loss = []

        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch_num + 1}"):

            for key in batch:
                if isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        batch[key][sub_key] = batch[key][sub_key].to(device)
                else:
                    assert isinstance(batch[key], torch.Tensor)
                    batch[key] = batch[key].to(device)

            output = model(batch)
            loss = output["loss"]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            step_num += 1

            running_loss.append(loss.item())

        print("Evaluating...")
        valid_metrics = evaluation(valid_dataloader, model, device, num_valid_batches)

        if best_metric_value is None or best_metric_value < valid_metrics[best_metric_name]:
            best_metric_value = valid_metrics[best_metric_name]
            best_checkpoint = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        msgs = []
        for metric_name, metrinc_value in valid_metrics.items():
            msgs.append(f"{metric_name}: {round(metrinc_value, 5)}")
        msg = ", ".join(msgs)
        print(f"Validation metrics: {msg}")

        print(f"Average loss at epoch #{epoch_num + 1}: {round(np.mean(running_loss), 5)}")

        epoch_num += 1

    print("Training completed!")

    return best_checkpoint
