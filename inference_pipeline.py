import os
import sys

import torch

sys.path.append(os.getcwd())

from src import config
from src.model import SASRecBackbone, SASRecModel
from src.utils import get_last


def load_model(checkpoint_path, num_items):
    backbone = SASRecBackbone(
        num_items=num_items,
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout_rate=config.DROPOUT_RATE,
        num_transformer_layers=config.NUM_TRANSFORMER_LAYERS,
    )
    model = SASRecModel(backbone)

    state_dict = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(config.DEVICE)
    model.eval()
    return model


def predict_next_item(model, item_sequence):
    """
    Predicts scores for the next item given a sequence of past items.
    """
    if len(item_sequence) > config.MAX_SEQ_LEN:
        item_sequence = item_sequence[-config.MAX_SEQ_LEN :]

    length = len(item_sequence)
    positions = list(range(length))

    inputs = {
        "history": {
            "item_id": torch.tensor([item_sequence], dtype=torch.long).to(config.DEVICE),
        },
        "labels": {"item_id": torch.zeros(1, dtype=torch.long).to(config.DEVICE)},
    }

    inputs["history"]["item_id"] = inputs["history"]["item_id"].flatten()
    inputs["history"]["positions"] = torch.tensor(positions, dtype=torch.long).to(config.DEVICE)
    inputs["history"]["lengths"] = torch.tensor([length], dtype=torch.long).to(config.DEVICE)
    inputs["labels"]["item_id"] = torch.tensor([0] * length, dtype=torch.long).to(config.DEVICE)

    with torch.no_grad():
        outputs = model(inputs)
        all_scores = outputs["all_scores"]

    top_k_scores, top_k_indices = torch.topk(all_scores, k=10, dim=-1)

    return top_k_indices[0].tolist(), top_k_scores[0].tolist()


if __name__ == "__main__":
    print("Inference Demo")
    print("Pre-requisite: Trained model checkpoint and known num_items")
    # model = load_model('checkpoints/sasrec_real_best.pt', num_items=304787)
    # recommendations = predict_next_item(model, [10, 20, 30])
    # print(recommendations)
