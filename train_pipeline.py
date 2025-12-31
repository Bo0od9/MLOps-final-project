import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from src import config
from src.dataset import YambdaDataset, collate_fn
from src.model import SASRecBackbone, SASRecInBatch, SASRecInBatchWithLogQ, SASRecReal
from src.preprocessing import compute_item_statistics, load_and_process_yambda
from src.trainer import train


def main(model_type="real"):
    config.set_seed(1337)

    print(f"Device: {config.DEVICE}")

    train_data, valid_data, eval_data, item_mapping = load_and_process_yambda(
        test_timestamp=config.TEST_TIMESTAMP, val_size=config.VAL_SIZE
    )

    train_dataset = YambdaDataset(dataframe=train_data, max_seq_len=config.MAX_SEQ_LEN)
    valid_dataset = YambdaDataset(dataframe=valid_data, max_seq_len=config.MAX_SEQ_LEN)
    # eval_dataset = YambdaDataset(dataframe=eval_data, max_seq_len=config.MAX_SEQ_LEN)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    backbone = SASRecBackbone(
        num_items=len(item_mapping),
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout_rate=config.DROPOUT_RATE,
        num_transformer_layers=config.NUM_TRANSFORMER_LAYERS,
    )

    if model_type == "real":
        model = SASRecReal(backbone=backbone).to(config.DEVICE)
    elif model_type == "inbatch":
        model = SASRecInBatch(backbone=backbone, num_negatives=config.TRAIN_BATCH_SIZE).to(config.DEVICE)
    elif model_type == "logq":
        print("Computing item statistics for LogQ...")
        item_freqs = torch.zeros(len(item_mapping), dtype=torch.float32)
        item_statistics, num_labels = compute_item_statistics(train_dataset)
        for key, val in item_statistics.items():
            item_freqs[key] = val / num_labels

        model = SASRecInBatchWithLogQ(
            backbone=backbone,
            num_negatives=config.TRAIN_BATCH_SIZE,
            item_freqs=item_freqs,
        ).to(config.DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    print(f"Starting training for model: {model_type}")
    best_checkpoint = train(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        model=model,
        optimizer=optimizer,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE,
    )

    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/sasrec_{model_type}_best.pt"
    torch.save(best_checkpoint, save_path)
    print(f"Saved best model to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="real",
        choices=["real", "inbatch", "logq"],
        help="Variant of SASRec to train",
    )
    args = parser.parse_args()
    main(args.model)
