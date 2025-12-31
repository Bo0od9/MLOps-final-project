import json
import os
import sys
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append("/app")

from src import config
from src.dataset import YambdaDataset, collate_fn
from src.model import SASRecBackbone, SASRecReal
from src.preprocessing import load_and_process_db
from src.trainer import train

POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://sasrec:password@postgres:5432/sasrec"
)
CHECKPOINT_DIR = "/app/checkpoints"


def main():
    print("Trainer Service Starting...")

    print(f"Connecting to DB: {POSTGRES_URL}")
    train_data, valid_data, _, item_mapping = load_and_process_db(
        db_url=POSTGRES_URL, min_seq_len=config.MIN_SEQ_LEN
    )

    num_items = len(item_mapping) + 1
    print(f"Data Loaded. Num Items: {num_items} (including padding)")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    vocab_path = os.path.join(CHECKPOINT_DIR, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(item_mapping, f)
    print(f"Vocab saved to {vocab_path}")

    model_config = {
        "num_items": num_items,
        "embedding_dim": config.EMBEDDING_DIM,
        "num_heads": config.NUM_HEADS,
        "max_seq_len": config.MAX_SEQ_LEN,
        "num_transformer_layers": config.NUM_TRANSFORMER_LAYERS,
        "dropout_rate": config.DROPOUT_RATE,
    }
    config_path = os.path.join(CHECKPOINT_DIR, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f)
    print(f"Model config saved to {config_path}")

    train_dataset = YambdaDataset(dataframe=train_data, max_seq_len=config.MAX_SEQ_LEN)
    valid_dataset = YambdaDataset(dataframe=valid_data, max_seq_len=config.MAX_SEQ_LEN)

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
        num_items=num_items,
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout_rate=config.DROPOUT_RATE,
        num_transformer_layers=config.NUM_TRANSFORMER_LAYERS,
    )

    model = SASRecReal(backbone=backbone).to(config.DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    print("Starting Training Loop...")
    best_checkpoint = train(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        model=model,
        optimizer=optimizer,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE,
    )

    save_path = os.path.join(CHECKPOINT_DIR, "sasrec_real_best.pt")
    torch.save(best_checkpoint, save_path)
    print(f"Saved best model to {save_path}")

    try:
        from kafka import KafkaProducer

        KAFKA_BOOTSTRAP_SERVERS = os.environ.get(
            "KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"
        )
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        message = {"event": "MODEL_UPDATED", "timestamp": time.time()}
        producer.send("system.control", message)
        producer.flush()
        print(f"Sent MODEL_UPDATED event to system.control")
    except Exception as e:
        print(f"Failed to send reload event: {e}")

    print("Training Complete.")


if __name__ == "__main__":
    main()
