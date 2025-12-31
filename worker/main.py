import json
import os
import sys
import threading
import time
import uuid

import torch
from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import Counter, Histogram, start_http_server

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import config
from src.model import SASRecBackbone, SASRecModel

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")


NUM_ITEMS = 304787  # Fallback
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "sasrec_real_best.pt")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "model_config.json")
VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "vocab.json")

VOCAB = {}
REVERSE_VOCAB = {}

WORKER_PROCESSED_TOTAL = Counter("worker_processed_total", "Total messages processed by worker")
WORKER_ERRORS_TOTAL = Counter("worker_errors_total", "Total errors in worker processing")
WORKER_INFERENCE_LATENCY = Histogram("worker_inference_latency_seconds", "Time taken for inference")
WORKER_HISTORY_LENGTH = Histogram(
    "worker_history_length",
    "Length of user interaction history",
    buckets=[0, 5, 10, 20, 50, 100, 200, float("inf")],
)
WORKER_PREDICTION_SCORE = Histogram(
    "worker_prediction_score",
    "Distribution of top-k prediction scores",
    buckets=[0, 5, 10, 12, 14, 16, 18, 20, 25, float("inf")],
)


def load_vocab_and_config():
    global NUM_ITEMS, VOCAB, REVERSE_VOCAB

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                conf = json.load(f)
                NUM_ITEMS = conf.get("num_items", NUM_ITEMS)

                print(f"Loaded config. Num Items: {NUM_ITEMS}")
        except Exception as e:
            print(f"Error loading config: {e}")

    if os.path.exists(VOCAB_PATH):
        try:
            with open(VOCAB_PATH, "r") as f:
                VOCAB = json.load(f)
                VOCAB = {int(k): v for k, v in VOCAB.items()}
                REVERSE_VOCAB = {v: k for k, v in VOCAB.items()}
                print(f"Loaded vocab. Size: {len(VOCAB)}")
        except Exception as e:
            print(f"Error loading vocab: {e}")


def load_model():
    load_vocab_and_config()

    print(f"Loading model on {config.DEVICE}...")
    backbone = SASRecBackbone(
        num_items=NUM_ITEMS,
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout_rate=config.DROPOUT_RATE,
        num_transformer_layers=config.NUM_TRANSFORMER_LAYERS,
    )
    model = SASRecModel(backbone)
    if os.path.exists(CHECKPOINT_PATH):
        try:
            state_dict = torch.load(CHECKPOINT_PATH, map_location=config.DEVICE)
            model.load_state_dict(state_dict, strict=False)
            model.to(config.DEVICE)
            model.eval()
            print("Model loaded.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None


MODEL_LOCK = threading.Lock()
MODEL = None


def reload_model_safely():
    global MODEL
    print("Reloading model safely...")

    with MODEL_LOCK:
        print("Acquired lock for reload.")
        new_model = load_model()
        if new_model:
            MODEL = new_model
            print("Model reloaded successfully.")
        else:
            print("Failed to reload model.")


def control_consumer_loop():
    unique_group_id = f"worker_control_{uuid.uuid4()}"
    consumer = KafkaConsumer(
        "system.control",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id=unique_group_id,
        auto_offset_reset="latest",
    )
    print(f"Control consumer started with group_id: {unique_group_id}")

    for msg in consumer:
        try:
            event = msg.value
            if event.get("event") == "MODEL_UPDATED":
                print(f"Received MODEL_UPDATED event. Timestamp: {event.get('timestamp')}")
                reload_model_safely()
        except Exception as e:
            print(f"Error in control loop: {e}")


def main():
    global MODEL
    print("Worker starting...")

    start_http_server(8001)
    print("Metrics server started on port 8001")

    time.sleep(10)

    MODEL = load_model()
    if not MODEL:
        print("Failed to load model. Exiting.")
        return

    consumer = KafkaConsumer(
        "inference.requests",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="sasrec_worker_group",
        auto_offset_reset="latest",
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    print("Worker listening for messages...")

    control_thread = threading.Thread(target=control_consumer_loop, daemon=True)
    control_thread.start()

    for message in consumer:
        WORKER_PROCESSED_TOTAL.inc()
        data = message.value
        request_id = data.get("request_id")
        history = data.get("history")
        k = data.get("k", 10)

        if history:
            WORKER_HISTORY_LENGTH.observe(len(history))
        else:
            WORKER_HISTORY_LENGTH.observe(0)

        print(f"Processing request {request_id}")

        try:
            start_time = time.time()

            mapped_history = []

            with MODEL_LOCK:
                current_model = MODEL
                current_vocab = VOCAB
                current_reverse_vocab = REVERSE_VOCAB
                current_num_items = NUM_ITEMS

            if not current_model:
                print("Model not ready yet.")
                continue

            if current_vocab:
                for item in history:
                    if item in current_vocab:
                        mapped_history.append(current_vocab[item])
            else:
                mapped_history = history

            if not mapped_history:
                pass

            length = len(mapped_history)
            if length > config.MAX_SEQ_LEN:
                mapped_history = mapped_history[-config.MAX_SEQ_LEN :]
                length = config.MAX_SEQ_LEN

            positions = list(range(length))

            inp_item_id = torch.tensor(mapped_history, dtype=torch.long).to(config.DEVICE)
            inp_positions = torch.tensor(positions, dtype=torch.long).to(config.DEVICE)
            inp_lengths = torch.tensor([length], dtype=torch.long).to(config.DEVICE)

            inputs = {
                "history": {"item_id": inp_item_id, "positions": inp_positions, "lengths": inp_lengths},
                "labels": {"item_id": torch.zeros(length, dtype=torch.long).to(config.DEVICE)},
            }

            with torch.no_grad():
                outputs = current_model(inputs)
                all_scores = outputs["all_scores"]

            WORKER_INFERENCE_LATENCY.observe(time.time() - start_time)

            k_val = min(k, current_num_items)
            top_k_scores, top_k_indices = torch.topk(all_scores, k=k_val, dim=-1)

            if current_reverse_vocab:
                predicted_indices = top_k_indices[0].tolist()
                final_item_ids = [current_reverse_vocab.get(idx, idx) for idx in predicted_indices]
            else:
                final_item_ids = top_k_indices[0].tolist()

            for score in top_k_scores[0]:
                WORKER_PREDICTION_SCORE.observe(score.item())

            result = {"item_ids": final_item_ids, "scores": top_k_scores[0].tolist()}

            producer.send("inference.results", {"request_id": request_id, "result": result})

            print(f"Finished request {request_id}")

        except Exception as e:
            WORKER_ERRORS_TOTAL.inc()
            print(f"Error processing request {request_id}: {e}")


if __name__ == "__main__":
    main()
