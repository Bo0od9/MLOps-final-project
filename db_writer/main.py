import json
import os
import time

import psycopg2
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
POSTGRES_URL = os.environ.get("POSTGRES_URL", "postgresql://sasrec:password@postgres:5432/sasrec")


def get_db_connection():
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        return conn
    except Exception as e:
        print(f"DB Connection failed: {e}")
        return None


def update_db(conn, request_id, status, result=None):
    try:
        cur = conn.cursor()
        if result:
            cur.execute(
                "UPDATE predictions SET status = %s, result_json = %s, updated_at = NOW() WHERE request_id = %s",
                (status, json.dumps(result), request_id),
            )
        else:
            cur.execute(
                "UPDATE predictions SET status = %s, updated_at = NOW() WHERE request_id = %s", (status, request_id)
            )
        conn.commit()
        cur.close()
        print(f"Updated DB for request {request_id}: {status}")
    except Exception as e:
        print(f"DB Update failed for {request_id}: {e}")
        conn.rollback()


def main():
    print("DB Writer starting...")
    time.sleep(10)

    conn = get_db_connection()

    consumer = KafkaConsumer(
        "inference.results",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="sasrec_db_writer_group",
        auto_offset_reset="earliest",
    )

    print("DB Writer listening for results...")

    for message in consumer:
        data = message.value
        request_id = data.get("request_id")
        result = data.get("result")

        if not conn or conn.closed:
            conn = get_db_connection()

        if conn:
            update_db(conn, request_id, "COMPLETED", result)
        else:
            print("Skipping DB update (no connection)")


if __name__ == "__main__":
    main()
