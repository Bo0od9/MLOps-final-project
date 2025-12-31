import os
import random
import time
from datetime import datetime, timedelta

import psycopg2

POSTGRES_URL = os.environ.get("POSTGRES_URL", "postgresql://sasrec:password@localhost:5432/sasrec")

NUM_USERS = 100
ITEMS_POOL = range(1, 1000)


def get_db_connection():
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        return conn
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None


def seed_data():
    conn = get_db_connection()
    if not conn:
        return

    cur = conn.cursor()

    print("Seeding data...")
    try:
        cur.execute("TRUNCATE TABLE interactions")

        users = [f"user_{i}" for i in range(1, NUM_USERS + 1)]

        for user in users:
            history_len = random.randint(5, 50)
            items = random.sample(ITEMS_POOL, history_len)

            base_time = datetime.now() - timedelta(days=30)

            for i, item in enumerate(items):
                timestamp = base_time + timedelta(hours=i)
                cur.execute(
                    "INSERT INTO interactions (user_id, item_id, created_at) VALUES (%s, %s, %s)",
                    (user, item, timestamp),
                )

        conn.commit()
        print(f"Seeded {NUM_USERS} users with random history.")

    except Exception as e:
        print(f"Error seeding: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    time.sleep(2)
    seed_data()
