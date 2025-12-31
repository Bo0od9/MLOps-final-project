import json
import os
import time
import uuid

import pandas as pd
import requests
import streamlit as st
from kafka import KafkaConsumer

API_URL = os.environ.get("API_URL", "http://api:8000")
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")

st.set_page_config(page_title="SasRec Async Recommender", layout="wide")

st.title("SasRec Recommender")
st.markdown("Uses **Kafka** + **PostgreSQL** for asynchronous inference.")

mode = st.sidebar.radio("Input Mode", ["Manual Input", "Select User"])
k_items = st.sidebar.slider("Number of Recommendations", 1, 50, 10)

history = []
user_id = "anon"

if mode == "Manual Input":
    user_input = st.text_input("Enter Item IDs (comma separated)", "10, 20, 30")
    if user_input:
        try:
            history = [int(x.strip()) for x in user_input.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid input format")

else:
    try:
        users_resp = requests.get(f"{API_URL}/users")
        if users_resp.status_code == 200:
            users = users_resp.json()
            if not users:
                st.warning("No users found in database.")
            else:
                selected_user = st.selectbox("Select User", users)
                user_id = selected_user

                hist_resp = requests.get(f"{API_URL}/users/{selected_user}/history")
                if hist_resp.status_code == 200:
                    full_history = hist_resp.json()["items"]
                    st.write(f"**Full History ({len(full_history)} items):**", full_history)

                    slice_len = st.slider("Use last N items", 1, len(full_history), min(len(full_history), 20))
                    history = full_history[-slice_len:]
                    st.write(f"**Using for inference:** {history}")
        else:
            st.error("Failed to fetch users list")
    except Exception as e:
        st.error(f"Error fetching users: {e}")

if st.button("Get Recommendations", type="primary"):
    if not history:
        st.warning("History is empty.")
    else:
        try:
            with st.spinner("Submitting request..."):
                payload = {"history": history, "k": k_items, "user_id": user_id}
                response = requests.post(f"{API_URL}/predict", json=payload)

            if response.status_code == 202:
                data = response.json()
                request_id = data["request_id"]
                st.info(f"Request Submitted! ID: `{request_id}`. Waiting for Kafka Event...")

                group_id = f"ui_consumer_{uuid.uuid4()}"

                consumer = KafkaConsumer(
                    "inference.results",
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                    group_id=group_id,
                    auto_offset_reset="earliest",
                )

                status_placeholder = st.empty()
                start_time = time.time()
                found = False
                result = None

                while time.time() - start_time < 30:  # 30s timeout
                    msg_batch = consumer.poll(timeout_ms=1000)

                    for tp, messages in msg_batch.items():
                        for message in messages:
                            if message.value.get("request_id") == request_id:
                                result = message.value.get("result")
                                if result:
                                    found = True
                                    consumer.close()
                                    break
                        if found:
                            break

                    if found:
                        break
                    status_placeholder.text(f"Waiting for result logic... ({int(time.time() - start_time)}s)")

                if found and result:
                    status_placeholder.success("Received Event from Kafka!")

                    st.subheader("Recommended Items")
                    if "item_ids" in result and "scores" in result:
                        df = pd.DataFrame({"Item ID": result["item_ids"], "Score": result["scores"]})

                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.write("Top Items Table")
                            st.dataframe(df)
                        with col2:
                            st.write("Score Distribution")
                            st.bar_chart(df.set_index("Item ID"))
                    else:
                        st.json(result)

                else:
                    consumer.close()
                    status_placeholder.error("Timeout: Result event not received.")

            else:
                st.error(f"Error submitting: {response.text}")

        except ValueError:
            st.error("Invalid input")
        except Exception as e:
            st.error(f"Error: {e}")
