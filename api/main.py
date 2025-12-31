import asyncio
import json
import os
import uuid
from typing import List, Optional

import asyncpg
from fastapi import FastAPI, HTTPException
from kafka import KafkaProducer
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
POSTGRES_URL = os.environ.get("POSTGRES_URL", "postgresql://sasrec:password@postgres:5432/sasrec")

app = FastAPI(title="SasRec Async API")
Instrumentator().instrument(app).expose(app)

producer = None


@app.on_event("startup")
async def startup_event():
    global producer
    for i in range(5):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )
            print("Kafka Producer connected.")
            break
        except Exception as e:
            print(f"Kafka connection failed (attempt {i}): {e}")
            await asyncio.sleep(5)


@app.on_event("shutdown")
def shutdown_event():
    if producer:
        producer.close()


class PredictionRequest(BaseModel):
    history: List[int]
    k: int = 10
    user_id: Optional[str] = "anon"


class SubmitResponse(BaseModel):
    request_id: str
    status: str


class ResultResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[dict] = None


@app.post("/predict", response_model=SubmitResponse, status_code=202)
async def submit_prediction(request: PredictionRequest):
    if not producer:
        raise HTTPException(status_code=503, detail="Kafka unavailable")

    request_id = str(uuid.uuid4())

    try:
        conn = await asyncpg.connect(POSTGRES_URL)
        await conn.execute(
            "INSERT INTO predictions (request_id, user_id, status) VALUES ($1, $2, $3)",
            request_id,
            request.user_id,
            "PENDING",
        )
        await conn.close()
    except Exception as e:
        print(f"DB Error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    message = {"request_id": request_id, "history": request.history, "k": request.k}
    producer.send("inference.requests", message)

    return SubmitResponse(request_id=request_id, status="PENDING")


@app.get("/result/{request_id}", response_model=ResultResponse)
async def get_result(request_id: str):
    try:
        conn = await asyncpg.connect(POSTGRES_URL)
        row = await conn.fetchrow("SELECT status, result_json FROM predictions WHERE request_id = $1", request_id)
        await conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Request ID not found")

        status = row["status"]
        result = json.loads(row["result_json"]) if row["result_json"] else None

        return ResultResponse(request_id=request_id, status=status, result=result)

    except Exception as e:
        print(f"DB Error: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


class UserHistoryResponse(BaseModel):
    user_id: str
    items: List[int]


@app.get("/users", response_model=List[str])
async def get_users():
    try:
        conn = await asyncpg.connect(POSTGRES_URL)
        rows = await conn.fetch("SELECT DISTINCT user_id FROM interactions ORDER BY user_id")
        await conn.close()
        return [r["user_id"] for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/history", response_model=UserHistoryResponse)
async def get_user_history(user_id: str):
    try:
        conn = await asyncpg.connect(POSTGRES_URL)
        rows = await conn.fetch("SELECT item_id FROM interactions WHERE user_id = $1 ORDER BY created_at ASC", user_id)
        await conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="User not found")

        items = [r["item_id"] for r in rows]
        return UserHistoryResponse(user_id=user_id, items=items)

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
