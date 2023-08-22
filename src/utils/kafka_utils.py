import pickle
import queue
import uuid

import kafka
import loguru
import numpy as np
import ujson
from numpy import typing as npt

TOPIC = "wine-quality"


class Producer:
    def __init__(
        self,
        kafka_server: str,
    ):
        self.producer = kafka.KafkaProducer(
            bootstrap_servers=f"{kafka_server}",
            value_serializer=lambda x: ujson.dumps(x).encode("utf-8"),
        )
        self.id = str(uuid.uuid4())

    def run(self, data: npt.NDArray[npt.NDArray[float]]):
        for i, row in enumerate(data):
            message_id = f"{self.id}:{i}"
            message = np.array2string(row)
            self.producer.send(TOPIC, {"id": message_id, "message": message})
            loguru.logger.info("PRODUCER sent message with id {}", message_id)


class Consumer:
    def __init__(
        self,
        kafka_server: str,
        model_path: str,
    ):
        self.consumer = kafka.KafkaConsumer(
            TOPIC,
            bootstrap_servers=f"{kafka_server}",
            enable_auto_commit=True,
            auto_offset_reset="earliest",
            value_deserializer=lambda x: ujson.loads(x.decode("utf-8")),
        )
        with open(model_path, "rb") as model_file:
            self.model = pickle.load(model_file)

    def run(self, id: str, result: queue.Queue):
        for record in self.consumer:
            data = record.value

            process_id, ind = data["id"].split(":")
            if process_id != id:
                continue

            loguru.logger.info("CONSUMER recieve message with id {}", data["id"])
            features = np.fromstring(data["message"][1:-1], sep=" ", dtype=np.float32)
            preds = self.model.predict([features])
            result.put({"ind": int(ind), "result": preds})
            if result.full():
                break
