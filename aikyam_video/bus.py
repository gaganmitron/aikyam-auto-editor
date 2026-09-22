"""Event transport: Kafka (production) and in-memory (tests / single-process). At-least-once delivery, per-group offsets;
combine with events.consume() for idempotent handling."""
from __future__ import annotations
import json, os, time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from .events import Event, TOPIC


class Bus(ABC):
    @abstractmethod
    def publish(self, event: Event) -> None: ...
    @abstractmethod
    def poll(self, group: str, timeout: float = 1.0) -> Optional[Event]: ...
    @abstractmethod
    def commit(self, group: str, event: Event) -> None: ...
    def close(self) -> None: ...


class MemoryBus(Bus):
    def __init__(self):
        self.log: List[Event] = []
        self.off: Dict[str, int] = {}

    def publish(self, event):
        self.log.append(event)

    def poll(self, group, timeout=0.0):
        i = self.off.get(group, 0)
        return self.log[i] if i < len(self.log) else None

    def commit(self, group, event):
        self.off[group] = self.off.get(group, 0) + 1


class KafkaBus(Bus):
    """confluent-kafka. Key = jobId (per-job ordering). acks=all + idempotent producer; manual commit after handling."""

    def __init__(self, bootstrap: Optional[str] = None, topic: str = TOPIC, partitions: int = 3, client_id: str = "aikyam-video"):
        from confluent_kafka import Consumer, Producer
        from confluent_kafka.admin import AdminClient, NewTopic
        self._C = Consumer
        self.conf = {"bootstrap.servers": bootstrap or os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")}
        for k, ek in (("security.protocol", "KAFKA_SECURITY_PROTOCOL"), ("sasl.mechanisms", "KAFKA_SASL_MECHANISM"),
                      ("sasl.username", "KAFKA_SASL_USERNAME"), ("sasl.password", "KAFKA_SASL_PASSWORD")):
            if os.environ.get(ek): self.conf[k] = os.environ[ek]
        self.topic = topic
        self.producer = Producer({**self.conf, "acks": "all", "enable.idempotence": True, "client.id": client_id})
        admin = AdminClient(self.conf)      # keep a reference: a temporary is GC'd before the futures resolve (_DESTROY)
        for f in admin.create_topics([NewTopic(topic, partitions, 1)]).values():
            try: f.result(10)
            except Exception as e:
                if "EXISTS" not in str(e).upper(): raise
        self.consumers: Dict[str, object] = {}
        self.msgs: Dict[str, object] = {}

    def publish(self, event):
        self.producer.produce(self.topic, key=event.jobId, value=event.model_dump_json().encode())
        self.producer.flush(10)

    def _consumer(self, group):
        if group not in self.consumers:
            c = self._C({**self.conf, "group.id": group, "enable.auto.commit": False, "auto.offset.reset": "earliest",
                         "max.poll.interval.ms": 3600000})   # one job at a time; a render may take long
            c.subscribe([self.topic]); self.consumers[group] = c
        return self.consumers[group]

    def poll(self, group, timeout=1.0):
        m = self._consumer(group).poll(timeout)
        if m is None or m.error():
            return None
        ev = Event(**json.loads(m.value())); self.msgs[f"{group}:{ev.eventId}"] = m
        return ev

    def commit(self, group, event):
        m = self.msgs.pop(f"{group}:{event.eventId}", None)
        if m is not None:
            self._consumer(group).commit(message=m, asynchronous=False)

    def close(self):
        for c in self.consumers.values(): c.close()


def from_env() -> Optional[Bus]:
    return KafkaBus() if os.environ.get("KAFKA_BOOTSTRAP") else None
