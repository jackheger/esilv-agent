from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from app.models import ConversationRecord, MessageRecord


class ConversationStore:
    def __init__(self, conversations_dir: Path) -> None:
        self.conversations_dir = conversations_dir
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[ConversationRecord]:
        records_with_ordering: list[tuple[ConversationRecord, int]] = []
        for path in self.conversations_dir.glob("*.json"):
            try:
                record = ConversationRecord.model_validate_json(path.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            try:
                modified_at_ns = path.stat().st_mtime_ns
            except OSError:
                modified_at_ns = 0
            records_with_ordering.append((record, modified_at_ns))
        records_with_ordering.sort(
            key=lambda item: (item[0].updated_at, item[1]),
            reverse=True,
        )
        return [record for record, _ in records_with_ordering]

    def exists(self, conversation_id: str) -> bool:
        return self._conversation_path(conversation_id).exists()

    def create(self) -> ConversationRecord:
        record = ConversationRecord(id=uuid4().hex)
        self._write_record(record)
        return record

    def load(self, conversation_id: str) -> ConversationRecord:
        path = self._conversation_path(conversation_id)
        if not path.exists():
            raise FileNotFoundError(f"Conversation {conversation_id} does not exist")
        return ConversationRecord.model_validate_json(path.read_text("utf-8"))

    def append_message(self, conversation_id: str, message: MessageRecord) -> ConversationRecord:
        record = self.load(conversation_id)
        record.messages.append(message)
        record.updated_at = message.timestamp

        first_user_message = next(
            (item for item in record.messages if item.role == "user" and item.content.strip()),
            None,
        )
        if record.title == "New conversation" and first_user_message is not None:
            record.title = self._make_title(first_user_message.content)

        self._write_record(record)
        return record

    def delete(self, conversation_id: str) -> None:
        path = self._conversation_path(conversation_id)
        lock_path = self._lock_path(conversation_id)
        with FileLock(str(lock_path)):
            if path.exists():
                path.unlink()

    def _write_record(self, record: ConversationRecord) -> None:
        path = self._conversation_path(record.id)
        lock_path = self._lock_path(record.id)
        with FileLock(str(lock_path)):
            path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    def _conversation_path(self, conversation_id: str) -> Path:
        return self.conversations_dir / f"{conversation_id}.json"

    def _lock_path(self, conversation_id: str) -> Path:
        return self.conversations_dir / f"{conversation_id}.lock"

    @staticmethod
    def _make_title(content: str, limit: int = 60) -> str:
        normalized = " ".join(content.split())
        if len(normalized) <= limit:
            return normalized or "New conversation"
        return f"{normalized[: limit - 3].rstrip()}..."
