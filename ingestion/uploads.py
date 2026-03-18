from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from app.models import UploadedDocumentRecord

ALLOWED_EXTENSIONS = {".pdf"}


class UploadRegistry:
    def __init__(self, registry_path: Path, files_dir: Path) -> None:
        self.registry_path = registry_path
        self.files_dir = files_dir
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[UploadedDocumentRecord]:
        records = self._read_registry()
        records.sort(key=lambda record: record.uploaded_at, reverse=True)
        return records

    def get(self, document_id: str) -> UploadedDocumentRecord | None:
        for record in self._read_registry():
            if record.id == document_id:
                return record
        return None

    def save(self, uploaded_file: object) -> UploadedDocumentRecord:
        file_name = getattr(uploaded_file, "name", "")
        mime_type = getattr(uploaded_file, "type", "application/octet-stream") or "application/octet-stream"
        file_bytes = self._read_uploaded_bytes(uploaded_file)
        return self.save_bytes(file_name=file_name, file_bytes=file_bytes, mime_type=mime_type)

    def save_bytes(self, file_name: str, file_bytes: bytes, mime_type: str) -> UploadedDocumentRecord:
        extension = Path(file_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension or 'missing extension'}")

        document_id = uuid4().hex
        stored_name = f"{document_id}{extension}"
        stored_path = self.files_dir / stored_name
        stored_path.write_bytes(file_bytes)

        record = UploadedDocumentRecord(
            id=document_id,
            filename=file_name,
            stored_path=stored_path.as_posix(),
            mime_type=mime_type,
            size=len(file_bytes),
        )
        records = self._read_registry()
        records.append(record)
        self._write_registry(records)
        return record

    def upsert(self, record: UploadedDocumentRecord) -> UploadedDocumentRecord:
        records = self._read_registry()
        replaced = False
        updated_records: list[UploadedDocumentRecord] = []
        for existing in records:
            if existing.id == record.id:
                updated_records.append(record)
                replaced = True
            else:
                updated_records.append(existing)

        if not replaced:
            updated_records.append(record)

        self._write_registry(updated_records)
        return record

    def delete(self, document_id: str) -> None:
        records = self._read_registry()
        removed_record: UploadedDocumentRecord | None = None
        kept_records: list[UploadedDocumentRecord] = []

        for record in records:
            if record.id == document_id and removed_record is None:
                removed_record = record
                continue
            kept_records.append(record)

        if removed_record is None:
            return

        stored_path = Path(removed_record.stored_path)
        if stored_path.exists():
            stored_path.unlink()
        self._write_registry(kept_records)

    def _read_registry(self) -> list[UploadedDocumentRecord]:
        if not self.registry_path.exists():
            return []
        try:
            payload = json.loads(self.registry_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [UploadedDocumentRecord.model_validate(item) for item in payload]

    def _write_registry(self, records: list[UploadedDocumentRecord]) -> None:
        lock_path = self.registry_path.with_suffix(".lock")
        with FileLock(str(lock_path)):
            payload = [record.model_dump(mode="json") for record in records]
            self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _read_uploaded_bytes(uploaded_file: object) -> bytes:
        if hasattr(uploaded_file, "getvalue"):
            return bytes(uploaded_file.getvalue())
        if hasattr(uploaded_file, "getbuffer"):
            return bytes(uploaded_file.getbuffer())
        if hasattr(uploaded_file, "read"):
            content = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            if isinstance(content, bytes):
                return content
            raise TypeError("Uploaded file reader returned non-bytes content")
        raise TypeError("Unsupported uploaded file object")
