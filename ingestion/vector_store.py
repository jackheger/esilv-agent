from __future__ import annotations

from pathlib import Path

from filelock import FileLock

from app.models import IndexedDocumentRecord, VectorStoreStats


class LocalVectorStore:
    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self.documents_dir = self.store_dir / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def upsert_document(self, document: IndexedDocumentRecord) -> IndexedDocumentRecord:
        path = self._document_path(document.document_id)
        lock_path = self._lock_path(document.document_id)
        with FileLock(str(lock_path)):
            path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        return document

    def get_document(self, document_id: str) -> IndexedDocumentRecord | None:
        path = self._document_path(document_id)
        if not path.exists():
            return None
        return IndexedDocumentRecord.model_validate_json(path.read_text("utf-8"))

    def list_documents(self) -> list[IndexedDocumentRecord]:
        documents: list[IndexedDocumentRecord] = []
        for path in self.documents_dir.glob("*.json"):
            try:
                documents.append(IndexedDocumentRecord.model_validate_json(path.read_text("utf-8")))
            except (OSError, ValueError):
                continue
        documents.sort(key=lambda record: record.indexed_at, reverse=True)
        return documents

    def delete_document(self, document_id: str) -> None:
        path = self._document_path(document_id)
        lock_path = self._lock_path(document_id)
        with FileLock(str(lock_path)):
            path.unlink(missing_ok=True)

    def stats(self) -> VectorStoreStats:
        documents = self.list_documents()
        return VectorStoreStats(
            document_count=len(documents),
            chunk_count=sum(document.chunk_count for document in documents),
        )

    def _document_path(self, document_id: str) -> Path:
        return self.documents_dir / f"{document_id}.json"

    def _lock_path(self, document_id: str) -> Path:
        return self.documents_dir / f"{document_id}.lock"
