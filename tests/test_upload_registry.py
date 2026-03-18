import pytest

from ingestion.uploads import UploadRegistry


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes, mime_type: str) -> None:
        self.name = name
        self._content = content
        self.type = mime_type

    def getvalue(self) -> bytes:
        return self._content


def test_upload_registry_saves_lists_and_deletes_documents(tmp_path):
    registry = UploadRegistry(tmp_path / "uploads" / "registry.json", tmp_path / "uploads" / "files")
    uploaded = FakeUploadedFile("brochure.pdf", b"pdf-bytes", "application/pdf")

    record = registry.save(uploaded)
    listed = registry.list()

    assert listed[0].id == record.id
    assert listed[0].filename == "brochure.pdf"

    registry.delete(record.id)
    assert registry.list() == []


def test_upload_registry_rejects_unsupported_file_types(tmp_path):
    registry = UploadRegistry(tmp_path / "uploads" / "registry.json", tmp_path / "uploads" / "files")

    with pytest.raises(ValueError):
        registry.save_bytes("notes.txt", b"binary", "text/plain")
