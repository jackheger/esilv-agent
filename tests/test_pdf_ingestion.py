from pathlib import Path
from types import SimpleNamespace

from ingestion.pdf_ingestion import (
    PdfIngestionService,
    parsed_chunks_from_docling_chunks,
)
from ingestion.uploads import UploadRegistry
from ingestion.vector_store import LocalVectorStore


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes, mime_type: str = "application/pdf") -> None:
        self.name = name
        self._content = content
        self.type = mime_type

    def getvalue(self) -> bytes:
        return self._content


class FakeEmbeddingClient:
    def __init__(self, dimensions: int = 4) -> None:
        self.dimensions = dimensions
        self.requests: list[list[str]] = []

    @property
    def configured(self) -> bool:
        return True

    def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        self.requests.append(list(texts))
        embeddings: list[list[float]] = []
        for index, text in enumerate(texts, start=1):
            value = float(len(text) + index)
            embeddings.append([value] * self.dimensions)
        return embeddings


class FakePdfIngestionService(PdfIngestionService):
    def __init__(self, *args, pages: list[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pages = pages

    def extract_pdf_chunks(self, file_path):
        return len(self._pages), [
            type("Chunk", (), {"page_number": index, "chunk_index": 0, "text": page_text})()
            for index, page_text in enumerate(self._pages, start=1)
        ]

def test_parsed_chunks_from_docling_chunks_use_docling_provenance_pages():
    fake_chunk = SimpleNamespace(
        text="Chunked text from Docling",
        meta=SimpleNamespace(
            doc_items=[
                SimpleNamespace(
                    prov=[
                        SimpleNamespace(page_no=4),
                        SimpleNamespace(page_no=4),
                    ]
                )
            ]
        ),
    )

    parsed_chunks = parsed_chunks_from_docling_chunks([fake_chunk])

    assert len(parsed_chunks) == 1
    assert parsed_chunks[0].page_number == 4
    assert parsed_chunks[0].chunk_index == 0
    assert parsed_chunks[0].text == "Chunked text from Docling"


def test_pdf_ingestion_indexes_pdf_and_updates_registry(tmp_path):
    registry = UploadRegistry(tmp_path / "uploads" / "registry.json", tmp_path / "uploads" / "files")
    vector_store = LocalVectorStore(tmp_path / "vector_store")
    embedding_client = FakeEmbeddingClient()
    ingestion = FakePdfIngestionService(
        upload_registry=registry,
        vector_store=vector_store,
        embedding_client=embedding_client,
        docling_artifacts_dir=tmp_path / "docling_artifacts",
        pages=[
            "Admissions requirements and timelines for international students.",
            "Engineering programs include core scientific modules and electives.",
        ],
    )

    record = ingestion.ingest(FakeUploadedFile("guide.pdf", b"%PDF-fake"))
    indexed_document = vector_store.get_document(record.id)

    assert record.status == "indexed"
    assert record.page_count == 2
    assert record.chunk_count >= 2
    assert record.error_message is None
    assert indexed_document is not None
    assert indexed_document.filename == "guide.pdf"
    assert indexed_document.chunk_count == record.chunk_count
    assert vector_store.stats().document_count == 1
    assert len(embedding_client.requests) == 1


def test_pdf_ingestion_delete_removes_registry_and_vectors(tmp_path):
    registry = UploadRegistry(tmp_path / "uploads" / "registry.json", tmp_path / "uploads" / "files")
    vector_store = LocalVectorStore(tmp_path / "vector_store")
    embedding_client = FakeEmbeddingClient()
    ingestion = FakePdfIngestionService(
        upload_registry=registry,
        vector_store=vector_store,
        embedding_client=embedding_client,
        docling_artifacts_dir=tmp_path / "docling_artifacts",
        pages=["Admissions office hours and contact information."],
    )

    record = ingestion.ingest(FakeUploadedFile("hours.pdf", b"%PDF-fake"))
    ingestion.delete_document(record.id)

    assert registry.get(record.id) is None
    assert vector_store.get_document(record.id) is None


def test_pdf_ingestion_with_docling_indexes_real_pdf(tmp_path):
    source_pdf = Path("data/pdfs/esilv_syllabus_ccc.pdf")
    registry = UploadRegistry(tmp_path / "uploads" / "registry.json", tmp_path / "uploads" / "files")
    vector_store = LocalVectorStore(tmp_path / "vector_store")
    embedding_client = FakeEmbeddingClient()
    ingestion = PdfIngestionService(
        upload_registry=registry,
        vector_store=vector_store,
        embedding_client=embedding_client,
        docling_artifacts_dir=tmp_path / "docling_artifacts",
    )

    record = registry.save_bytes(source_pdf.name, source_pdf.read_bytes(), "application/pdf")
    indexed = ingestion.ingest_document(record.id)
    stored = vector_store.get_document(record.id)

    assert indexed.status == "indexed"
    assert indexed.page_count == 25
    assert indexed.chunk_count > 50
    assert stored is not None
    assert stored.chunk_count == indexed.chunk_count
    assert any(chunk.page_number == 25 for chunk in stored.chunks)
    assert any("cloud" in chunk.text.lower() or "cybers" in chunk.text.lower() for chunk in stored.chunks)
