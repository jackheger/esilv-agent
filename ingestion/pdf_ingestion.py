from __future__ import annotations

from collections import defaultdict
import re
from itertools import islice
from pathlib import Path
from typing import Protocol

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.chunking import HybridChunker
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import LayoutOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.model_downloader import download_models
from docling_core.transforms.chunker import BaseChunk
from google import genai
from google.genai import types

from app.models import IndexedDocumentRecord, ParsedPdfChunk, UploadedDocumentRecord, VectorChunkRecord
from app.models import utc_now_iso
from ingestion.uploads import UploadRegistry
from ingestion.vector_store import LocalVectorStore


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def parsed_chunks_from_docling_chunks(chunks: list[BaseChunk]) -> list[ParsedPdfChunk]:
    parsed_chunks: list[ParsedPdfChunk] = []
    chunk_index_by_page: dict[int, int] = defaultdict(int)
    for chunk in chunks:
        text = normalize_text(chunk.text)
        if not text:
            continue

        # The current vector-store schema stores a single page number per chunk,
        # so multi-page Docling chunks are anchored to their earliest source page.
        page_numbers = sorted(
            {
                provenance.page_no
                for item in getattr(chunk.meta, "doc_items", [])
                for provenance in getattr(item, "prov", [])
            }
        )
        page_number = page_numbers[0] if page_numbers else 1
        chunk_index = chunk_index_by_page[page_number]
        parsed_chunks.append(
            ParsedPdfChunk(
                page_number=page_number,
                chunk_index=chunk_index,
                text=text,
            )
        )
        chunk_index_by_page[page_number] += 1
    return parsed_chunks


class EmbeddingClientProtocol(Protocol):
    @property
    def configured(self) -> bool: ...

    def embed_texts(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]: ...


class GoogleEmbeddingClient:
    def __init__(self, api_key: str | None, model: str, batch_size: int = 20) -> None:
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self._client = genai.Client(api_key=api_key) if api_key else None

    @property
    def configured(self) -> bool:
        return self._client is not None

    def embed_texts(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        if self._client is None:
            raise RuntimeError("Gemini API key is not configured")

        embeddings: list[list[float]] = []
        for batch in batched(texts, self.batch_size):
            response = self._client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    taskType=task_type,
                    outputDimensionality=768,
                ),
            )
            embeddings.extend([list(item.values) for item in response.embeddings])
        return embeddings


def batched(items: list[str], size: int) -> list[list[str]]:
    batches: list[list[str]] = []
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        batches.append(batch)
    return batches


class PdfIngestionService:
    def __init__(
        self,
        upload_registry: UploadRegistry,
        vector_store: LocalVectorStore,
        embedding_client: EmbeddingClientProtocol,
        docling_artifacts_dir: Path,
    ) -> None:
        self.upload_registry = upload_registry
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.docling_artifacts_dir = docling_artifacts_dir
        self._docling_converter: DocumentConverter | None = None
        self._docling_chunker: HybridChunker | None = None

    def ingest(self, uploaded_file: object) -> UploadedDocumentRecord:
        record = self.upload_registry.save(uploaded_file)
        return self.ingest_document(record.id)

    def ingest_document(self, document_id: str) -> UploadedDocumentRecord:
        record = self.upload_registry.get(document_id)
        if record is None:
            raise FileNotFoundError(f"Document {document_id} does not exist")

        if not self.embedding_client.configured:
            record.status = "failed"
            record.error_message = "Gemini API key is not configured for embeddings"
            self.upload_registry.upsert(record)
            raise RuntimeError(record.error_message)

        try:
            page_count, chunks = self.extract_pdf_chunks(Path(record.stored_path))
            if not chunks:
                raise ValueError("No extractable text was found in this PDF")

            embeddings = self.embedding_client.embed_texts([chunk.text for chunk in chunks])
            vector_chunks = [
                VectorChunkRecord(
                    id=f"{record.id}:{chunk.page_number}:{chunk.chunk_index}",
                    document_id=record.id,
                    filename=record.filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    embedding=embedding,
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]

            indexed_at = utc_now_iso()
            self.vector_store.upsert_document(
                IndexedDocumentRecord(
                    document_id=record.id,
                    filename=record.filename,
                    indexed_at=indexed_at,
                    page_count=page_count,
                    chunk_count=len(vector_chunks),
                    chunks=vector_chunks,
                )
            )
            record.status = "indexed"
            record.page_count = page_count
            record.chunk_count = len(vector_chunks)
            record.indexed_at = indexed_at
            record.error_message = None
            self.upload_registry.upsert(record)
            return record
        except Exception as exc:
            record.status = "failed"
            record.error_message = str(exc)
            self.upload_registry.upsert(record)
            raise

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
        self.upload_registry.delete(document_id)

    def extract_pdf_chunks(self, file_path: Path) -> tuple[int, list[ParsedPdfChunk]]:
        conversion = self._docling_converter_for_ingestion().convert(file_path)
        if conversion.status != ConversionStatus.SUCCESS:
            raise ValueError(f"Docling did not fully parse this PDF: {conversion.status}")

        page_count = len(conversion.document.pages)
        chunks = list(self._docling_chunker_for_ingestion().chunk(conversion.document))
        return page_count, parsed_chunks_from_docling_chunks(chunks)

    def _docling_converter_for_ingestion(self) -> DocumentConverter:
        if self._docling_converter is not None:
            return self._docling_converter

        pipeline_options = PdfPipelineOptions()
        pipeline_options.artifacts_path = self._ensure_docling_models()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        pipeline_options.do_code_enrichment = False
        pipeline_options.do_formula_enrichment = False
        pipeline_options.do_picture_classification = False
        pipeline_options.do_picture_description = False
        pipeline_options.do_chart_extraction = False
        pipeline_options.layout_batch_size = 1
        pipeline_options.generate_parsed_pages = False

        # PyPdfium backend proved stable on the provided syllabus PDF, while the
        # docling-parse backend hit std::bad_alloc and returned partial output.
        self._docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )
        return self._docling_converter

    def _docling_chunker_for_ingestion(self) -> HybridChunker:
        if self._docling_chunker is None:
            self._docling_chunker = HybridChunker()
        return self._docling_chunker

    def _ensure_docling_models(self) -> Path:
        expected_layout_dir = self.docling_artifacts_dir / LayoutOptions().model_spec.model_repo_folder
        if not expected_layout_dir.exists():
            download_models(
                output_dir=self.docling_artifacts_dir,
                with_layout=True,
                with_tableformer=False,
                with_tableformer_v2=False,
                with_code_formula=False,
                with_picture_classifier=False,
                with_smolvlm=False,
                with_granitedocling=False,
                with_granitedocling_mlx=False,
                with_smoldocling=False,
                with_smoldocling_mlx=False,
                with_granite_vision=False,
                with_granite_chart_extraction=False,
                with_rapidocr=False,
                with_easyocr=False,
                progress=False,
            )
        return self.docling_artifacts_dir
