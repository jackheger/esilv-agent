from agents.retrieval import RetrievalAgent
from app.models import IndexedDocumentRecord, VectorChunkRecord
from ingestion.vector_store import LocalVectorStore


class FakeEmbeddingClient:
    def __init__(self, embedding_by_query: dict[str, list[float]]) -> None:
        self.embedding_by_query = embedding_by_query
        self.requests: list[tuple[list[str], str]] = []

    @property
    def configured(self) -> bool:
        return True

    def embed_texts(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        self.requests.append((list(texts), task_type))
        return [self.embedding_by_query[text] for text in texts]


def upsert_document(
    vector_store: LocalVectorStore,
    document_id: str,
    filename: str,
    text: str,
    embedding: list[float],
    page_number: int = 1,
) -> None:
    vector_store.upsert_document(
        IndexedDocumentRecord(
            document_id=document_id,
            filename=filename,
            indexed_at="2026-03-18T12:00:00+00:00",
            page_count=1,
            chunk_count=1,
            chunks=[
                VectorChunkRecord(
                    id=f"{document_id}:{page_number}:0",
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    chunk_index=0,
                    text=text,
                    embedding=embedding,
                )
            ],
        )
    )


def test_retrieval_ranks_admissions_document_above_unrelated_report(tmp_path):
    vector_store = LocalVectorStore(tmp_path / "vector_store")
    upsert_document(
        vector_store,
        document_id="doc-admissions",
        filename="admissions.pdf",
        text="Admissions office hours are Monday to Friday from 9am to 5pm.",
        embedding=[1.0, 0.0],
    )
    upsert_document(
        vector_store,
        document_id="doc-report",
        filename="report.pdf",
        text="Quarterly infrastructure report about budget and procurement.",
        embedding=[0.1, 0.99],
    )

    seed_query = "What are the admissions office hours?"
    probe_agent = RetrievalAgent(vector_store=vector_store, embedding_client=FakeEmbeddingClient({}))
    expanded_query = probe_agent.expand_query(seed_query)
    embeddings = FakeEmbeddingClient({expanded_query: [1.0, 0.0]})
    agent = RetrievalAgent(vector_store=vector_store, embedding_client=embeddings)

    hits = agent.search(seed_query, top_k=2)

    assert embeddings.requests == [([expanded_query], "RETRIEVAL_QUERY")]
    assert len(hits) == 2
    assert hits[0].filename == "admissions.pdf"
    assert hits[0].page_number == 1
    assert hits[0].score > hits[1].score


def test_retrieval_lexical_bonus_breaks_close_semantic_ties(tmp_path):
    vector_store = LocalVectorStore(tmp_path / "vector_store")
    upsert_document(
        vector_store,
        document_id="doc-admissions",
        filename="admissions.pdf",
        text="Admissions office hours are Monday to Friday from 9am to 5pm.",
        embedding=[0.98, 0.2],
    )
    upsert_document(
        vector_store,
        document_id="doc-general",
        filename="general.pdf",
        text="The schedule includes opening times and reception availability.",
        embedding=[1.0, 0.0],
    )

    seed_query = "admissions office hours"
    probe_agent = RetrievalAgent(vector_store=vector_store, embedding_client=FakeEmbeddingClient({}))
    expanded_query = probe_agent.expand_query(seed_query)
    embeddings = FakeEmbeddingClient({expanded_query: [1.0, 0.0]})
    agent = RetrievalAgent(vector_store=vector_store, embedding_client=embeddings)

    hits = agent.search(seed_query, top_k=2)

    assert len(hits) == 2
    assert hits[0].filename == "admissions.pdf"
    assert hits[0].lexical_overlap > hits[1].lexical_overlap
    assert hits[0].score > hits[1].score


def test_retrieval_is_weak_when_vector_store_is_empty(tmp_path):
    vector_store = LocalVectorStore(tmp_path / "vector_store")
    embeddings = FakeEmbeddingClient({"unused query": [1.0, 0.0]})
    agent = RetrievalAgent(vector_store=vector_store, embedding_client=embeddings)

    hits = agent.search("unused query", top_k=3)

    assert hits == []
    assert agent.is_weak(hits) is True


def test_retrieval_expands_sql_query_before_embedding(tmp_path):
    vector_store = LocalVectorStore(tmp_path / "vector_store")
    upsert_document(
        vector_store,
        document_id="doc-database",
        filename="database.pdf",
        text="Database systems and relational modeling are covered in this module.",
        embedding=[1.0, 0.0],
    )

    seed_query = "sql courses"
    probe_agent = RetrievalAgent(vector_store=vector_store, embedding_client=FakeEmbeddingClient({}))
    expanded_query = probe_agent.expand_query(seed_query)
    embeddings = FakeEmbeddingClient({expanded_query: [1.0, 0.0]})
    agent = RetrievalAgent(vector_store=vector_store, embedding_client=embeddings)

    agent.search(seed_query, top_k=1)

    assert "database" in expanded_query.lower()
    assert embeddings.requests == [([expanded_query], "RETRIEVAL_QUERY")]
