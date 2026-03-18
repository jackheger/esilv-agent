import os

import pytest
import requests

from agents.web_search import SiteSearchAgent


class FakeTavilyClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_site_search_uses_tavily_and_ranks_finance_major_first(tmp_path):
    agent = SiteSearchAgent(
        cache_dir=tmp_path / "site_cache",
        allowed_domains=["www.esilv.fr", "devinci.fr"],
        api_key="tvly-test-key",
        ttl_hours=24,
        session=requests.Session(),
    )
    fake_client = FakeTavilyClient(
        {
            "query": "ESILV finance major",
            "results": [
                {
                    "url": "https://www.esilv.fr/formations/cycle-ingenieur/majeures/ingenierie-financiere/",
                    "title": "Ingenierie financiere | ESILV",
                    "content": "La majeure Financial Engineering forme les etudiants a la finance quantitative.",
                    "score": 0.72,
                },
                {
                    "url": "https://www.esilv.fr/formations/cycle-ingenieur/majeures/objets-connectes-et-cybersecurite/",
                    "title": "Objets connectes et cybersecurite | ESILV",
                    "content": "Une majeure orientee cybersecurite et IoT.",
                    "score": 0.76,
                },
                {
                    "url": "https://www.devinci.fr/programmes/finance/",
                    "title": "Finance | De Vinci",
                    "content": "External result that should be filtered out.",
                    "score": 0.99,
                },
            ],
        }
    )
    agent.client = fake_client

    hits = agent.search("Proposez vous une majeur de Finance?", top_k=3)

    assert hits
    assert hits[0].url == "https://www.esilv.fr/formations/cycle-ingenieur/majeures/ingenierie-financiere/"
    assert "Financial Engineering" in hits[0].snippet
    assert all("esilv.fr" in hit.url for hit in hits)

    request_body = fake_client.calls[0]
    assert request_body["include_domains"] == ["www.esilv.fr"]
    assert request_body["search_depth"] == "advanced"
    assert "financial engineering" in request_body["query"].lower()
    assert request_body["timeout"] == 20


def test_site_search_reuses_cached_results_without_tavily_key(tmp_path):
    seeded_agent = SiteSearchAgent(
        cache_dir=tmp_path / "site_cache",
        allowed_domains=["esilv.fr", "www.esilv.fr"],
        api_key="tvly-test-key",
        ttl_hours=24,
        session=requests.Session(),
    )
    seeded_agent.client = FakeTavilyClient(
        {
            "query": "ESILV admissions",
            "results": [
                {
                    "url": "https://www.esilv.fr/admissions/",
                    "title": "Admissions | ESILV",
                    "content": "Admissions requirements and application steps for ESILV.",
                    "score": 0.88,
                }
            ],
        }
    )

    seeded_hits = seeded_agent.search("How do ESILV admissions work?", top_k=3)

    offline_agent = SiteSearchAgent(
        cache_dir=tmp_path / "site_cache",
        allowed_domains=["esilv.fr", "www.esilv.fr"],
        api_key=None,
        ttl_hours=24,
        session=requests.Session(),
    )
    cached_hits = offline_agent.search("How do ESILV admissions work?", top_k=3)

    assert seeded_hits
    assert cached_hits == seeded_hits


def test_site_search_expands_sql_query_toward_database_terms(tmp_path):
    agent = SiteSearchAgent(
        cache_dir=tmp_path / "site_cache",
        allowed_domains=["www.esilv.fr"],
        api_key="tvly-test-key",
        ttl_hours=24,
        session=requests.Session(),
    )
    fake_client = FakeTavilyClient({"query": "ESILV sql courses", "results": []})
    agent.client = fake_client

    agent.search("Does ESILV propose SQL courses?", top_k=3)

    request_body = fake_client.calls[0]
    assert "database" in request_body["query"].lower()


@pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="TAVILY_API_KEY is not configured")
def test_site_search_live_finance_major_query(tmp_path):
    agent = SiteSearchAgent(
        cache_dir=tmp_path / "site_cache",
        allowed_domains=["esilv.fr", "www.esilv.fr"],
        api_key=os.environ["TAVILY_API_KEY"],
        ttl_hours=24,
    )

    hits = agent.search("Proposez vous une majeur de Finance?", top_k=5)

    assert hits
    haystacks = [" ".join((hit.title, hit.url, hit.snippet)).lower() for hit in hits]
    assert any(
        marker in haystack
        for haystack in haystacks
        for marker in ("financial engineering", "ingenierie financiere")
    )
