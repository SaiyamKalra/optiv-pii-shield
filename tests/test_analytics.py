import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from analytics import metrics
from database.db import Base, DocumentSummary, PIIDetectionLog


@pytest.fixture
def setup_test_db(monkeypatch):
    """Creates a temporary in-memory database and patches the engine in metrics."""
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)
    
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestSession()

    # Seed mock data
    doc1 = DocumentSummary(
        file_id="doc_ci_1",
        file_name="test1.pdf",
        page_number=1,
        status="SUCCESS",
        verification_passed=True,
        total_entities_detected=2,
    )
    doc2 = DocumentSummary(
        file_id="doc_ci_2",
        file_name="test2.pdf",
        page_number=1,
        status="FLAGGED_FOR_REVIEW",
        verification_passed=False,
        total_entities_detected=1,
    )
    db.add_all([doc1, doc2])

    log1 = PIIDetectionLog(
        file_id="doc_ci_1",
        token="[PERSON_001]",
        entity_type="PERSON",
        confidence_score=0.95,
        page_number=1,
    )
    log2 = PIIDetectionLog(
        file_id="doc_ci_1",
        token="[EMAIL_001]",
        entity_type="EMAIL_ADDRESS",
        confidence_score=1.0,
        page_number=1,
    )
    log3 = PIIDetectionLog(
        file_id="doc_ci_2",
        token="[PERSON_002]",
        entity_type="PERSON",
        confidence_score=0.90,
        page_number=1,
    )
    db.add_all([log1, log2, log3])
    db.commit()

    # Monkeypatch the engine used in metrics module to use the test_engine
    monkeypatch.setattr(metrics, "engine", test_engine)

    yield

    db.close()


def test_fetch_executive_kpis(setup_test_db):
    kpis = metrics.fetch_executive_kpis()
    assert kpis["total_documents"] == 2
    assert kpis["pass_rate_pct"] == 50.0
    assert kpis["total_pii_masked"] == 3


def test_fetch_entity_distribution(setup_test_db):
    df = metrics.fetch_entity_distribution()
    assert not df.empty
    assert "entity_type" in df.columns
    assert df[df["entity_type"] == "PERSON"]["occurrences"].values[0] == 2


def test_fetch_confidence_score(setup_test_db):
    df = metrics.fetch_confidence_score()
    assert not df.empty
    assert "avg_confidence" in df.columns


def test_fetch_risk_exposure_matrix(setup_test_db):
    df = metrics.fetch_risk_exposure_matrix()
    assert not df.empty
    assert "file_id" in df.columns
    assert "PERSON" in df.columns


def test_fetch_recent_pass_rates(setup_test_db):
    df = metrics.fetch_recent_pass_rates()
    assert not df.empty
    assert "pass_rate_pct" in df.columns