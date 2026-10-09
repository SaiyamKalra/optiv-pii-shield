import os
from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///optiv_shield.db")
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class DocumentSummary(Base):
    __tablename__ = "document_summary"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String, index=True)
    file_name = Column(String)
    page_number = Column(Integer)
    status = Column(String)
    verification_passed = Column(Boolean)
    total_entities_detected = Column(Integer)
    processed_at = Column(DateTime, default=lambda: datetime.now(UTC))


class PIIDetectionLog(Base):
    __tablename__ = "pii_detection_logs"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String, index=True)
    token = Column(String)
    entity_type = Column(String, index=True)
    confidence_score = Column(Float)
    page_number = Column(Integer)
    processed_at = Column(DateTime, default=lambda: datetime.now(UTC))


def init_db():
    Base.metadata.create_all(bind=engine)
