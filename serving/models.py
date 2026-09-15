"""SQLAlchemy decision log model -- written to run unchanged on Postgres
later, so no Postgres-only column types."""

from datetime import UTC, datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Decision(Base):
    """Persistent record of a fraud decision for a transaction."""

    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True)
    transaction_id = Column(String, index=True, nullable=False)
    model_score = Column(Float, nullable=False)
    decision = Column(String, nullable=False)
    triggered_rules = Column(JSON, nullable=False, default=list)
    decision_source = Column(String, nullable=False)
    model_run_id = Column(String, nullable=False)
    shap_top_features = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))


def make_session_factory(db_url: str) -> sessionmaker:
    """Create an SQLAlchemy engine and session factory for the given database URL.

    Automatically creates all tables if they do not exist.

    Args:
        db_url: Database URL (e.g., 'sqlite:///test.db' or 'postgresql://...')

    Returns:
        A sessionmaker instance bound to the created engine.
    """
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
