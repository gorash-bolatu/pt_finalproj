from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, ForeignKey
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# CONFIG

MYSQL_USER = "root"
MYSQL_PASS = "new_password"
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
MYSQL_DB   = "sentiment_db"

# mysqlclient driver:
DB_URL = (
    f"mysql+mysqldb://{MYSQL_USER}:{MYSQL_PASS}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    "?charset=utf8mb4"
)

engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


# ORM TABLES

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(255), index=True)
    user_id = Column(String(255), index=True)
    rating = Column(Float)
    review_text = Column(Text)
    sentiment = Column(Integer, nullable=True)  # 1 = pos, 0 = neg, None = not processed


class Product(Base):
    __tablename__ = "products"

    id = Column(String(255), primary_key=True)
    title = Column(Text)
    category = Column(String(255))
    price = Column(Float)

    reviews = relationship("Review", backref="product")


# INIT FUNCTION — creates tables if missing

def init_db():
    Base.metadata.create_all(engine)


# SESSION GENERATOR

def get_session():
    """Context manager for DB session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
