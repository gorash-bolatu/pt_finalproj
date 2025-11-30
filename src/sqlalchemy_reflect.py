from sqlalchemy import create_engine, MetaData, Table, select
from sqlalchemy.orm import Session
from db import engine  # if db.py uses create_engine with same URL, import it

meta = MetaData()
meta.reflect(bind=engine)  # reflect existing DB (created by Django migrations)

User = Table('reviews_user', meta, autoload_with=engine)  # modify names if Django uses app_label_modelname
# But recommended: keep table names simple (see below)

with Session(engine) as session:
    stmt = select(User).limit(5)
    rows = session.execute(stmt).all()
    print(rows)
