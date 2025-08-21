import os

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

feature_store_url = os.getenv("FEATURE_STORE_URL", "").replace(
    "mysql+pymysql", "mysql+aiomysql"
)

engine = create_async_engine(feature_store_url, echo=False)

# SQLAlchemy 1.4에서는 sessionmaker에 class_=AsyncSession을 전달합니다.
SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

Base = declarative_base()