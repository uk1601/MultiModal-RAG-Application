from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config.settings import settings
from app.models.user_model import Base

# Create the SQLAlchemy engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Ensures that connections are alive before using them
    pool_size=10,  # Defines the number of connections in the pool
    max_overflow=20  # Maximum number of connections that can be created beyond the pool size
)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables (used during initial setup)
Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print(f"Database session error: {str(e)}")  # Optional: You can add logging here
        raise  # Re-raise the exception after logging
    finally:
        db.close()  # Ensure the session is closed after usage
