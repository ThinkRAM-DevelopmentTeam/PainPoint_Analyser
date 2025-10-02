import asyncio
from pydantic_settings import BaseSettings
import asyncpg

# This script is to be run ONCE to set up your database table.

class Settings(BaseSettings):
    # Expect all three variables, just like in main.py
    google_places_api_key: str
    openai_api_key: str
    database_url: str

    class Config:
        env_file = ".env"

settings = Settings()

async def create_table():
    """Connects to the database and creates the cache table if it doesn't exist."""
    conn = None
    try:
        print("Connecting to the database...")
        conn = await asyncpg.connect(settings.database_url)
        print("Connection successful.")
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS places_cache (
                place_id TEXT PRIMARY KEY,
                details_json JSONB NOT NULL,
                fetched_at TIMESTAMPTZ NOT NULL
            );
        """)
        
        print("Table 'places_cache' created or already exists.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            await conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    asyncio.run(create_table())
