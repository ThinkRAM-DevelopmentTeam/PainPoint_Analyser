# --------------------------------------------------------------------------
# This file lists all the Python packages required to run the application.
# You can install all of them at once by running:
# pip install -r requirements.txt
# --------------------------------------------------------------------------

# Core web framework for building the API endpoints (/business-search, /chat).
fastapi

# ASGI server that runs the FastAPI application, allowing it to handle web traffic.
# The '[standard]' part includes recommended extras for better performance.
uvicorn[standard]

# A modern, asynchronous HTTP client used to make fast API calls to the
# Google Places API. It's crucial for the `async` and `await` functionality.
httpx

# Library for processing textual data. We use it specifically for its simple
# and effective sentiment analysis to identify "pain points" in reviews.
textblob

# Automatically loads environment variables from a `.env` file into the application's
# environment. This is how we securely manage API keys without hardcoding them.
python-dotenv

# An add-on for Pydantic that allows easy management of application settings,
# especially for loading variables from the .env file in Pydantic v2.
pydantic-settings

# The official Python client library for interacting with the OpenAI API.
# This is used for the /chat endpoint to send data to GPT models.
openai

# A library providing various caching mechanisms. We use its TTLCache (Time-to-Live Cache)
# to temporarily store results from the Google Places API, which significantly
# reduces costs by preventing duplicate API calls.
cachetools

# NEW: Asynchronous PostgreSQL driver for connecting to Neon DB
asyncpg
asyncio-windows-events

# main.py

import os
import httpx
import asyncio
import json
import time
from typing import List, Optional
from enum import Enum
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from textblob import TextBlob
from openai import OpenAI
import asyncpg
import asyncio

# This line is important for running asyncio on Windows
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Configuration ---
class Settings(BaseSettings):
    google_places_api_key: str
    openai_api_key: str
    database_url: str
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
openai_client = OpenAI(api_key=settings.openai_api_key)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Production-Grade Business Analyst AI",
    description="Search businesses with persistent database caching, AI analysis, and update capabilities.",
    version="8.0.0" # Version updated to reflect refactor
)

# --- Database Connection Pool ---
db_pool = None

@app.on_event("startup")
async def startup_event():
    """On app startup, create a database connection pool."""
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(settings.database_url)
        print("Database connection pool created successfully.")
    except Exception as e:
        print(f"FATAL: Could not connect to the database: {e}")
        db_pool = None

@app.on_event("shutdown")
async def shutdown_event():
    """On app shutdown, close the database connection pool."""
    if db_pool:
        await db_pool.close()
        print("Database connection pool closed.")

# --- Enums for User Selection ---
class DayOfWeek(str, Enum):
    monday = "Monday"; tuesday = "Tuesday"; wednesday = "Wednesday"; thursday = "Thursday"
    friday = "Friday"; saturday = "Saturday"; sunday = "Sunday"

class DataField(str, Enum):
    rating = "rating"; reviews = "reviews"; opening_hours = "opening_hours"
    website = "website"; phone_number = "formatted_phone_number"; email = "email"

# --- Pydantic Models (Full Version Restored) ---
class Review(BaseModel):
    author_name: str; rating: int; text: str; relative_time_description: str

class PainPoint(BaseModel):
    sentiment_polarity: float; review_text: str

class Business(BaseModel):
    name: str
    address: Optional[str] = None
    phone: Optional[str] = Field(None, alias="formatted_phone_number")
    website: Optional[str] = None
    opening_hours: Optional[List[str]] = None
    reviews: List[Review] = []
    pain_points: List[PainPoint] = []
    rating: Optional[float] = None
    total_ratings: Optional[int] = None
    google_maps_url: Optional[str] = Field(None, alias="url")
    data_source: Optional[str] = None
    cached_at: Optional[datetime] = None
    email: Optional[str] = None

class ChatRequest(BaseModel):
    business_data: List[Business]
    user_question: str = Field(..., example="Which business has the lowest rating?")
    use_lean_payload: bool = Field(True, description="Send a summarized version of the data to the AI to reduce cost.")

class ChatResponse(BaseModel):
    answer: str

class BusinessUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    email: Optional[str] = None

# --- Helper Functions (Full Versions Restored) ---
async def search_nearby_places(client: httpx.AsyncClient, params: dict, desired_total_results: int) -> list:
    all_places = []
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    while len(all_places) < desired_total_results:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        all_places.extend(data.get("results", []))
        pagetoken = data.get("next_page_token")
        if not pagetoken or len(all_places) >= desired_total_results: break
        params["pagetoken"] = pagetoken
        await asyncio.sleep(2)
    return all_places[:desired_total_results]

async def get_place_details(client: httpx.AsyncClient, place_id: str) -> dict:
    if not db_pool: raise HTTPException(status_code=503, detail="Database connection is not available.")
    async with db_pool.acquire() as connection:
        record = await connection.fetchrow("SELECT details_json, fetched_at FROM places_cache WHERE place_id = $1", place_id)
        if record and (datetime.now(timezone.utc) - record['fetched_at']) < timedelta(days=30):
            print(f"Cache HIT for place_id: {place_id}")
            cached_data = json.loads(record['details_json'])
            cached_data['data_source'] = f"DB Cache (fetched {record['fetched_at'].date()})"
            cached_data['cached_at'] = record['fetched_at']
            return cached_data
    
    print(f"Cache MISS for place_id: {place_id}")
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    full_fields = "place_id,name,formatted_address,rating,user_ratings_total,url,formatted_phone_number,website,reviews,opening_hours,email"
    params = {"key": settings.google_places_api_key, "place_id": place_id, "fields": full_fields}
    response = await client.get(url, params=params)
    response.raise_for_status()
    result = response.json().get("result", {})
    if result:
        current_time = datetime.now(timezone.utc)
        async with db_pool.acquire() as connection:
            await connection.execute("""
                INSERT INTO places_cache (place_id, details_json, fetched_at) VALUES ($1, $2, $3)
                ON CONFLICT (place_id) DO UPDATE SET details_json = $2, fetched_at = $3;
            """, place_id, json.dumps(result), current_time)
        result['data_source'] = "Live Google API (and saved to cache)"
        result['cached_at'] = current_time
    return result

def analyze_reviews_for_pain_points(reviews: Optional[List[dict]]) -> List[PainPoint]:
    if not reviews: return []
    return [
        PainPoint(sentiment_polarity=TextBlob(rev['text']).sentiment.polarity, review_text=rev['text'])
        for rev in reviews if rev.get("text") and TextBlob(rev['text']).sentiment.polarity < 0
    ]

# --- API Endpoints ---

# --- REFACTORED AND CORRECTED /business-search ENDPOINT ---
@app.get("/business-search", response_model=List[Business], summary="Search, Fetch, then Filter (with DB Caching)")
async def business_search(
    keyword: str = Query(..., example="restaurant"), lat: float = Query(..., example=40.7128),
    lng: float = Query(..., example=-74.0060), radius_m: int = Query(5000, gt=0),
    fields: List[DataField] = Query(..., description="Select data fields to return."),
    desired_total_results: int = Query(20, ge=1, le=60),
    max_results_to_process: int = Query(10, ge=1, le=60),
    min_rating: float = Query(0.0, ge=0.0, le=5.0),
    max_rating: float = Query(5.0, ge=0.0, le=5.0),
    min_reviews: int = Query(0, ge=0),
    has_pain_points: Optional[bool] = Query(None),
    open_on_day: Optional[DayOfWeek] = Query(None),
):
    # Backend validation remains crucial
    if (min_rating > 0.0 or max_rating < 5.0 or min_reviews > 0) and DataField.rating not in fields:
        raise HTTPException(status_code=400, detail="To filter by rating/reviews, you must select the 'rating' field.")
    # ... other validations ...

    async with httpx.AsyncClient() as client:
        try:
            search_params = {"key": settings.google_places_api_key, "location": f"{lat},{lng}", "radius": radius_m, "keyword": keyword}
            nearby_places = await search_nearby_places(client, search_params, desired_total_results)
            
            place_ids_to_process = [p["place_id"] for p in nearby_places[:max_results_to_process]]
            
            tasks = [get_place_details(client, place_id) for place_id in place_ids_to_process]
            place_details_list = await asyncio.gather(*tasks)

            final_results = []
            for details in place_details_list:
                if not details: continue

                place_rating = details.get("rating")
                place_reviews_total = details.get("user_ratings_total")
                if place_rating is None or not (min_rating <= place_rating <= max_rating): continue
                if place_reviews_total is None or place_reviews_total < min_reviews: continue

                reviews_data = details.get("reviews", [])
                pain_points = analyze_reviews_for_pain_points(reviews_data)
                if has_pain_points is True and not pain_points: continue
                if has_pain_points is False and pain_points: continue
                
                # ... other filters like open_on_day ...

                # Build the complete data object before checking which fields to return
                full_business_data = details.copy()
                full_business_data['pain_points'] = pain_points
                
                # Create a limited dictionary ONLY with fields the user requested
                requested_data_dict = {"name": details.get("name"), "address": details.get("formatted_address")}
                
                if DataField.rating in fields:
                    requested_data_dict.update({"rating": details.get("rating"), "total_ratings": details.get("user_ratings_total")})
                if DataField.reviews in fields:
                    requested_data_dict.update({"reviews": reviews_data, "pain_points": pain_points})
                if DataField.opening_hours in fields:
                     requested_data_dict.update({"opening_hours": details.get("opening_hours", {}).get("weekday_text", [])})
                if DataField.website in fields:
                    requested_data_dict.update({"website": details.get("website")})
                if DataField.phone_number in fields:
                    requested_data_dict.update({"formatted_phone_number": details.get("formatted_phone_number")})
                if DataField.email in fields:
                    requested_data_dict.update({"email": details.get("email")})

                # Always include these for context
                requested_data_dict.update({
                    "url": details.get("url"),
                    "data_source": details.get("data_source"),
                    "cached_at": details.get("cached_at")
                })
                
                final_results.append(Business(**requested_data_dict))

            return final_results
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=f"Error with Google API: {exc.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# --- RESTORED /chat Endpoint ---
@app.post("/chat", response_model=ChatResponse, summary="Chat with an AI with user-controlled payload optimization")
async def chat_with_ai(request: ChatRequest):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client is not initialized.")
    
    if request.use_lean_payload:
        lean_business_data = [
            b.model_dump(include={'name', 'rating', 'total_ratings', 'phone', 'website', 'address', 'email'})
            for b in request.business_data
        ]
        business_data_json = json.dumps(lean_business_data, indent=2)
        system_prompt = "You are a helpful business analyst. Answer questions ONLY based on the summarized JSON data provided. If info is not in the data, say it's not available."
    else:
        business_data_json = json.dumps([b.model_dump() for b in request.business_data], indent=2, default=str)
        system_prompt = "You are a helpful business analyst. Answer questions ONLY based on the complete JSON data provided. Do not make up information. If info is not in the data, say it's not available."

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the business data:\n{business_data_json}"},
                {"role": "user", "content": f"Please answer this question: {request.user_question}"}
            ]
        )
        return ChatResponse(answer=completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error with OpenAI: {str(e)}")


# --- RESTORED /business/{place_id} Endpoint ---
@app.put("/business/{place_id}", status_code=200, summary="Update Business Details in Cache")
async def update_business_details(place_id: str, business_update: BusinessUpdate):
    """
    Updates the details_json for a given place_id in the database cache.
    """
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database connection is not available.")

    async with db_pool.acquire() as connection:
        record = await connection.fetchrow("SELECT details_json FROM places_cache WHERE place_id = $1", place_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Business with place_id '{place_id}' not found in cache.")

        current_details = json.loads(record['details_json'])
        update_data = business_update.model_dump(exclude_unset=True)
        
        # Map model fields to the Google Places structure
        if 'name' in update_data: current_details['name'] = update_data['name']
        if 'phone' in update_data: current_details['formatted_phone_number'] = update_data['phone']
        if 'website' in update_data: current_details['website'] = update_data['website']
        if 'email' in update_data: current_details['email'] = update_data['email'] # Assuming email can be updated

        updated_details_json = json.dumps(current_details)
        await connection.execute(
            "UPDATE places_cache SET details_json = $1 WHERE place_id = $2",
            updated_details_json, place_id
        )

    return {"message": f"Business '{place_id}' updated successfully.", "updated_data": current_details}
