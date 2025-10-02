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
import asyncpg # Use the PostgreSQL database driver
import asyncio
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
# The app object needs to be defined before the lifespan events
app = FastAPI(
    title="Production-Grade Business Analyst AI",
    description="Search businesses with persistent database caching to dramatically reduce long-term API costs. Features user-driven field selection and an AI analyst.",
    version="7.0.0"
)

# --- Database Connection Pool ---
# A pool is the standard way to manage database connections in a web app
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
    rating = "rating"
    reviews = "reviews"
    opening_hours = "opening_hours"
    website = "website"
    phone_number = "formatted_phone_number"

# --- Pydantic Models ---
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
class ChatRequest(BaseModel):
    business_data: List[Business]
    user_question: str = Field(..., example="Which business has the lowest rating?")
    use_lean_payload: bool = Field(True, description="Send a summarized version of the data to the AI to reduce cost.")
class ChatResponse(BaseModel):
    answer: str

# --- Helper Functions ---
async def search_nearby_places(client: httpx.AsyncClient, params: dict, desired_total_results: int) -> list:
    all_places = []
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    while len(all_places) < desired_total_results:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        all_places.extend(data.get("results", []))
        pagetoken = data.get("next_page_token")
        if not pagetoken or len(all_places) >= desired_total_results:
            break
        params["pagetoken"] = pagetoken
        await asyncio.sleep(2)
    return all_places[:desired_total_results]

async def get_place_details(client: httpx.AsyncClient, place_id: str) -> dict:
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database connection is not available.")
    async with db_pool.acquire() as connection:
        record = await connection.fetchrow("SELECT details_json, fetched_at FROM places_cache WHERE place_id = $1", place_id)
        if record:
            age = datetime.now(timezone.utc) - record['fetched_at']
            if age < timedelta(days=30):
                print(f"Cache HIT for place_id: {place_id}")
                cached_data = json.loads(record['details_json'])
                cached_data['data_source'] = f"Database Cache (fetched on {record['fetched_at'].date()})"
                cached_data['cached_at'] = record['fetched_at']
                return cached_data
            print(f"Cache STALE for place_id: {place_id}. Data is {age.days} days old. Refetching.")
    
    print(f"Cache MISS for place_id: {place_id}. Calling Google API.")
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    full_fields = "place_id,name,formatted_address,rating,user_ratings_total,url,formatted_phone_number,website,reviews,opening_hours"
    params = {"key": settings.google_places_api_key, "place_id": place_id, "fields": full_fields}
    response = await client.get(url, params=params)
    response.raise_for_status()
    result = response.json().get("result", {})
    if result:
        result_json = json.dumps(result)
        current_time = datetime.now(timezone.utc)
        async with db_pool.acquire() as connection:
            await connection.execute("""
                INSERT INTO places_cache (place_id, details_json, fetched_at) VALUES ($1, $2, $3)
                ON CONFLICT (place_id) DO UPDATE SET details_json = $2, fetched_at = $3;
            """, place_id, result_json, current_time)
        result['data_source'] = "Live Google API (and saved to cache)"
        result['cached_at'] = current_time
    return result

def analyze_reviews_for_pain_points(reviews: Optional[List[dict]]) -> List[PainPoint]:
    if not reviews: return []
    pain_points = []
    for review in reviews:
        text = review.get("text")
        if not text: continue
        analysis = TextBlob(text)
        if analysis.sentiment.polarity < 0:
            pain_points.append(PainPoint(sentiment_polarity=analysis.sentiment.polarity, review_text=text))
    return pain_points

# --- API Endpoints ---
@app.get("/business-search", response_model=List[Business], summary="Select fields, then filter (with DB Caching)")
async def business_search(
    keyword: str = Query(..., example="restaurant"), lat: float = Query(..., example=40.7128),
    lng: float = Query(..., example=-74.0060), radius_m: int = Query(5000, gt=0),
    fields: List[DataField] = Query(..., description="Select data fields. Determines which filters you can use."),
    desired_total_results: int = Query(20, description="Total number of initial results to fetch (max 60).", ge=1, le=60),
    has_pain_points: Optional[bool] = Query(None), min_rating: Optional[float] = Query(None),
    max_rating: Optional[float] = Query(None), min_reviews: Optional[int] = Query(None),
    open_on_day: Optional[DayOfWeek] = Query(None),
    max_results_to_process: Optional[int] = Query(10, description="Limit processing to the top N results for cost control.")
):
    # --- Backend Validation ---
    if (min_rating is not None or max_rating is not None or min_reviews is not None) and DataField.rating not in fields:
        raise HTTPException(status_code=400, detail="To filter by rating/reviews, you must select the 'rating' field.")
    if has_pain_points and DataField.reviews not in fields:
        raise HTTPException(status_code=400, detail="To use 'has_pain_points' filter, you must select the 'reviews' field.")
    if open_on_day and DataField.opening_hours not in fields:
        raise HTTPException(status_code=400, detail="To filter by 'open_on_day', you must select the 'opening_hours' field.")

    async with httpx.AsyncClient() as client:
        try:
            # --- STAGE 1: Paginated Search ---
            search_params = {"key": settings.google_places_api_key, "location": f"{lat},{lng}", "radius": radius_m, "keyword": keyword}
            nearby_places = await search_nearby_places(client, search_params, desired_total_results)
            
            # --- STAGE 2: Pre-filtering ---
            pre_filtered_place_ids = []
            if DataField.rating in fields:
                for place in nearby_places:
                    if min_rating and (place.get("rating") is None or place.get("rating") < min_rating): continue
                    if max_rating and (place.get("rating") is None or place.get("rating") > max_rating): continue
                    if min_reviews and (place.get("user_ratings_total") is None or place.get("user_ratings_total") < min_reviews): continue
                    pre_filtered_place_ids.append(place["place_id"])
            else:
                pre_filtered_place_ids = [p["place_id"] for p in nearby_places]

            limited_place_ids = pre_filtered_place_ids[:max_results_to_process]
            
            # --- STAGE 3: Detailed Fetch (Now using the DB-aware function) ---
            tasks = [get_place_details(client, place_id) for place_id in limited_place_ids]
            place_details_list = await asyncio.gather(*tasks)

            # --- STAGE 4: Final Filtering ---
            final_results = []
            for details in place_details_list:
                if not details: continue
                # Apply filters based on the rich data from the cache/API
                if DataField.rating in fields:
                    if min_rating and (details.get("rating") is None or details.get("rating") < min_rating): continue
                    if max_rating and (details.get("rating") is None or details.get("rating") > max_rating): continue
                    if min_reviews and (details.get("user_ratings_total") is None or details.get("user_ratings_total") < min_reviews): continue
                
                reviews_data = details.get("reviews", [])
                pain_points = analyze_reviews_for_pain_points(reviews_data)
                if has_pain_points and not pain_points: continue
                
                opening_hours = details.get("opening_hours", {}).get("weekday_text", [])
                if open_on_day and not any(day.startswith(open_on_day.value) and "Closed" not in day for day in opening_hours): continue
                
                # Only include fields the user actually asked for in the final response
                requested_business_data = {
                    "data_source": details.get("data_source"), "cached_at": details.get("cached_at"),
                    "name": details.get("name"), "address": details.get("formatted_address"),
                    "google_maps_url": details.get("url")
                }
                if DataField.rating in fields:
                    requested_business_data.update({"rating": details.get("rating"), "total_ratings": details.get("user_ratings_total")})
                if DataField.reviews in fields:
                    requested_business_data.update({"reviews": reviews_data, "pain_points": pain_points})
                if DataField.opening_hours in fields:
                    requested_business_data.update({"opening_hours": opening_hours})
                if DataField.website in fields:
                    requested_business_data.update({"website": details.get("website")})
                if DataField.phone_number in fields:
                    requested_business_data.update({"phone": details.get("formatted_phone_number")})

                final_results.append(Business(**requested_business_data))

            return final_results
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=f"An error with Google API: {exc.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error: {str(e)}")


# --- Chat Endpoint (Unchanged from the previous version) ---
@app.post("/chat", response_model=ChatResponse, summary="Chat with an AI with user-controlled payload optimization")
async def chat_with_ai(request: ChatRequest):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client is not initialized.")
    
    if request.use_lean_payload:
        lean_business_data = []
        for business in request.business_data:
            lean_business_data.append({
                "name": business.name, "rating": business.rating, "total_ratings": business.total_ratings,
                "pain_points": [p.review_text for p in business.pain_points],
                "phone": business.phone, "website": business.website, "address": business.address
            })
        business_data_json = json.dumps(lean_business_data, indent=2)
        system_prompt = "You are a helpful business analyst. Answer questions ONLY based on the summarized JSON data provided. If info is not in the data, say it's not available."
    else:
        business_data_json = json.dumps([b.model_dump() for b in request.business_data], indent=2)
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
