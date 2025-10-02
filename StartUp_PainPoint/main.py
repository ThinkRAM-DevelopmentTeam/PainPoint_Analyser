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

# main.py

import os
import httpx
import asyncio
import json
from typing import List, Optional
from enum import Enum

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from textblob import TextBlob
from openai import OpenAI
from cachetools import TTLCache # Import the caching library

# --- Configuration ---
class Settings(BaseSettings):
    google_places_api_key: str
    openai_api_key: str
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
openai_client = OpenAI(api_key=settings.openai_api_key)

# --- Caching Setup ---
# Create a cache that holds up to 500 items, and each item expires after 1 hour (3600 seconds)
# This means if you search the same area again, you won't be re-charged for the same business details.
details_cache = TTLCache(maxsize=500, ttl=3600)

# --- Enum for Day of the Week Filter ---
class DayOfWeek(str, Enum):
    monday = "Monday"; tuesday = "Tuesday"; wednesday = "Wednesday"; thursday = "Thursday"
    friday = "Friday"; saturday = "Saturday"; sunday = "Sunday"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Cost-Optimized Business Analyst AI",
    description="Search businesses, filter them efficiently to reduce API costs, and use an AI chatbot to analyze the results.",
    version="4.0.0"
)

# --- Pydantic Models (No changes here) ---
class Review(BaseModel):
    author_name: str; rating: int; text: str; relative_time_description: str
class PainPoint(BaseModel):
    sentiment_polarity: float; review_text: str
class Business(BaseModel):
    name: str; address: Optional[str]; phone: Optional[str]; website: Optional[str]
    opening_hours: Optional[List[str]]; reviews: List[Review]; pain_points: List[PainPoint]
    rating: Optional[float]; total_ratings: Optional[int]; google_maps_url: Optional[str]
class ChatRequest(BaseModel):
    business_data: List[Business]
    user_question: str = Field(..., example="Which business has the lowest rating?")
class ChatResponse(BaseModel):
    answer: str

# --- Helper Functions ---
async def search_nearby_places(client: httpx.AsyncClient, lat: float, lng: float, keyword: str, radius_m: int) -> list:
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {"key": settings.google_places_api_key, "location": f"{lat},{lng}", "radius": radius_m, "keyword": keyword}
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json().get("results", [])

# UPDATED get_place_details to use the cache
async def get_place_details(client: httpx.AsyncClient, place_id: str) -> dict:
    """Gets details for a place, using a cache to avoid duplicate API calls."""
    if place_id in details_cache:
        return details_cache[place_id] # Return from cache if available
    
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"key": settings.google_places_api_key, "place_id": place_id, "fields": "name,formatted_address,formatted_phone_number,website,opening_hours,reviews,rating,user_ratings_total,url"}
    response = await client.get(url, params=params)
    response.raise_for_status()
    result = response.json().get("result", {})
    
    if result:
        details_cache[place_id] = result # Store successful result in cache
    return result

def analyze_reviews_for_pain_points(reviews: Optional[List[dict]]) -> List[PainPoint]:
    if not reviews: return []
    # ... (code unchanged)
    pain_points = []
    for review in reviews:
        text = review.get("text")
        if not text: continue
        analysis = TextBlob(text)
        if analysis.sentiment.polarity < 0:
            pain_points.append(PainPoint(sentiment_polarity=analysis.sentiment.polarity, review_text=text))
    return pain_points

# --- API Endpoint: Business Search (Completely Reworked for Cost Optimization) ---
@app.get("/business-search", response_model=List[Business], summary="Cost-optimized business search with advanced filtering")
async def business_search(
    keyword: str = Query(..., example="restaurant"), lat: float = Query(..., example=40.7128),
    lng: float = Query(..., example=-74.0060), radius_m: int = Query(5000, gt=0),
    has_pain_points: Optional[bool] = Query(None), min_rating: Optional[float] = Query(None, ge=1, le=5),
    max_rating: Optional[float] = Query(None, ge=1, le=5), min_reviews: Optional[int] = Query(None, ge=0),
    open_on_day: Optional[DayOfWeek] = Query(None)
):
    if not settings.google_places_api_key or "YOUR_GOOGLE_API_KEY" in settings.google_places_api_key:
        raise HTTPException(status_code=500, detail="Google Places API key is not configured.")

    async with httpx.AsyncClient() as client:
        try:
            # --- STAGE 1: Initial (Cheaper) Search ---
            nearby_places = await search_nearby_places(client, lat, lng, keyword, radius_m)

            # --- STAGE 2: PRE-FILTERING (Cost-Saving Step) ---
            # Apply filters that don't require the expensive "Details" call.
            pre_filtered_place_ids = []
            for place in nearby_places:
                rating = place.get("rating")
                total_ratings = place.get("user_ratings_total")
                
                if min_rating is not None and (rating is None or rating < min_rating):
                    continue
                if max_rating is not None and (rating is None or rating > max_rating):
                    continue
                if min_reviews is not None and (total_ratings is None or total_ratings < min_reviews):
                    continue
                
                pre_filtered_place_ids.append(place["place_id"])

            # --- STAGE 3: DETAILED FETCH (Expensive Step on a smaller list) ---
            # Now, only get full details for the places that passed the pre-filter.
            tasks = [get_place_details(client, place_id) for place_id in pre_filtered_place_ids]
            place_details_list = await asyncio.gather(*tasks)

            # --- STAGE 4: FINAL FILTERING AND RESPONSE ---
            final_results = []
            for details in place_details_list:
                if not details: continue
                
                reviews_data = details.get("reviews", [])
                pain_points = analyze_reviews_for_pain_points(reviews_data)

                # Final filter for pain points
                if has_pain_points is not None and has_pain_points and not pain_points:
                    continue
                
                # Final filter for opening day
                opening_hours = details.get("opening_hours", {}).get("weekday_text", [])
                if open_on_day is not None:
                    if not opening_hours: continue
                    is_open = any(day.startswith(open_on_day.value) and "Closed" not in day for day in opening_hours)
                    if not is_open:
                        continue
                
                business_info = Business(
                    name=details.get("name"), address=details.get("formatted_address"), phone=details.get("formatted_phone_number"),
                    website=details.get("website"), opening_hours=opening_hours,
                    reviews=[Review(**review) for review in reviews_data] if reviews_data else [],
                    pain_points=pain_points, rating=details.get("rating"), total_ratings=details.get("user_ratings_total"),
                    google_maps_url=details.get("url")
                )
                final_results.append(business_info)

            return final_results
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=f"An error with Google API: {exc.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error: {str(e)}")


# --- Chat Endpoint (No changes needed here) ---
@app.post("/chat", response_model=ChatResponse, summary="Chat with an AI to analyze business data")
async def chat_with_ai(request: ChatRequest):
    # ... (code unchanged)
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client is not initialized. Check your OPENAI_API_KEY.")
    business_data_json = json.dumps([b.model_dump() for b in request.business_data], indent=2)
    system_prompt = "You are a helpful business analyst. Answer questions ONLY based on the JSON data provided. If info is not in the data, say it's not available."
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
