from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Dict, Any, List
import motor.motor_asyncio
import numpy as np
from annoy import AnnoyIndex
import asyncio
from bson import ObjectId

app = FastAPI(title="Hokie Event Recommendation System")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("EXPRESS_BACKEND_URL", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = mongo_client.events_db

# Feature space definition based on your updated categories and subcategories
categories = [
    "Technology", "Sports", "Music", "Art", "Travel", "Gaming", "Fitness", "Social Events", "Others"
]
subcategories = [
    "Software & Web Development", "Data Science & Machine Learning", "Cybersecurity & Privacy", "Tech Innovation & Startups",
    "Team Sports", "Outdoor & Adventure Sports", "Individual Sports",
    "Live Performances & Concerts", "Music Festivals",
    "Performing Arts", "Art Exhibitions & Workshops",
    "Adventure Travel & Expeditions", "Cultural & Heritage Tours", "Weekend Getaways & Road Trips",
    "eSports & Competitive Gaming", "Board Games & Tabletop Meetups", "Virtual Reality & Tech Gaming",
    "Yoga & Mindfulness", "Outdoor Fitness",
    "Networking & Professional Meetups", "Community Volunteering & Social Causes", "Social Gatherings & Parties", "Book Clubs & Interest-Based Groups",
    "Sub-Others"
]

feature_space = categories + subcategories

@app.post("/recommend/{user_id}")
async def recommend_events(user_id: str, top_n: int = 10):
    try:
        # Fetch user profile, clickthroughs, and RSVPs
        user_profile = await db.userprofiles.find_one({"_id": ObjectId(user_id)})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")

        click_counts = await db.clickcounts.find_one({"userId": user_profile["emailAddresses"]})
        if not click_counts:
            click_counts = {"category": "", "categoryCount": 0, "subCategories": []}

        rsvp_counts = await db.events.find({"rsvps": user_profile["emailAddresses"]}).to_list(length=None)

        user_vector = create_user_vector(user_profile, click_counts, rsvp_counts, feature_space)

        # Fetch all events to create the Annoy index
        events_cursor = db.events.find()
        events = await events_cursor.to_list(length=None)

        # Create Annoy index for event vectors
        annoy_index = AnnoyIndex(len(feature_space) + 2, 'angular')
        event_vectors = []
        for idx, event in enumerate(events):
            event_vector = create_event_vector(event, feature_space)
            event_vectors.append((idx, event))
            annoy_index.add_item(idx, event_vector)

        # Build the Annoy index
        annoy_index.build(n_trees=10)

        # Get recommendations
        similar_events_idx = annoy_index.get_nns_by_vector(user_vector, top_n, include_distances=True)
        recommendations = [
            {"event": events[idx]["title"], "distance": distance}
            for idx, distance in zip(similar_events_idx[0], similar_events_idx[1])
        ]

        return {"recommended_events": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_user_vector(user_profile: Dict[str, Any], click_counts: Dict[str, Any], rsvp_counts: List[Dict[str, Any]], feature_space: List[str]) -> np.ndarray:
    vector = np.zeros(len(feature_space))

    # Encode user interests
    user_interests = user_profile.get("interests", [])
    for interest in user_interests:
        if interest in feature_space:
            vector[feature_space.index(interest)] = 1  # Assign a score for user interest

    # Encode clickthrough and RSVP counts
    for subcategory_info in click_counts.get("subCategories", []):
        subcategory = subcategory_info.get("category", "")
        count = subcategory_info.get("categoryCount", 0)
        if subcategory in feature_space:
            vector[feature_space.index(subcategory)] += count

    # Encode RSVPs
    for rsvp_event in rsvp_counts:
        main_category = rsvp_event.get("main_category", "")
        sub_category = rsvp_event.get("sub_category", "")
        if main_category in feature_space:
            vector[feature_space.index(main_category)] += 2  # RSVPs weighted more heavily
        if sub_category in feature_space:
            vector[feature_space.index(sub_category)] += 2

    # Dummy location data (could be enhanced with user location)
    vector = np.append(vector, [0, 0])  # Latitude and longitude as placeholders

    return vector

def create_event_vector(event: Dict[str, Any], feature_space: List[str]) -> np.ndarray:
    vector = np.zeros(len(feature_space))

    # Encode main category and subcategory
    main_category = event.get("main_category", "")
    sub_category = event.get("sub_category", "")

    if main_category in feature_space:
        vector[feature_space.index(main_category)] = 1

    if sub_category in feature_space:
        vector[feature_space.index(sub_category)] = 1

    # Encode event location (latitude and longitude)
    latitude = event.get("Latitude", 0)
    longitude = event.get("Longitude", 0)
    vector = np.append(vector, [latitude, longitude])

    return vector

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
