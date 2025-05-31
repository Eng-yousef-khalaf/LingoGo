from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from typing import List, Optional

app = FastAPI()

# Load the model globally when the app starts
try:
    with open("user_performance_model.pkl", "rb") as f:
        model = pickle.load(f)
        print("Model loaded successfully!")  # Confirm that the model is loaded
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Define Pydantic models for the request body
class Slide(BaseModel):
    id: int
    source: str
    nativeLanguage: str
    translation: str

class Presentation(BaseModel):
    userId: str
    translatingFrom: str
    translatingTo: str
    slides: List[Slide]

class Level(BaseModel):
    userId: str
    score: int
    attempts: int
    completed: bool
    levelNumber: int
    topic: str
    nativeLanguage: Optional[str]
    targetLanguage: Optional[str]

class PerformanceData(BaseModel):
    presentations: List[Presentation]
    levels: List[Level]

@app.post("/predict_performance")
async def predict_performance(data: PerformanceData):
    presentations = data.presentations
    levels = data.levels

    # If neither presentations nor levels are provided, return an error
    if not presentations and not levels:
        raise HTTPException(status_code=400, detail="Must provide either presentations or levels data.")

    users_data = []
    user_ids = set()

    # Collect unique userIds from presentations and levels
    for p in presentations:
        user_ids.add(p.userId)
    for l in levels:
        user_ids.add(l.userId)

    # Process data for each user
    for user_id in user_ids:
        user_presentations = [p for p in presentations if p.userId == user_id]
        user_levels = [l for l in levels if l.userId == user_id]

        num_presentations = len(user_presentations)
        num_levels = len(user_levels)
        total_quiz_score = sum(l.score for l in user_levels) if user_levels else 0
        total_attempts = sum(l.attempts for l in user_levels) if user_levels else 1
        avg_quiz_score = total_quiz_score / max(num_levels, 1) if num_levels > 0 else 0
        num_slides = sum(len(p.slides) for p in user_presentations) if user_presentations else 0

        # Add more features (e.g., level-related features like 'topic', 'nativeLanguage', etc.)
        topics = [l.topic for l in user_levels]  # Example: topic feature
        unique_topics = len(set(topics))  # Count unique topics
        language_pair = sum(1 for p in user_presentations if p.translatingFrom == p.translatingTo)  # Count same language pairs

        users_data.append({
            "userId": user_id,
            "num_presentations": num_presentations,
            "num_levels": num_levels,
            "total_quiz_score": total_quiz_score,
            "avg_quiz_score": avg_quiz_score,
            "num_slides": num_slides,
            "unique_topics": unique_topics,  # New feature: unique topics
            "language_pair": language_pair,  # New feature: language pairs
        })

    # If no users' data is generated or model is not loaded, raise an error
    if not users_data or model is None:
        raise HTTPException(status_code=500, detail="Not enough data to predict performance or model not loaded.")

    # Debug: Print the features that are being passed to the model
    print("Users data:", users_data)

    # Create a DataFrame for prediction
    df = pd.DataFrame(users_data)

    # Check if the expected columns are present
    expected_columns = ["num_presentations", "num_levels", "total_quiz_score", "avg_quiz_score", "num_slides", "unique_topics", "language_pair"]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=500, detail=f"Missing columns: {', '.join(missing_columns)}")

    print("Features used for prediction:", df.columns.tolist())  # Debugging: print columns being used

    # Pass data to model and predict
    X = df[expected_columns]

    try:
        # Predict user performance using the model
        df["overall"] = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {str(e)}")

    # Calculate other skills based on overall performance
    df["vocabulary"] = df["overall"] * 0.4
    df["grammar"] = df["overall"] * 0.3
    df["writing_skills"] = df["overall"] * 0.3

    # Format the result
    formatted_data = [
        {
            "ID": row["userId"],
            "Overall": round(row["overall"], 2),
            "Vocabulary": round(row["vocabulary"], 2),
            "Grammar": round(row["grammar"], 2),
            "Writing Skills": round(row["writing_skills"], 2)
        }
        for _, row in df.iterrows()
    ]

    return {"status": "success", "data": formatted_data}
