# api/main.py - FastAPI Backend with FREE Hugging Face Inference API
import os
import re
import time
import logging
from datetime import datetime
from typing import List

import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, validator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone

# --- Configuration & Initialization ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI with proper documentation
app = FastAPI(
    title="PM Internship Matching API",
    description="AI-powered internship matching system using vector similarity",
    version="1.1.0"
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # More permissive for development, lock down for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# --- Service Connections ---

# Initialize Pinecone using modern SDK (v3+)
try:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "internships")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    logger.info(f"Pinecone initialized with index: {PINECONE_INDEX_NAME}")
except Exception as e:
    pinecone_index = None
    logger.error(f"Failed to initialize Pinecone: {e}")

# Database connection function (optimized for serverless)
def get_db_connection():
    """Get a new database connection. Best practice for serverless."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DATABASE"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=int(os.getenv("POSTGRES_PORT", 5432))
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")

# Hugging Face FREE Inference API details
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN", "") # Optional but recommended for higher rate limits

# --- Input Validation and Sanitization ---

def sanitize_string(text: str) -> str:
    """Sanitize input string to prevent injection attacks."""
    if not text:
        return ""
    return re.sub(r'[<>\"\'%;()&+\x00-\x1f\x7f-\x9f]', '', str(text).strip())

def validate_uid(uid: str) -> str:
    """Validate user ID format."""
    if not uid or not re.match(r'^[a-zA-Z0-9_-]{3,50}$', uid):
        raise ValueError("Invalid user ID format")
    return uid

# --- Pydantic Models ---

class UserProfile(BaseModel):
    uid: str
    name: str
    email: str
    phone: str
    domain: str
    skills: str
    location: str
    degree: str
    duration: str
    stipend: str
    
    @validator('uid')
    def validate_uid_field(cls, v):
        return validate_uid(v)
    
    @validator('name', 'domain', 'skills', 'location', 'degree', 'duration', 'stipend')
    def validate_text_fields(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Field cannot be empty")
        return sanitize_string(v)

class InternshipResponse(BaseModel):
    id: str
    title: str
    company: str
    location: str
    duration: str
    stipend: str
    domain: str
    skills_required: str
    description: str
    eligibility: str
    match_score: float

class ApplicationRequest(BaseModel):
    user_id: str
    internship_id: str
    
    @validator('user_id', 'internship_id')
    def validate_ids(cls, v):
        return validate_uid(v)

# --- Core Logic ---

def create_embedding(text: str, max_retries: int = 3) -> List[float]:
    """Create embedding using Hugging Face's FREE Inference API with retry logic."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {"inputs": text[:1024], "options": {"wait_for_model": True}}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)
            
            if response.status_code == 200:
                embedding = response.json()
                if isinstance(embedding, list) and len(embedding) > 0:
                    result = embedding[0] if isinstance(embedding[0], list) else embedding
                    if len(result) == 384:
                        return result
                raise ValueError(f"Unexpected embedding format: {embedding}")

            wait_time = min(2 ** attempt, 15)
            logger.warning(f"HF API returned {response.status_code}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(2 ** attempt)
            else:
                raise HTTPException(status_code=504, detail="Embedding API timeout")
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail="Failed to create embedding")
            time.sleep(2 ** attempt)
            
    raise HTTPException(status_code=500, detail="Failed to create embedding after all retries")

def create_profile_text(profile: UserProfile) -> str:
    """Convert user profile to a descriptive text string for embedding."""
    return (
        f"Desired Domain: {profile.domain}. "
        f"Skills: {profile.skills}. "
        f"Preferred Location: {profile.location}. "
        f"Education: {profile.degree}. "
        f"Preferred Duration: {profile.duration}. "
        f"Stipend Expectation: {profile.stipend}."
    )

def calculate_weighted_score(profile: UserProfile, internship: dict, base_score: float) -> float:
    """Calculate a weighted match score based on vector similarity and keyword matches."""
    score = base_score
    
    # Boost score for exact domain match
    if profile.domain.lower() in internship.get('domain', '').lower():
        score += 0.1
        
    # Boost score based on skill overlap
    profile_skills = set(s.strip().lower() for s in profile.skills.split(','))
    required_skills = set(s.strip().lower() for s in internship.get('skills_required', '').split(','))
    if required_skills:
        skill_overlap = len(profile_skills.intersection(required_skills)) / len(required_skills)
        score += skill_overlap * 0.2 # Skills are heavily weighted
        
    return min(score, 1.0)

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "PM Internship Matching API is running"}

@app.post("/match-internships", response_model=List[InternshipResponse])
async def match_internships(profile: UserProfile):
    """
    Matches a user profile with internships using vector similarity search.
    This endpoint now performs a single query to Pinecone to get all necessary data.
    """
    try:
        logger.info(f"Matching internships for user: {profile.uid}")
        
        # 1. Create embedding from user profile
        profile_text = create_profile_text(profile)
        user_embedding = create_embedding(profile_text)
        
        # 2. Query Pinecone for similar internships, including metadata
        if not pinecone_index:
            raise HTTPException(status_code=503, detail="Vector search service unavailable")
            
        query_results = pinecone_index.query(
            vector=user_embedding,
            top_k=15,
            include_metadata=True # CRITICAL CHANGE FOR PERFORMANCE
        )
        
        if not query_results['matches']:
            logger.info("No initial matches found in vector search.")
            return []
            
        # 3. Process results without a second database call
        results = []
        for match in query_results['matches']:
            internship_meta = match['metadata']
            
            # Ensure metadata exists before processing
            if not internship_meta:
                continue

            weighted_score = calculate_weighted_score(profile, internship_meta, match['score'])
            
            results.append(InternshipResponse(
                id=match['id'],
                title=sanitize_string(internship_meta.get('title', '')),
                company=sanitize_string(internship_meta.get('company', '')),
                location=sanitize_string(internship_meta.get('location', '')),
                duration=sanitize_string(internship_meta.get('duration', '')),
                stipend=sanitize_string(internship_meta.get('stipend', '')),
                domain=sanitize_string(internship_meta.get('domain', '')),
                skills_required=sanitize_string(internship_meta.get('skills_required', '')),
                description=sanitize_string(internship_meta.get('description', '')),
                eligibility=sanitize_string(internship_meta.get('eligibility', '')),
                match_score=round(weighted_score, 3)
            ))
            
        # 4. Sort by the final weighted score and return top 10
        results.sort(key=lambda x: x.match_score, reverse=True)
        logger.info(f"Returning {len(results[:10])} matches for user: {profile.uid}")
        return results[:10]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in match_internships: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred")

@app.post("/apply-internship")
async def apply_for_internship(application: ApplicationRequest):
    """Record an internship application in the database."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Check if already applied
            cur.execute(
                "SELECT id FROM applications WHERE user_id = %s AND internship_id = %s",
                (application.user_id, application.internship_id)
            )
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="Already applied to this internship")
            
            # Insert new application
            cur.execute(
                """
                INSERT INTO applications (user_id, internship_id, applied_at, status)
                VALUES (%s, %s, %s, %s) RETURNING id
                """,
                (application.user_id, application.internship_id, datetime.now(), 'pending')
            )
            application_id = cur.fetchone()[0]
            conn.commit()
            
        logger.info(f"Application created: {application_id} for user: {application.user_id}")
        return {"success": True, "application_id": application_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying for internship: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail="Could not process application")
    finally:
        if conn:
            conn.close()

@app.get("/user-applications/{user_id}")
async def get_user_applications(user_id: str):
    """Get all applications for a specific user."""
    conn = None
    try:
        validated_user_id = validate_uid(user_id)
        conn = get_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT a.id, a.applied_at, a.status, 
                       i.id as internship_id, i.title, i.company, i.location
                FROM applications a
                JOIN internships i ON a.internship_id = i.id
                WHERE a.user_id = %s
                ORDER BY a.applied_at DESC
                """,
                (validated_user_id,)
            )
            applications = cur.fetchall()
            
        return applications
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching user applications: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch applications")
    finally:
        if conn:
            conn.close()

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    db_status = "disconnected"
    conn = None
    try:
        conn = get_db_connection()
        conn.cursor().execute("SELECT 1")
        db_status = "connected"
    except Exception:
        pass
    finally:
        if conn:
            conn.close()

    pinecone_status = "unavailable"
    if pinecone_index:
        try:
            pinecone_index.describe_index_stats()
            pinecone_status = "connected"
        except Exception:
            pass

    return {
        "status": "healthy" if db_status == "connected" and pinecone_status == "connected" else "unhealthy",
        "services": {
            "database": db_status,
            "pinecone": pinecone_status
        }
    }

# For Vercel serverless deployment
handler = app