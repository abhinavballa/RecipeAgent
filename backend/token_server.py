from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from livekit import api
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="LiveKit Token Server", version="1.0.0")

# Add CORS for your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TokenRequest(BaseModel):
    room: str
    identity: str

@app.get("/")
async def root():
    return {"message": "LiveKit Token Server is running"}

@app.post("/api/token")
async def generate_token(request: TokenRequest):
    try:
        # Get LiveKit credentials from environment
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        
        if not api_key or not api_secret:
            raise HTTPException(
                status_code=500, 
                detail="LiveKit credentials not configured"
            )
        
        # Use official LiveKit SDK to create token (following docs exactly)
        token = api.AccessToken(api_key, api_secret) \
            .with_identity(request.identity) \
            .with_name(request.identity) \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=request.room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True
            )).to_jwt()
        
        return {
            "token": token,
            "room": request.room,
            "identity": request.identity,
            "message": "Token generated using official LiveKit SDK"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token generation failed: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting LiveKit Token Server...")
    print("📡 Server will run on http://localhost:8000")
    print("🔑 JWT tokens available at POST /api/token")
    uvicorn.run(app, host="0.0.0.0", port=8000)
