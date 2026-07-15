import os
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from fastapi import HTTPException

from app.system_prompt import SYSTEM_PROMPT

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = "gemini-3.1-flash-lite"

client = genai.Client(api_key=GEMINI_API_KEY)
generate_config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)


def call_gemini(question: str) -> str:
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=question,
            config=generate_config,
        )
        return response.text
    except genai_errors.APIError as e:
        # ---> THIS WILL PRINT THE REAL ERROR IN YOUR TERMINAL <---
        print("\n" + "="*50)
        print("🔍 GOOGLE GEMINI API ERROR DETECTED:")
        print(f"Status Code: {e.code}")
        print(f"Message: {e.message}")
        print("="*50 + "\n")
        
        # Propagate the actual message down so you see it in Postman/Browser
        raise HTTPException(
            status_code=502, 
            detail=f"Gemini rejected request: Code {e.code} - {e.message}"
        )

