from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
from pymongo import DESCENDING
import re
from typing import List
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import json
import logging
from pymongo.errors import ServerSelectionTimeoutError
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
import sys
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-very-secret-key")
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory="templates")

try:
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
    db = client["englishpractice"]
    collection_history = db["histories"]
    collection_question = db["questions"]
    # 연결 확인
    client.admin.command("ping")
    logger.info("MongoDB connected successfully.")
except ServerSelectionTimeoutError:
    logger.exception("MongoDB connection failed.")
    raise RuntimeError("Could not connect to MongoDB. Please check your URI and network.")

class ChatRequest(BaseModel):
    sentence: str
    question: str
    questionIndex: int

class ChatResponse(BaseModel):
    reply: str

prompt = """Given the sentence below, please do the following:
1. Correct grammar errors and provide a more natural, colloquial version of the sentence.
2. Score the original sentence on Conversational fluency (Very Good, Good, Okay, Fair, Meh)
3. 어떤 부분을 왜 교정했는지 한국어로 설명.
Return your answer in JSON format as follows:
{{
  "corrected": "...",
  "explanations": ["반드시 한국어로 설명"],
  "conversational_fluency_score": "Very Good"
}}
Sentence: {0}"""

@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def post_login(request: Request, username: str = Form(...), password: str = Form(...)):
    request.session["user"] = username
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/question", response_class=HTMLResponse)
async def get_questions(request: Request):
    user = request.session.get("user")
    if user != 'Joel':
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("question.html", {"request": request})

@app.get("/questions", response_class=HTMLResponse)
async def get_questions():
    latest_doc = await collection_question.find_one(sort=[("created_at", DESCENDING)])
    return JSONResponse(content=latest_doc['questions']) 

@app.post("/questions", response_class=HTMLResponse)
async def post_questions(request: Request):
    user = request.session.get("user")
    if user != 'Joel':
        return RedirectResponse(url="/login", status_code=302)
    data = await request.json()
    questions = data.get("questions", [])

    document = { "questions": questions, "createdAt": datetime.now(timezone.utc)}
    collection_question.insert_one(document)
    
    return JSONResponse(content={"message": "Questions received successfully"})

@app.get("/detail/{id}", response_class=HTMLResponse)
async def get_detail(id: str, request: Request):
    user = request.session.get("user")
    history = await collection_history.find_one({"user": user, "_id": ObjectId(id)})
    reply = json.loads(history['reply'])
    reply['question'] = history['question']
    reply['answer'] = history['answer']
    return templates.TemplateResponse("detail.html", {"request": request, "reply": reply})

@app.get("/histories", response_class=HTMLResponse)
async def get_histories(request: Request, page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100)):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    skip = (page - 1) * size

    cursor = (
        collection_history.find({"user": user})
        .sort("createdAt", -1)
        .skip(skip)
        .limit(size)
    )

    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["createdAt"] = doc["createdAt"].isoformat() 
        results.append(doc)
    return JSONResponse(content=results) 

@app.post("/chat", response_model=ChatResponse)
async def chat_with_lmstudio(request: Request, chat: ChatRequest):
    user = request.session.get("user")
    if not user:
         return RedirectResponse(url="/login", status_code=302)
    
    
    client = OpenAI(api_key=os.getenv("OPENAPI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in English grammar and conversational fluency."},
            {"role": "user", "content": prompt.format(chat.sentence)}
        ],
    )

    try:
        reply = response.choices[0].message.content
        print(reply)
        results = extract_clean_json_strings(reply.strip())
        print(results[0].replace("\n", ""))
        
        await collection_history.insert_one({
            "user": user, 
            "reply": results[0].replace("\n", ""), 
            "questionIndex": chat.questionIndex,
            "question": chat.question, 
            "answer":chat.sentence, 
            "createdAt": datetime.now(timezone.utc)})
        return {"reply": results[0].replace("\n", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import json
from typing import List

def extract_clean_json_strings(text: str) -> List[str]:
    pattern = r'\{.*?\}'  # 중괄호로 감싸인 JSON 문자열 추출 (비재귀, 단순)
    raw_matches = re.findall(pattern, text, re.DOTALL)

    results = []
    for m in raw_matches:
        try:
            obj = json.loads(m)
            clean_json_str = json.dumps(obj, ensure_ascii=False)
            results.append(clean_json_str)
        except Exception:
            results.append(m)

    if len(results) == 0:
        results = ["""{
  "corrected": "Error",
  "explanations": [],
  "conversational_fluency_score": "Error"
}"""]

    return results


