import certifi
from fastapi import FastAPI, Request, Form, HTTPException, Query, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from faster_whisper import WhisperModel
from pymongo import ASCENDING, DESCENDING
import re
from typing import List
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import json
import logging
from typing import List
from pymongo.errors import ServerSelectionTimeoutError
from dotenv import load_dotenv
import bcrypt
import os
import openai
from openai import OpenAI
import tempfile
import base64
import httpx
import sys
sys.stdout.reconfigure(encoding='utf-8')
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-very-secret-key")
app.mount("/images", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory="templates")

try:
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=50000)

    db = client["englishpractice"]
    collection_history = db["histories"]
    collection_question = db["questions"]
    collection_user = db["user"]
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
1. Correct grammar errors and provide a more natural sentence. If the original sentence is fluent enough, keep it as is.
2. Score the original sentence on Conversational fluency (Very Good, Good, Okay, Fair, Meh)
3. 어떤 부분을 왜 교정했는지 한국어로 설명. If the original sentence is fluent enough, provide an empty array [].
Return your answer in **valid JSON** format as follows:
{{
  "corrected": "...",
  "explanations": ["반드시 한국어로 설명"],
  "conversational_fluency_score": "Very Good or Good or Okay or Fair or Meh"
}}
Sentence: {0}"""

@app.get("/", response_class=HTMLResponse)
async def get_login(request: Request):
    user = request.session.get("user")
    if user:
        return RedirectResponse(url="/main", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def post_register(request: Request):
    data = await request.json()
    userid = data.get("userid", "")
    password = data.get("password", "")
    gender = data.get("gender", "")

    document = { "userid": userid, "password": hash_password(password), "gender": gender, "createdAt": datetime.now(timezone.utc)}
    collection_user.insert_one(document)
    return JSONResponse(content={"message": "User registered successfully"})

@app.post("/login", response_class=HTMLResponse)
async def post_login(request: Request, userid: str = Form(...), password: str = Form(...)):
    user = await collection_user.find_one({"userid": userid})
    if user and verify_password(password, user["password"]):
        request.session["user"] = userid
    else:
        raise HTTPException(status_code=401, detail="아이디 혹은 패스워드가 잘못되었습니다.")
    return RedirectResponse(url="/main", status_code=302)

@app.post("/change_password", response_class=HTMLResponse)
async def post_change_password(request: Request):
    data = await request.json()
    userid = data.get("userid")
    original_password = data.get("original_password")
    new_password = data.get("new_password")
    confirm_password = data.get("confirm_password")

    if new_password != confirm_password:
        raise HTTPException(status_code=400, detail="New password and confirmation do not match.")

    user_doc = await collection_user.find_one({"userid": userid})
    if user_doc and verify_password(original_password, user_doc["password"]):
        await collection_user.update_one({"userid": userid}, {"$set": {"password": hash_password(new_password)}})
    else:
        return JSONResponse(content={  "message": "The original password is incorrect."}, status_code=400) 
    return JSONResponse(content={"message": "Password changed successfully"})

@app.get("/main", response_class=HTMLResponse)
async def get_questions(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/question", response_class=HTMLResponse)
async def get_questions(request: Request):
    user = request.session.get("user")
    if user != 'Joel':
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("question.html", {"request": request})

@app.get("/questions", response_class=HTMLResponse)
async def get_questions():
    latest_doc = await collection_question.find_one(sort=[("createdAt", DESCENDING)])
    return JSONResponse(content=latest_doc['questions'])

@app.get("/users", response_class=HTMLResponse)
async def get_users():
    cursor = collection_user.find({}, {"_id": 0, "userid": 1}).sort("userid", ASCENDING)
    users = await cursor.to_list(length=None)
    names = [user["userid"] for user in users]
    return JSONResponse(content=names)

@app.post("/questions", response_class=HTMLResponse)
async def post_questions(request: Request):
    user = request.session.get("user")
    if user != 'Joel':
        return RedirectResponse(url="/", status_code=302)
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
    reply['_id'] = str(history['_id'])
    reply['audioUrl'] = str(history['audioUrl']) if 'audioUrl' in history else ''
    return templates.TemplateResponse("detail.html", {"request": request, "reply": reply})

@app.get("/histories", response_class=HTMLResponse)
async def get_histories(request: Request, page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100)):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/", status_code=302)
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
         return RedirectResponse(url="/", status_code=302)
    
    
    client = OpenAI(api_key=os.getenv("OPENAPI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert in English grammar and conversational fluency."},
            {"role": "user", "content": prompt.format(chat.sentence)}
        ],
    )

    try:
        reply = response.choices[0].message.content
        results = await extract_clean_json_strings(reply.strip())
        formatted_reply = results[0].replace("\n", "")
        if not is_valid_json(formatted_reply):
            formatted_reply = await fix_json_string(formatted_reply)
        
        await collection_history.insert_one({
            "user": user,
            "reply": formatted_reply,
            "questionIndex": chat.questionIndex,
            "question": chat.question,
            "answer": chat.sentence,
            "createdAt": datetime.now(timezone.utc)
        })
        return {"reply": formatted_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#@app.post("/transcribe")
#async def transcribe_audio(file: UploadFile = File(...)):
#    try:
#        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_file:
#            content = await file.read()
#            temp_file.write(content)
#            temp_path = temp_file.name
#        client = OpenAI(api_key=os.getenv("OPENAPI_API_KEY"))
#        with open(temp_path, "rb") as audio_file:
#            response = client.audio.transcriptions.create(
#                model="whisper-1",
#                file=audio_file,
#                response_format="json",
#                language="en"
#            )
#
#        return JSONResponse(content={"text": response.text})
#
#    except Exception as e:
#        return JSONResponse(status_code=500, content={"error": str(e)})
#
#    finally:
#        if os.path.exists(temp_path):
#            os.remove(temp_path) 

model = WhisperModel("base", device="cpu", compute_type="int8")
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        segments, info = model.transcribe(temp_path)
        text = " ".join([segment.text for segment in segments])
        return JSONResponse(content={"text": text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/upload-voice")
async def upload_voice(request: Request):
    user = request.session.get("user")
    if not user:
         return RedirectResponse(url="/", status_code=302)
    currentUser = await collection_user.find_one({"userid": user})
    gender = currentUser.pop("gender", "female")
    voice = "shimmer" if gender == "female" else "echo"
    data = await request.json()
    historyId = data.get("historyId", "")
    corrected = data.get("corrected", "")
    
    client = OpenAI(api_key=os.getenv("OPENAPI_API_KEY"))
    speech_response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=corrected
    )

    encoded_content = base64.b64encode(speech_response.content).decode("utf-8")

    filename = f"uploads/{historyId}.mp3"
    api_url = f"https://api.github.com/repos/{os.getenv('GITHUB_REPO')}/contents/{filename}"

    headers = {
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
        "Accept": "application/vnd.github+json",
    }

    data = {
        "message": f"upload {historyId}.mp3",
        "content": encoded_content,
        "branch": os.getenv("GITHUB_BRANCH"),
    }

    async with httpx.AsyncClient() as client:
        response = await client.put(api_url, headers=headers, json=data)

    if response.status_code in [200, 201]:
        audioUrl = f"https://raw.githubusercontent.com/{os.getenv('GITHUB_REPO')}/{os.getenv('GITHUB_BRANCH')}/{filename}"
        dbUpdateResult = await collection_history.update_one({"_id": ObjectId(historyId)}, {"$set": {"audioUrl": audioUrl}})
        return JSONResponse(content={"message": "File uploaded", "audioUrl": audioUrl})
    else:
        return JSONResponse(
            status_code=response.status_code,
            content={"error": "GitHub upload failed", "detail": response.text},
        )
    
async def fix_json_string(json_str: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAPI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are an expert in json formatting."},
            {"role": "user", "content": "Fix the following string to **valid JSON format** and return the JSON: " + json_str}
        ],
    )
    return response.choices[0].message.content

def is_valid_json(json_str: str) -> bool:
    try:
        json.loads(json_str)
        return True
    except ValueError:
        return False

async def extract_clean_json_strings(text: str) -> List[str]:
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

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


