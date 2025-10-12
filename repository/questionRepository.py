from motor.motor_asyncio import AsyncIOMotorDatabase
from model.question import Question
from mongo import db

collection_question = db["question"]

async def registerQuestion(param: Question):
    await collection_question.insert_one(param.model_dump())

async def retrieveQuestion(group: str):
    return await collection_question.find_one({"group": group}, sort=[("createdAt", -1)])
