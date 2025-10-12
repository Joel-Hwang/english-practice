from motor.motor_asyncio import AsyncIOMotorDatabase
from model.user import User
from mongo import db

collection_user = db["user"]

async def insertUser(user: User):
    await collection_user.insert_one(user.dict())

async def findUserById(user_id: str):
    return await collection_user.find_one({"id": user_id})
