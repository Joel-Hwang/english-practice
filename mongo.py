from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ServerSelectionTimeoutError
import logging
import os
logger = logging.getLogger(__name__)
try:
    client = AsyncIOMotorClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=50000)

    db = client["englishpractice"]
    
    # 연결 확인
    client.admin.command("ping")
    logger.info("MongoDB connected successfully.")
except ServerSelectionTimeoutError:
    logger.exception("MongoDB connection failed.")
    raise RuntimeError("Could not connect to MongoDB. Please check your URI and network.")
