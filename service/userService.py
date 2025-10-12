from fastapi import HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from model.user import User, UserCreate, UserLogin
from repository import userRepository
import bcrypt

async def registerUser(user_create: UserCreate):
    if not user_create.id or not user_create.password:
        raise HTTPException(status_code=422, detail="ID and password are required.")

    existing_user = await userRepository.findUserById(user_create.id)
    if existing_user:
        raise HTTPException(status_code=400, detail="Somebody already took the ID. Please use different one.")
    
    hashed_password = bcrypt.hashpw(user_create.password.encode('utf-8'), bcrypt.gensalt())
    user = User(id=user_create.id, password=hashed_password.decode('utf-8'))
    await userRepository.insertUser(user)

async def login( user_login: UserCreate) -> UserLogin:
    if not user_login.id or not user_login.password:
        raise HTTPException(status_code=422, detail="ID and password are required.")

    user_data = await userRepository.findUserById(user_login.id)
    if not user_data:
        raise HTTPException(status_code=400, detail="We can't find your ID. Please try with a different one.")

    user = User(**user_data)

    if user.status != 'active':
        raise HTTPException(status_code=400, detail="Sorry, you aren't approved yet.")

    if not bcrypt.checkpw(user_login.password.encode('utf-8'), user.password.encode('utf-8')):
        raise HTTPException(status_code=400, detail="Please check your password again.")

    return UserLogin(id=user.id, gender=user.gender, status=user.status, createdAt=user.createdAt)
