from fastapi import APIRouter, Depends, HTTPException, Request
from motor.motor_asyncio import AsyncIOMotorDatabase
from model.user import UserCreate, UserLogin, ChangePassword
from service import userService

router = APIRouter(
    prefix="/user"
)

@router.post("/register")
async def registerUser(user: UserCreate):
    await userService.registerUser(user)
    return {"message": "Congratulations! We will reach out ASAP!"}

@router.post("/login", response_model=UserLogin)
async def login(user: UserCreate, request: Request):
    user_data = await userService.login(user)
    user_dict = user_data.dict()
    user_dict['createdAt'] = user_data.createdAt.isoformat()
    request.session["user"] = user_dict
    return user_data

@router.post("/changepassword")
async def changePassword(param: ChangePassword):
    await userService.changePassword(param)
    return {"message": "Your password has been successfully changed."}
