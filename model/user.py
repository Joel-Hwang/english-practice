from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class User(BaseModel):
    id: str
    password: str
    gender: str = "female"
    status: str = "inactive"
    group: Optional[str]= None
    level: int = 1
    createdAt: datetime = datetime.now()

class UserCreate(BaseModel):
    id: str
    password: str

class UserLogin(BaseModel):
    id: str
    gender: str
    status: str
    group: str
    level: int
    createdAt: datetime

class ChangePassword(BaseModel):
    id: str
    newPassword: str
    oldPassword: str
