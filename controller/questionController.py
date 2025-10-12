from fastapi import APIRouter, Depends, Request
from model.question import Question
from service import questionService
from model.user import User

router = APIRouter(
    prefix="/question"
)

def get_user_from_session(request: Request) -> User:
    user_dict = request.session.get("user")
    if not user_dict:
        return None
    return User(**user_dict)

@router.post("/register")
async def registerQuestion(param: Question, user: User = Depends(get_user_from_session)):
    await questionService.registerQuestion(param, user)
    return {"message": "The questions have been successfully registered."}

@router.get("/")
async def retrieveQuestion(user: User = Depends(get_user_from_session)):
    return await questionService.retrieveQuestion(user.group)
