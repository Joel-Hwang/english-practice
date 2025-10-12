from fastapi import HTTPException
from model.question import Question
from repository import questionRepository
from model.user import User

async def registerQuestion(param: Question, user: User):
    if user.id != 'Joel':
        raise HTTPException(status_code=302, detail="You are not authorized to register questions.")
    await questionRepository.registerQuestion(param)

async def retrieveQuestion(group: str):
    question = await questionRepository.retrieveQuestion(group)
    if not question:
        return []
    return question
