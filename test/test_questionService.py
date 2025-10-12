import unittest
from unittest.mock import patch, AsyncMock
from fastapi import HTTPException
from service.questionService import registerQuestion, retrieveQuestion
from model.question import Question
from model.user import User
from datetime import datetime

class TestQuestionService(unittest.IsolatedAsyncioTestCase):

    @patch('repository.questionRepository.registerQuestion', new_callable=AsyncMock)
    async def test_registerQuestion_success(self, mock_registerQuestion):
        user = User(id='Joel', password='password', gender='male', status='active', group='default')
        question = Question(group='test_group', questions=['q1', 'q2'])
        
        await registerQuestion(question, user)
        
        mock_registerQuestion.assert_called_once_with(question)

    async def test_registerQuestion_unauthorized(self):
        user = User(id='NotJoel', password='password', gender='male', status='active', group='default')
        question = Question(group='test_group', questions=['q1', 'q2'])
        
        with self.assertRaises(HTTPException) as cm:
            await registerQuestion(question, user)
        
        self.assertEqual(cm.exception.status_code, 302)
        self.assertEqual(cm.exception.detail, 'You are not authorized to register questions.')

    @patch('repository.questionRepository.retrieveQuestion', new_callable=AsyncMock)
    async def test_retrieveQuestion_success(self, mock_retrieveQuestion):
        group = 'test_group'
        expected_question = {
            'group': group,
            'questions': ['q1', 'q2'],
            'createdAt': datetime.now()
        }
        mock_retrieveQuestion.return_value = expected_question
        
        result = await retrieveQuestion(group)
        
        self.assertEqual(result, expected_question)
        mock_retrieveQuestion.assert_called_once_with(group)

    @patch('repository.questionRepository.retrieveQuestion', new_callable=AsyncMock)
    async def test_retrieveQuestion_not_found(self, mock_retrieveQuestion):
        group = 'non_existent_group'
        mock_retrieveQuestion.return_value = None
        
        result = await retrieveQuestion(group)
        
        self.assertEqual(result, [])
        mock_retrieveQuestion.assert_called_once_with(group)

if __name__ == '__main__':
    unittest.main()
