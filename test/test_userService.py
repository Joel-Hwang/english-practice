import unittest
from unittest.mock import patch, AsyncMock
from fastapi import HTTPException
from service.userService import registerUser, login
from model.user import UserCreate, User
from datetime import datetime

class TestUserService(unittest.IsolatedAsyncioTestCase):

    @patch('repository.userRepository.findUserById', new_callable=AsyncMock)
    @patch('repository.userRepository.insertUser', new_callable=AsyncMock)
    @patch('bcrypt.hashpw')
    async def test_registerUser_success(self, mock_hashpw, mock_insertUser, mock_findUserById):
        mock_hashpw.return_value = b'hashed_password'
        mock_findUserById.return_value = None
        
        user_create = UserCreate(id='test@example.com', password='password123')
        await registerUser(None, user_create) # Passing None for db as it's mocked
        
        mock_insertUser.assert_called_once()

    @patch('repository.userRepository.findUserById', new_callable=AsyncMock)
    async def test_registerUser_duplicate_id(self, mock_findUserById):
        mock_findUserById.return_value = {"id": "test@example.com", "password": "password123"}
        
        user_create = UserCreate(id='test@example.com', password='password123')
        with self.assertRaises(HTTPException) as cm:
            await registerUser(None, user_create)
        
        self.assertEqual(cm.exception.status_code, 400)
        self.assertEqual(cm.exception.detail, 'Somebody already took the ID. Please use different one.')

    async def test_registerUser_missing_fields(self):
        with self.assertRaises(HTTPException) as cm:
            await registerUser(None, UserCreate(id='', password='password123'))
        self.assertEqual(cm.exception.status_code, 422)
        self.assertEqual(cm.exception.detail, 'ID and password are required.')

        with self.assertRaises(HTTPException) as cm:
            await registerUser(None, UserCreate(id='test@example.com', password=''))
        self.assertEqual(cm.exception.status_code, 422)
        self.assertEqual(cm.exception.detail, 'ID and password are required.')

    @patch('repository.userRepository.findUserById', new_callable=AsyncMock)
    @patch('bcrypt.checkpw')
    async def test_login_success(self, mock_checkpw, mock_findUserById):
        mock_checkpw.return_value = True
        mock_findUserById.return_value = {
            "id": "test@example.com", 
            "password": "hashed_password",
            "gender": "female",
            "status": "active",
            "createdAt": datetime.now()
        }
        
        login_data = UserCreate(id='test@example.com', password='password123')
        user_login = await login(None, login_data)
        
        self.assertEqual(user_login.id, 'test@example.com')

    @patch('repository.userRepository.findUserById', new_callable=AsyncMock)
    async def test_login_invalid_id(self, mock_findUserById):
        mock_findUserById.return_value = None
        
        with self.assertRaises(HTTPException) as cm:
            await login(None, UserCreate(id='nonexistent@example.com', password='password123'))
        
        self.assertEqual(cm.exception.status_code, 400)
        self.assertEqual(cm.exception.detail, "We can't find your ID. Please try with a different one.")

    @patch('repository.userRepository.findUserById', new_callable=AsyncMock)
    @patch('bcrypt.checkpw')
    async def test_login_wrong_password(self, mock_checkpw, mock_findUserById):
        mock_checkpw.return_value = False
        mock_findUserById.return_value = {
            "id": "test@example.com", 
            "password": "hashed_password",
            "gender": "female",
            "status": "active",
            "createdAt": datetime.now()
        }
        
        login_data = UserCreate(id='test@example.com', password='wrongpassword')
        with self.assertRaises(HTTPException) as cm:
            await login(None, login_data)
            
        self.assertEqual(cm.exception.status_code, 400)
        self.assertEqual(cm.exception.detail, "Please check your password again.")

    async def test_login_missing_fields(self):
        with self.assertRaises(HTTPException) as cm:
            await login(None, UserCreate(id='', password='password123'))
        self.assertEqual(cm.exception.status_code, 422)
        self.assertEqual(cm.exception.detail, 'ID and password are required.')

    @patch('repository.userRepository.findUserById', new_callable=AsyncMock)
    async def test_login_inactive_user(self, mock_findUserById):
        mock_findUserById.return_value = {
            "id": "test@example.com", 
            "password": "hashed_password",
            "gender": "female",
            "status": "inactive",
            "createdAt": datetime.now()
        }
        
        login_data = UserCreate(id='test@example.com', password='password123')
        with self.assertRaises(HTTPException) as cm:
            await login(None, login_data)
            
        self.assertEqual(cm.exception.status_code, 400)
        self.assertEqual(cm.exception.detail, "Sorry, you aren't approved yet.")

if __name__ == '__main__':
    unittest.main()
