from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import List

class Question(BaseModel):
    group: str
    questions: List[str]
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
