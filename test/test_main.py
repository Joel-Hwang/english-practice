import pytest
import json
import openai
from main import extract_clean_json_strings

def test_valid_json_single():
    text = "{\"corrected\": It was just alright\", \"explanations\": [\"'okay'를 'alright'로 바꿈. 'alright'는 조금 더 자연스럽고 일상적인 표현임.\"], \"conversational_fluency_score\": \"Good\"}"
    result = extract_clean_json_strings(text)

    data = json.loads(result)
    assert len(result) == 1
    assert '"corrected": "It was just alright"' in result[0]


def test_text_to_speech():
    openai.api_key = "Your key"
    speech_response = openai.audio.speech.create(
        model="tts-1",  # 또는 tts-1-hd
        voice="nova",  # 선택: nova, shimmer, echo
        input="do you want to play a game?",
    )
    with open("gpt_voice.mp3", "wb") as f:
        f.write(speech_response.content)