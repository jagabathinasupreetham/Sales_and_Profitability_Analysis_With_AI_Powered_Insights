import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b")
SARVAM_BASE_URL = os.getenv(
    "SARVAM_BASE_URL",
    "https://api.sarvam.ai/v1/chat/completions",
)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
