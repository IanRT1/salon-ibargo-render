# ai_utils.py

from typing import List
import os
import logging
import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

import json
from datetime import datetime
from zoneinfo import ZoneInfo

PST = ZoneInfo("America/Los_Angeles")


# -------------------------------------------------
# Environment
# -------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in .env")

# -------------------------------------------------
# OpenAI client (OFFICIAL SDK)
# -------------------------------------------------

client = OpenAI(api_key=OPENAI_API_KEY)

SUM_MODEL = "gpt-5-nano"
STD_MODEL = "gpt-5-mini"

# -------------------------------------------------
# Logging (minimal)
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_utils")

# -------------------------------------------------
# Models
# -------------------------------------------------

class TranscriptItem(BaseModel):
    role: str
    content: str

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def transcript_to_single_line(transcript: list[dict]) -> str:
    """
    Expects:
    [
        {"role": "user" | "assistant", "content": "text"},
        ...
    ]
    """

    return " | ".join(
        f"{item['role'].upper()}: {item['content'].replace('\n', ' ').strip()}"
        for item in transcript
        if item.get("content")
    )

def extract_text(response) -> str:
    for item in response.output:
        # Only items that actually contain content
        if hasattr(item, "content") and item.content:
            for content in item.content:
                if getattr(content, "type", None) == "output_text":
                    return content.text.strip()
    return ""



# -------------------------------------------------
# Public API
# -------------------------------------------------

async def summarize_transcript(transcript: List[TranscriptItem]) -> str:
    """
    Summarize a call transcript into ONE short paragraph.
    """

    transcript_text = transcript_to_single_line(transcript)

    prompt = (
        "Resume la siguiente llamada telefónica en UN SOLO PÁRRAFO breve. "
        "No uses listas ni encabezados. "
        "Describe la intención del cliente y cómo terminó la llamada.\n\n"
        f"{transcript_text}"
    )

    logger.info(f"summarize_transcript: calling {SUM_MODEL}")

    response = client.responses.create(
        model=SUM_MODEL,
        input=prompt,
    )

    return extract_text(response)


async def normalize_visit_datetime_pst(
    visit_date: str,
    visit_time: str,
) -> dict:
    """
    Uses LLM to resolve relative date/time language.
    ai_utils owns reference time + timezone.
    Returns normalized components + confidence.
    """

    reference_dt = datetime.now(PST)

    prompt = f"""
        Resuelve fecha y hora a valores explícitos usando la referencia dada.

        REGLAS:
        - Devuelve SOLO JSON válido
        - No formatees para humanos
        - No inventes valores
        - Indica confidence si hay duda

        Referencia (ISO):
        {reference_dt.isoformat()}

        Entrada:
        fecha: "{visit_date}"
        hora: "{visit_time}"

        Formato EXACTO:
        {{
        "date": "YYYY-MM-DD",
        "time": "HH:MM",
        "confidence": "high|medium|low"
        }}
    """

    response = client.responses.create(
        model=STD_MODEL,
        input=prompt,
    )

    data = json.loads(extract_text(response))

    # HARD validation (still deterministic)
    datetime.strptime(data["date"], "%Y-%m-%d")
    datetime.strptime(data["time"], "%H:%M")

    dt = datetime.strptime(
        f'{data["date"]} {data["time"]}',
        "%Y-%m-%d %H:%M",
    ).replace(tzinfo=PST)

    return {
        "visit_date": dt.strftime("%Y-%m-%d"),     # ISO date
        "visit_time": dt.strftime("%H:%M"),        # 24h time
        "visit_datetime_iso": dt.isoformat(),      # full ISO
        "timezone": "America/Los_Angeles",
        "confidence": data.get("confidence", "low"),
    }


# -------------------------------------------------
# Self-test
# -------------------------------------------------

if __name__ == "__main__":

    async def _test():
        print("=== ai_utils self-test ===\n")

        normalized = await normalize_visit_datetime_pst(
            visit_date="2026-01-20",
            visit_time="7 pm",
            current_datetime="domingo, 19/01/2026 05:45PM PST",
        )

        print("Normalized Datetime:")
        print(normalized)
        print()

        transcript = [
            TranscriptItem(role="user", content="Hola, quiero saber si hay fechas disponibles."),
            TranscriptItem(role="assistant", content="Claro, ¿qué fecha tienes en mente?"),
            TranscriptItem(role="user", content="El 20 de enero por la tarde."),
            TranscriptItem(role="assistant", content="Perfecto, esa fecha está disponible."),
        ]

        summary = await summarize_transcript(transcript)

        print("Transcript summary:")
        print(summary)
        print("\n=== ai_utils self-test completed ===")

    asyncio.run(_test())
