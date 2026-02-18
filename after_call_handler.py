import logging
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import csv

from ai_utils import summarize_transcript, transcript_to_single_line
from gsheet_utils import append_row_to_sheet

logger = logging.getLogger("after_call_handler")

PST = ZoneInfo("America/Los_Angeles")
BASE_DIR = Path(__file__).resolve().parent

CALLS_CSV = BASE_DIR / "calls_log.csv"
SCHEDULE_CSV = BASE_DIR / "scheduled_visits.csv"

CALL_HEADERS = [
    "created_at_pst",
    "call_started_at",
    "call_ended_at",
    "call_duration_seconds",
    "transcript",
    "summary",
    "call_id",
]

SCHEDULE_HEADERS = [
    "created_at_pst",
    "name",
    "purpose",
    "visit_date",
    "visit_time",
    "call_id",
]


def append_csv(path: Path, headers, row):
    try:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info("CSV appended: %s", path.name)
    except Exception:
        logger.exception("Failed writing CSV: %s", path.name)


async def handle_after_call(
    call_id,
    call_started_at,
    transcript,
    session_userdata,
):
    """
    Fully isolated, never raises.
    """

    try:
        logger.info("After-call handler started call_id=%s", call_id)

        call_ended = datetime.now(tz=PST)
        duration = int((call_ended - call_started_at).total_seconds())

        # ------------------------------
        # Summarization (safe)
        # ------------------------------
        summary = None
        if transcript:
            try:
                summary = await summarize_transcript(transcript)
            except Exception:
                logger.exception("Transcript summarization failed")

        call_row = {
            "created_at_pst": call_ended.strftime("%Y-%m-%d %H:%M:%S"),
            "call_started_at": call_started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "call_ended_at": call_ended.strftime("%Y-%m-%d %H:%M:%S"),
            "call_duration_seconds": duration,
            "transcript": transcript_to_single_line(transcript),
            "summary": summary,
            "call_id": call_id,
        }

        # ------------------------------
        # Local CSV
        # ------------------------------
        append_csv(CALLS_CSV, CALL_HEADERS, call_row)

        # ------------------------------
        # Google Sheets (run in thread)
        # ------------------------------
        try:
            await asyncio.to_thread(
                append_row_to_sheet,
                sheet_name="Llamadas",
                headers=CALL_HEADERS,
                row=call_row,
            )
            logger.info("Google Sheets append success (Llamadas)")
        except Exception:
            logger.exception("Google Sheets append failed (Llamadas)")

        # ------------------------------
        # Scheduled Visit
        # ------------------------------
        confirmed_visit = session_userdata.get("confirmed_visit")

        if confirmed_visit:
            visit_row = {
                "created_at_pst": call_row["created_at_pst"],
                "name": confirmed_visit["name"],
                "purpose": confirmed_visit["purpose"],
                "visit_date": confirmed_visit["visit_date"],
                "visit_time": confirmed_visit["visit_time"],
                "call_id": call_id,
            }

            append_csv(SCHEDULE_CSV, SCHEDULE_HEADERS, visit_row)

            try:
                await asyncio.to_thread(
                    append_row_to_sheet,
                    sheet_name="Citas",
                    headers=SCHEDULE_HEADERS,
                    row=visit_row,
                )
                logger.info("Google Sheets append success (Citas)")
            except Exception:
                logger.exception("Google Sheets append failed (Citas)")

        logger.info("After-call handler completed call_id=%s", call_id)

    except Exception:
        logger.exception("After-call handler crashed (should never happen)")
