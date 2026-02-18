from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import csv

from ai_utils import summarize_transcript, transcript_to_single_line
from gsheet_utils import append_row_to_sheet

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
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

async def handle_after_call(
    call_id,
    call_started_at,
    transcript,
    session_userdata,
):

    call_ended = datetime.now(tz=PST)
    duration = int((call_ended - call_started_at).total_seconds())

    summary = None
    if transcript:
        summary = await summarize_transcript(transcript)

    call_row = {
        "created_at_pst": call_ended.strftime("%Y-%m-%d %H:%M:%S"),
        "call_started_at": call_started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "call_ended_at": call_ended.strftime("%Y-%m-%d %H:%M:%S"),
        "call_duration_seconds": duration,
        "transcript": transcript_to_single_line(transcript),
        "summary": summary,
        "call_id": call_id,
    }

    append_csv(CALLS_CSV, CALL_HEADERS, call_row)

    append_row_to_sheet(
        sheet_name="Llamadas",
        headers=CALL_HEADERS,
        row=call_row,
    )

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

        append_row_to_sheet(
            sheet_name="Citas",
            headers=SCHEDULE_HEADERS,
            row=visit_row,
        )
