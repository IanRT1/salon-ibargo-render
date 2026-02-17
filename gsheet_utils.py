from typing import Dict, List
from pathlib import Path
import logging
import os
import json
import tempfile

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

logger = logging.getLogger("gsheet_utils")

# ------------------------
# Config
# ------------------------

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SPREADSHEET_ID = "1fvo3qrZgvLiHUrjXgm3yxuwH3C5qgzaL43O8JJJp6OI"

BASE_DIR = Path(__file__).resolve().parent
LOCAL_SERVICE_ACCOUNT_FILE = BASE_DIR / "service_account.json"

# ------------------------
# Credential Loader
# ------------------------

def _load_credentials():
    """
    Loads credentials from:
    1) GOOGLE_SERVICE_ACCOUNT_JSON env var (Render production)
    2) local service_account.json file (local dev fallback)
    """

    json_env = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

    # 🔹 Production: load from env variable
    if json_env:
        try:
            service_account_info = json.loads(json_env)
            return Credentials.from_service_account_info(
                service_account_info,
                scopes=SCOPES,
            )
        except Exception as e:
            raise RuntimeError(
                "Invalid GOOGLE_SERVICE_ACCOUNT_JSON environment variable"
            ) from e

    # 🔹 Local fallback: load from file
    if LOCAL_SERVICE_ACCOUNT_FILE.exists():
        return Credentials.from_service_account_file(
            LOCAL_SERVICE_ACCOUNT_FILE,
            scopes=SCOPES,
        )

    raise RuntimeError(
        "No Google credentials found. "
        "Set GOOGLE_SERVICE_ACCOUNT_JSON or provide service_account.json locally."
    )


# ------------------------
# Client (lazy)
# ------------------------

_service = None

def _get_sheets_service():
    global _service
    if _service is None:
        creds = _load_credentials()
        _service = build("sheets", "v4", credentials=creds)
    return _service


# ------------------------
# Public API
# ------------------------

def append_row_to_sheet(
    *,
    sheet_name: str,
    headers: List[str],
    row: Dict,
):
    """
    Generic, order-safe append.
    Assumes headers already exist in the sheet.
    """

    service = _get_sheets_service()

    values = [[row.get(h) for h in headers]]

    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{sheet_name}!A1",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body={"values": values},
    ).execute()

    logger.info("sheet_row_appended sheet=%s", sheet_name)


# ------------------------
# Manual Test
# ------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from datetime import datetime
    from zoneinfo import ZoneInfo

    PST = ZoneInfo("America/Los_Angeles")

    HEADERS = [
        "created_at_pst",
        "name",
        "purpose",
        "visit_date",
        "visit_time",
        "call_id",
    ]

    test_row = {
        "created_at_pst": datetime.now(tz=PST).isoformat(),
        "name": "Prueba Google Sheets",
        "purpose": "Test append desde gsheet_utils",
        "visit_date": "2026-01-20",
        "visit_time": "15:30",
        "call_id": "TEST-CALL-123",
    }

    logger.info("Appending test row to Google Sheet...")
    append_row_to_sheet(
        sheet_name="Citas",
        headers=HEADERS,
        row=test_row,
    )
    logger.info("Done.")
