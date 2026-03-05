import os
import secrets
import logging
import httpx

from datetime import datetime
from zoneinfo import ZoneInfo

# =====================================================
# LOGGING
# =====================================================

logger = logging.getLogger("automation_utils")


# =====================================================
# CONSTANTS
# =====================================================

PST = ZoneInfo("America/Los_Angeles")

AUTOMATION_BASE_URL = os.getenv(
    "AUTOMATION_BASE_URL",
    "https://bandia-toolkit-qwt3.onrender.com",
)

HTTP_TIMEOUT = 60


# =====================================================
# SHARED HTTP CLIENT
# =====================================================

_http_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _http_client

    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)

    return _http_client


# =====================================================
# CALL ID GENERATION
# =====================================================

def generate_call_id() -> str:
    """
    Generates a unique call identifier.

    Format:
        call_YYYYMMDDHHMMSS_<random_hex>

    Example:
        call_20260304200114_a3f9d2c1
    """
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rand = secrets.token_hex(4)
    return f"call_{ts}_{rand}"


# =====================================================
# AUTOMATION API CLIENT
# =====================================================

async def call_automation(endpoint: str, payload: dict):
    """
    Sends a POST request to the automation backend.

    Parameters
    ----------
    endpoint : str
        API path (ex: /salon_ibargo_after_call)

    payload : dict
        JSON payload

    Returns
    -------
    dict
        Parsed JSON response

    Raises
    ------
    httpx.HTTPStatusError
        If backend returns 4xx/5xx
    """

    url = f"{AUTOMATION_BASE_URL}{endpoint}"

    logger.info("Calling automation endpoint: %s", url)

    client = _get_client()

    response = await client.post(
        url,
        json=payload,
    )

    response.raise_for_status()

    try:
        return response.json()
    except Exception:
        logger.error("Automation response was not JSON: %s", response.text)
        raise


# =====================================================
# TIME FORMATTER
# =====================================================

def get_current_time_spanish_pst() -> str:
    """
    Returns the current time in PST formatted
    in natural Spanish.

    Example:
        Lunes 4 de marzo, 2026, 7:21 de la tarde
    """

    now = datetime.now(tz=PST)

    dias = [
        "lunes",
        "martes",
        "miércoles",
        "jueves",
        "viernes",
        "sábado",
        "domingo",
    ]

    meses = [
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ]

    dia_semana = dias[now.weekday()]
    dia = now.day
    mes = meses[now.month - 1]
    año = now.year

    hora = now.hour
    minutos = now.minute

    hora_12 = hora % 12
    if hora_12 == 0:
        hora_12 = 12

    if 0 <= hora < 12:
        periodo = "de la mañana"
    elif 12 <= hora < 19:
        periodo = "de la tarde"
    else:
        periodo = "de la noche"

    return f"{dia_semana.capitalize()} {dia} de {mes}, {año}, {hora_12}:{minutos:02d} {periodo}"


# =====================================================
# OPTIONAL CLEANUP
# =====================================================

async def close_http_client():
    """
    Gracefully closes the shared HTTP client.
    Useful if you ever add shutdown hooks.
    """
    global _http_client

    if _http_client:
        await _http_client.aclose()
        _http_client = None