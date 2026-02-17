# actions.py

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from livekit.agents import function_tool, RunContext

from ai_utils import normalize_visit_datetime_pst

PST = ZoneInfo("America/Los_Angeles")

logger = logging.getLogger("actions")

# -------------------------------------------------
# MULTIPLY
# -------------------------------------------------

@function_tool()
async def multiplica_numeros(
    context: RunContext,
    number1: int,
    number2: int,
) -> str:
    call_id = context.session.userdata.get("call_id")

    result = number1 * number2

    logger.info(
        "multiply_numbers call_id=%s n1=%s n2=%s result=%s",
        call_id,
        number1,
        number2,
        result,
    )

    return f"The product of {number1} and {number2} is {result}"


# -------------------------------------------------
# SCHEDULE VISIT
# -------------------------------------------------

@function_tool()
async def agendar_cita_disponibilidad(
    context: RunContext,
    name: str,
    visit_date: str,
    visit_time: str,
    purpose: str,
) -> str:

    session_data = context.session.userdata
    call_id = session_data.get("call_id")

    logger.info(
        "schedule_visit_raw call_id=%s name=%s date=%s time=%s purpose=%s",
        call_id,
        name,
        visit_date,
        visit_time,
        purpose,
    )

    # 🔹 Normalize via LLM
    normalized = await normalize_visit_datetime_pst(
        visit_date=visit_date,
        visit_time=visit_time,
    )

    if normalized.get("confidence") != "high":
        logger.info(
            "schedule_visit_low_confidence call_id=%s confidence=%s",
            call_id,
            normalized.get("confidence"),
        )
        return "No quedó clara la fecha u hora. ¿Me la puedes confirmar?"

    # 🔹 Simulated availability
    available = True

    if not available:
        return "Ese horario no está disponible. ¿Quieres intentar con otro?"

    confirmed_visit = {
        "name": name,
        "purpose": purpose,
        "visit_date": normalized["visit_date"],
        "visit_time": normalized["visit_time"],
    }

    session_data["confirmed_visit"] = confirmed_visit

    logger.info(
        "schedule_visit_confirmed call_id=%s date=%s time=%s",
        call_id,
        normalized["visit_date"],
        normalized["visit_time"],
    )

    return (
        f"Perfecto {name}. Tu visita quedó agendada para el "
        f"{normalized['visit_date']} a las {normalized['visit_time']}."
    )


# -------------------------------------------------
# COTIZAR EVENTO
# -------------------------------------------------

@function_tool()
async def cotizar_evento(
    context: RunContext,
    tipo_evento: str,
    fecha_tentativa: str,
    numero_invitados: int,
) -> str:

    call_id = context.session.userdata.get("call_id")

    logger.info(
        "cotizar_evento call_id=%s tipo=%s fecha=%s invitados=%s",
        call_id,
        tipo_evento,
        fecha_tentativa,
        numero_invitados,
    )

    base_price = 5000
    price_per_guest = 350

    cotizacion = base_price + (numero_invitados * price_per_guest)

    tipo = tipo_evento.lower()

    if tipo in {"boda", "wedding"}:
        cotizacion *= 1.2
    elif tipo in {"conferencia", "corporativo"}:
        cotizacion *= 1.1

    cotizacion = int(cotizacion)

    return (
        f"Para un {tipo_evento} con aproximadamente {numero_invitados} invitados, "
        f"la cotización estimada es de {cotizacion} MXN."
    )
