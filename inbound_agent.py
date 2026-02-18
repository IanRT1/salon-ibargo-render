import logging
import os
import secrets
import asyncio
import httpx
import tempfile
import json

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    AutoSubscribe,
    function_tool,
    RunContext,
)

from livekit.plugins import silero
from livekit.plugins import openai
from livekit.plugins import deepgram
from livekit.plugins.google import tts as google_tts
from livekit import api


# =====================================================
# CONFIG
# =====================================================

AUTOMATION_BASE_URL = "https://bandia-toolkit.onrender.com"

PST = ZoneInfo("America/Los_Angeles")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("inbound_agent")

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

INSTRUCTIONS_PATH = BASE_DIR / "instructions.txt"


# =====================================================
# GOOGLE SERVICE ACCOUNT BOOTSTRAP (REQUIRED FOR RENDER)
# =====================================================

service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

if service_account_json:
    # Production (Render)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(service_account_json.encode())
        tmp.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
else:
    # Local fallback
    local_creds = BASE_DIR / "service_account.json"

    if not local_creds.exists():
        raise RuntimeError(
            "No Google credentials found. "
            "Set GOOGLE_SERVICE_ACCOUNT_JSON or provide service_account.json locally."
        )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(local_creds)


# =====================================================
# UTILITIES
# =====================================================

def generate_call_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rand = secrets.token_hex(4)
    return f"call_{ts}_{rand}"


async def call_automation(endpoint: str, payload: dict):
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{AUTOMATION_BASE_URL}{endpoint}",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

def get_current_time_spanish_pst() -> str:
    now = datetime.now(tz=PST)

    dias = [
        "lunes", "martes", "miércoles", "jueves",
        "viernes", "sábado", "domingo"
    ]

    meses = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]

    dia_semana = dias[now.weekday()]
    dia = now.day
    mes = meses[now.month - 1]
    año = now.year

    # Convert to 12-hour format
    hora = now.hour
    minutos = now.minute

    hora_12 = hora % 12
    if hora_12 == 0:
        hora_12 = 12

    # Determine period in natural Spanish
    if 0 <= hora < 12:
        periodo = "de la mañana"
    elif 12 <= hora < 19:
        periodo = "de la tarde"
    else:
        periodo = "de la noche"

    return f"{dia_semana.capitalize()} {dia} de {mes}, {año}, {hora_12}:{minutos:02d} {periodo}"


# =====================================================
# FUNCTION TOOLS (FORWARDERS)
# =====================================================

@function_tool()
async def end_call(
    context: RunContext,
    reason: str,
) -> str:
    """
    Usa esta función únicamente cuando la conversación haya terminado de forma natural.

    Llama esta función cuando:
    - El cliente se despida.
    - El cliente confirme que no necesita nada más.
    - La conversación haya llegado claramente a su cierre.

    Antes de llamar esta función:
    - Despídete de manera natural.
    - No anuncies que vas a colgar.
    - No expliques que estás ejecutando una función.

    No la uses en medio de la conversación.
    No la uses si aún hay información pendiente.
    """

    logger.info("end_call triggered. reason=%s", reason)

    await context.session.say(
        "Gracias por llamar a Salon Ibargo. Que tengas excelente día.",
        allow_interruptions=False,
    )

    await asyncio.sleep(8)

    try:
        lkapi = api.LiveKitAPI()

        room_name = context.session.userdata.get("room_name")
        identity = context.session.userdata.get("participant_identity")

        await lkapi.room.remove_participant(
            api.RoomParticipantIdentity(
                room=room_name,
                identity=identity,
            )
        )

        logger.info(
            "Successfully hung up participant %s",
            context.session.participant.identity,
        )

    except Exception as e:
        logger.warning("Error while ending call: %s", e)

    return "Call ended."


@function_tool()
async def multiplica_numeros(
    context: RunContext,
    number1: int,
    number2: int,
) -> str:
    """
    Usa esta función solo cuando el cliente solicite explícitamente
    multiplicar dos números.

    No la uses como ejemplo.
    No la uses para cálculos internos.
    Solo úsala si el cliente pide directamente una multiplicación.
    """

    call_id = context.session.userdata.get("call_id")

    payload = {
        "call_id": call_id,
        "number1": number1,
        "number2": number2,
    }

    result = await call_automation("/salon_ibargo_multiplica_numeros", payload)
    return result["message"]


@function_tool()
async def agendar_cita_disponibilidad(
    context: RunContext,
    name: str,
    visit_date: str,
    visit_time: str,
    purpose: str,
) -> str:
    """
    Usa esta función únicamente cuando ya tengas confirmados:
    - Nombre del cliente
    - Fecha exacta
    - Hora exacta
    - Motivo de la visita

    Esta función verifica disponibilidad y confirma la cita.

    No la uses si falta algún dato.
    No la uses para preguntar disponibilidad general.
    Solo ejecútala cuando toda la información esté confirmada.
    """

    await context.session.say(
        "Espera un momento mientras verifico la disponibilidad para esa fecha y hora...",
        allow_interruptions=True,
    )

    call_id = context.session.userdata.get("call_id")

    payload = {
        "call_id": call_id,
        "name": name,
        "visit_date": visit_date,
        "visit_time": visit_time,
        "purpose": purpose,
    }

    result = await call_automation("/salon_ibargo_agendar_cita_disponibilidad", payload)

    if result.get("confirmed_visit"):
        context.session.userdata["confirmed_visit"] = result["confirmed_visit"]

    return result["message"]


@function_tool()
async def cotizar_evento(
    context: RunContext,
    tipo_evento: str,
    fecha_tentativa: str,
    numero_invitados: int,
) -> str:
    """
    Usa esta función únicamente cuando ya tengas confirmados:
    - Tipo de evento
    - Fecha tentativa
    - Número aproximado de invitados

    Esta función genera una cotización estimada.

    No la uses si falta alguno de los datos.
    No menciones precios antes de ejecutar esta función.
    """

    await context.session.say(
        "Un momento, por favor. Estoy preparando una cotización para tu evento...",
        allow_interruptions=True,
    )

    call_id = context.session.userdata.get("call_id")

    payload = {
        "call_id": call_id,
        "tipo_evento": tipo_evento,
        "fecha_tentativa": fecha_tentativa,
        "numero_invitados": numero_invitados,
    }

    result = await call_automation("/salon_ibargo_cotizar_evento", payload)
    return result["message"]


# =====================================================
# AGENT
# =====================================================

class Assistant(Agent):
    multiplica_numeros = multiplica_numeros
    agendar_cita_disponibilidad = agendar_cita_disponibilidad
    cotizar_evento = cotizar_evento
    end_call = end_call

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.5,
        min_speech_duration=0.15,
        min_silence_duration=0.3,
    )


# =====================================================
# ENTRYPOINT
# =====================================================

async def entrypoint(ctx: JobContext):

    call_id = generate_call_id()
    ctx.proc.userdata["call_id"] = call_id

    call_started_at = datetime.now(tz=PST)
    transcript: list[dict[str, str]] = []

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    ctx.proc.userdata["participant_identity"] = participant.identity
    ctx.proc.userdata["room_name"] = ctx.room.name

    with open(INSTRUCTIONS_PATH, "r", encoding="utf-8") as f:
        raw_instructions = f.read()

    current_time_str = get_current_time_spanish_pst()

    GENERAL_INSTRUCTIONS = raw_instructions.format(
        current_time=current_time_str
    )

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="es",
            punctuate=True,
            smart_format=True,
            interim_results=True,
        ),
        llm=openai.LLM(
            model="gpt-5.2",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        tts=google_tts.TTS(
            language="es-US",
            voice_name="es-US-Chirp3-HD-Achernar",
            model_name="chirp_3",
        ),
        vad=ctx.proc.userdata["vad"],
        userdata=ctx.proc.userdata,
    )

    def on_conversation_item(ev):
        text = "".join(
            part for part in ev.item.content if isinstance(part, str)
        ).strip()

        if text:
            transcript.append({"role": ev.item.role, "content": text})

    session.on("conversation_item_added", on_conversation_item)

    async def on_shutdown(reason: str):

        payload = {
            "call_id": call_id,
            "call_started_at": call_started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "call_ended_at": datetime.now(tz=PST).strftime("%Y-%m-%d %H:%M:%S"),
            "transcript": transcript,
            "confirmed_visit": ctx.proc.userdata.get("confirmed_visit"),
        }

        try:
            await call_automation("/salon_ibargo_after_call", payload)
            logger.info("After-call forwarded successfully")
        except Exception:
            logger.exception("After-call forwarding failed")

    ctx.add_shutdown_callback(on_shutdown)

    agent = Assistant(instructions=GENERAL_INSTRUCTIONS)

    await session.start(agent=agent, room=ctx.room)

    await session.say(
        "Hola, soy Mia. ¿En qué puedo ayudarte?",
        allow_interruptions=True,
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="inbound_agent",
            prewarm_fnc=prewarm,
        )
    )
