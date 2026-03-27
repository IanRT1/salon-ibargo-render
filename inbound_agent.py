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
from livekit.protocol import sip as proto_sip

from utils import (
    generate_call_id,
    call_automation,
    get_current_time_spanish_pst,
    PST,
)


# =====================================================
# CONFIG
# =====================================================


logger = logging.getLogger("inbound_agent")
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

INSTRUCTIONS_PATH = BASE_DIR / "instructions.txt"

# --- NOTA: Estos valores ahora se gestionan en el Toolkit (main.py) ---
# FORWARD_NUMBER = "+526865102851"  
# BUSINESS_HOURS_START = 10   
# BUSINESS_HOURS_END   = 17   
# FORWARD_TIMEOUT_SECS = 30


# =====================================================
# GOOGLE SERVICE ACCOUNT BOOTSTRAP
# =====================================================

service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

if service_account_json:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(service_account_json.encode())
        tmp.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
else:
    local_creds = BASE_DIR / "service_account.json"
    if not local_creds.exists():
        raise RuntimeError("No Google credentials found.")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(local_creds)


# =====================================================
# FUNCTION TOOLS
# =====================================================

@function_tool()
async def end_call(context: RunContext, reason: str) -> str:
    """Usa esta función únicamente cuando la conversación haya terminado."""
    logger.info("end_call triggered. reason=%s", reason)
    await context.session.say(
        "Gracias por llamar a Salon Ibargo. Que tengas excelente día.",
        allow_interruptions=False,
    )
    room_name = context.session.userdata.get("room_name")
    identity = context.session.userdata.get("participant_identity")
    lkapi = api.LiveKitAPI()
    try:
        await lkapi.room.remove_participant(
            api.RoomParticipantIdentity(room=room_name, identity=identity)
        )
    except Exception as e:
        logger.warning("Error while ending call: %s", e)
    finally:
        await lkapi.aclose()
    return "Call ended."


@function_tool()
async def agendar_cita_disponibilidad(
    context: RunContext,
    name: str,
    visit_date: str,
    visit_time: str,
    purpose: str,
) -> str:
    """Verifica disponibilidad y confirma la cita."""
    call_id = context.session.userdata.get("conversation_id")
    payload = {
        "conversation_id": call_id,
        "channel": "voice",
        "name": name,
        "visit_date": visit_date,
        "visit_time": visit_time,
        "purpose": purpose,
    }
    api_task = None
    try:
        api_task = asyncio.create_task(
            call_automation("/salon_ibargo_agendar_cita_disponibilidad", payload)
        )
        await context.session.say(
            "Gracias. Espera un momento mientras verifico la disponibilidad.",
            allow_interruptions=False,
        )
        result = await asyncio.wait_for(api_task, timeout=20)
        if result.get("confirmed_visit"):
            context.session.userdata["confirmed_visit"] = result["confirmed_visit"]
        return result.get("message", "Hubo un problema al confirmar.")
    except Exception:
        logger.exception("Error in agendar_cita_disponibilidad")
        return "Ocurrió un problema al verificar la disponibilidad."
    finally:
        if api_task and not api_task.done():
            api_task.cancel()

# =====================================================
# TRANSFER (COMENTADO - AHORA EN EL TOOLKIT / TWILIO)
# =====================================================
# def is_business_hours() -> bool:
#     now = datetime.now(tz=PST)
#     return BUSINESS_HOURS_START <= now.hour < BUSINESS_HOURS_END

# async def try_conference_forward(ctx: JobContext, room_name: str) -> bool:
#     ... (toda la lógica de transferencia manual queda obsoleta aquí) ...


# =====================================================
# AGENT CONFIG
# =====================================================

class Assistant(Agent):
    agendar_cita_disponibilidad = agendar_cita_disponibilidad
    end_call = end_call

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(activation_threshold=0.5)


# =====================================================
# ENTRYPOINT
# =====================================================

async def entrypoint(ctx: JobContext):

    conversation_id = generate_call_id()
    ctx.proc.userdata["conversation_id"] = conversation_id
    watchdog_task = None
    transcript: list[dict[str, str]] = []

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    call_started_at = datetime.now(tz=PST)

    try:
        participant = await ctx.wait_for_participant()
    except RuntimeError:
        return

    # Metadata
    attrs = participant.attributes or {}
    ctx.proc.userdata["from_phone_number"] = attrs.get("sip.phoneNumber")
    ctx.proc.userdata["to_phone_number"] = attrs.get("sip.trunkPhoneNumber")
    ctx.proc.userdata["call_sid"] = attrs.get("sip.twilio.callSid")
    ctx.proc.userdata["participant_identity"] = participant.identity
    ctx.proc.userdata["room_name"] = ctx.room.name

    # Cargar Instrucciones
    try:
        with open(INSTRUCTIONS_PATH, "r", encoding="utf-8") as f:
            raw_instructions = f.read()
    except OSError:
        return

    instructions = raw_instructions.format(current_time=get_current_time_spanish_pst())

    # Configurar Sesión
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="es"),
        llm=openai.LLM(model="gpt-4o", api_key=os.environ.get("OPENAI_API_KEY")), # Ajustado a modelo real
        tts=google_tts.TTS(voice_name="es-US-Chirp3-HD-Achernar"),
        vad=ctx.proc.userdata["vad"],
        userdata=ctx.proc.userdata,
    )

    def on_conversation_item(ev):
        item = ev.item
        if hasattr(item, "content") and item.content:
            text = "".join(part for part in item.content if isinstance(part, str))
            if text.strip():
                transcript.append({"role": item.role, "content": text})

    session.on("conversation_item_added", on_conversation_item)

    async def on_shutdown(reason: str):
        if watchdog_task: watchdog_task.cancel()
        payload = {
            "conversation_id": conversation_id,
            "channel": "voice",
            "from_phone_number": ctx.proc.userdata.get("from_phone_number"),
            "to_phone_number": ctx.proc.userdata.get("to_phone_number"),
            "conversation_started_at": call_started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "conversation_ended_at": datetime.now(tz=PST).strftime("%Y-%m-%d %H:%M:%S"),
            "call_sid": ctx.proc.userdata.get("call_sid"),
            "transcript": transcript,
            "confirmed_visit": ctx.proc.userdata.get("confirmed_visit"),
        }
        await call_automation("/salon_ibargo_after_call", payload)

    ctx.add_shutdown_callback(on_shutdown)

    # ── LOGICA DE TRANSFERENCIA ANTERIOR (COMENTADA) ─────────────────────────
    # if is_business_hours():
    #     forwarded = await try_conference_forward(ctx, ctx.room.name)
    # else:
    #     forwarded = False
    # if forwarded: return 
    # ─────────────────────────────────────────────────────────────────────────

    # Iniciar Agente (Si la llamada llegó aquí es porque Twilio ya decidió que la IA conteste)
    agent = Assistant(instructions=instructions)
    await session.start(agent=agent, room=ctx.room)
    watchdog_task = asyncio.create_task(enforce_max_call_duration(session))

    await session.say(
        "Hola, soy Mia de salon de eventos Ibargo. ¿En qué puedo ayudarte?",
        allow_interruptions=True,
    )

async def enforce_max_call_duration(session: AgentSession):
    MAX_CALL_SECONDS = int(os.getenv("MAX_CALL_SECONDS", 600))
    try:
        await asyncio.sleep(MAX_CALL_SECONDS)
        await session.say("La llamada ha alcanzado el tiempo máximo. Gracias por llamar.")
        room_name = session.userdata.get("room_name")
        identity = session.userdata.get("participant_identity")
        lkapi = api.LiveKitAPI()
        try:
            await lkapi.room.remove_participant(api.RoomParticipantIdentity(room=room_name, identity=identity))
        finally:
            await lkapi.aclose()
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="inbound_agent", prewarm_fnc=prewarm))
