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
# FUNCTION TOOLS (FORWARDERS)
# =====================================================

@function_tool()
async def end_call(
    context: RunContext,
    reason: str,
) -> str:
    """
    Usa esta función únicamente cuando la conversación haya terminado de forma natural.
    """

    logger.info("end_call triggered. reason=%s", reason)

    # Speak closing message
    await context.session.say(
        "Gracias por llamar a Salon Ibargo. Que tengas excelente día.",
        allow_interruptions=False,
    )

    # Small buffer
    #await asyncio.sleep(5)

    room_name = context.session.userdata.get("room_name")
    identity = context.session.userdata.get("participant_identity")

    lkapi = api.LiveKitAPI()

    try:
        await lkapi.room.remove_participant(
            api.RoomParticipantIdentity(
                room=room_name,
                identity=identity,
            )
        )

        logger.info("Successfully hung up participant %s", identity)

    except Exception as e:
        if "not_found" in str(e):
            logger.info("Participant already disconnected.")
        else:
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
        # Start API immediately
        api_task = asyncio.create_task(
            call_automation(
                "/salon_ibargo_agendar_cita_disponibilidad",
                payload,
            )
        )

        # Speak while backend runs
        await context.session.say(
            "Gracias por proporcionar los datos para agendar tu cita. "
            "Espera un momento mientras verifico tus datos para mayor precisión",
            allow_interruptions=False,
        )

        result = await asyncio.wait_for(api_task, timeout=20)

        if not isinstance(result, dict):
            logger.error("Invalid API response: %s", result)
            return "Lo siento, ocurrió un problema al verificar la disponibilidad."

        if result.get("confirmed_visit"):
            context.session.userdata["confirmed_visit"] = result["confirmed_visit"]

        message = result.get("message")

        if not message:
            logger.error("API response missing message: %s", result)
            return "Hubo un problema al confirmar la cita."

        return message

    except httpx.HTTPStatusError as e:
        # Backend returned 4xx / 5xx
        try:
            error_json = e.response.json()
            detail = error_json.get("detail")

            if detail:
                logger.info("API returned detail: %s", detail)
                return detail

        except Exception:
            logger.exception("Failed to parse error response")

        logger.warning("HTTP error from appointment API: %s", e)
        return "Lo siento, ocurrió un problema al verificar la disponibilidad."

    except asyncio.TimeoutError:
        logger.warning("Appointment API timed out")
        return (
            "Lo siento, el sistema está tardando más de lo esperado "
            "en verificar la disponibilidad."
        )

    except Exception:
        logger.exception("Unexpected error in agendar_cita_disponibilidad")
        return (
            "Lo siento, ocurrió un problema al verificar la disponibilidad."
        )

    finally:
        if api_task and not api_task.done():
            api_task.cancel()


# =====================================================
# AGENT
# =====================================================

class Assistant(Agent):
    agendar_cita_disponibilidad = agendar_cita_disponibilidad
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

    conversation_id  = generate_call_id()
    ctx.proc.userdata["conversation_id"] = conversation_id 

    call_started_at = datetime.now(tz=PST)
    transcript: list[dict[str, str]] = []

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    logger.info("Participant attributes: %s", participant.attributes)

    attrs = participant.attributes or {}

    caller_number = attrs.get("sip.phoneNumber")
    to_number = attrs.get("sip.trunkPhoneNumber")
    call_sid = attrs.get("sip.twilio.callSid") 

    ctx.proc.userdata["from_phone_number"] = caller_number
    ctx.proc.userdata["to_phone_number"] = to_number
    ctx.proc.userdata["call_sid"] = call_sid

    logger.info(
        "Call metadata | from=%s | to=%s",
        caller_number,
        to_number,
    )

    ctx.proc.userdata["participant_identity"] = participant.identity
    ctx.proc.userdata["room_name"] = ctx.room.name

    with open(INSTRUCTIONS_PATH, "r", encoding="utf-8") as f:
        raw_instructions = f.read()

    current_time_str = get_current_time_spanish_pst()

    class SafeDict(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    GENERAL_INSTRUCTIONS = raw_instructions.format_map(
        SafeDict(current_time=current_time_str)
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
        item = ev.item

        # Safely extract role
        role = getattr(item, "role", "unknown")

        # Safely extract content
        content = getattr(item, "content", None)
        if not content:
            return

        # Join only string parts (ignore tool metadata parts)
        text = "".join(
            part for part in content
            if isinstance(part, str)
        ).strip()

        if not text:
            return

        # Store transcript
        transcript.append({
            "role": role,
            "content": text
        })

        # Log cleanly
        logger.info(
            "SPEECH | role=%s | text=%s",
            role,
            text
        )

    session.on("conversation_item_added", on_conversation_item)

    async def on_shutdown(reason: str):

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
        
        logger.info("on_shutdown payload: %s", payload)

        try:
            await call_automation("/salon_ibargo_after_call", payload)
            logger.info("After-call forwarded successfully")
        except Exception:
            logger.exception("After-call forwarding failed")

    ctx.add_shutdown_callback(on_shutdown)

    agent = Assistant(instructions=GENERAL_INSTRUCTIONS)

    await session.start(agent=agent, room=ctx.room)

    await session.say(
        "Hola, soy Mia de salon de eventos Ibargo. ¿En qué puedo ayudarte?",
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
