import logging
import os
import secrets
import tempfile

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
)

from livekit.plugins import silero
from livekit.plugins import openai
from livekit.plugins import deepgram
from livekit.plugins.google import tts as google_tts
from livekit.agents import AutoSubscribe


from actions import (
    multiplica_numeros,
    agendar_cita_disponibilidad,
    cotizar_evento,
)

from after_call_handler import handle_after_call


# -------------------------------------------------
# Logging
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("inbound_agent")


# -------------------------------------------------
# Environment & Credentials
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
if not service_account_json:
    raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")

with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
    tmp.write(service_account_json.encode())
    tmp.flush()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name


# -------------------------------------------------
# Timezone
# -------------------------------------------------

PST = ZoneInfo("America/Los_Angeles")


# -------------------------------------------------
# Instructions loader
# -------------------------------------------------

class SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def load_instructions(variables: dict | None = None) -> str:
    instructions_path = BASE_DIR / "instructions.txt"
    template = instructions_path.read_text(encoding="utf-8")

    if not variables:
        return template

    return template.format_map(SafeFormatDict(variables))


def generate_call_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rand = secrets.token_hex(4)
    return f"call_{ts}_{rand}"


# -------------------------------------------------
# Agent Definition
# -------------------------------------------------

class Assistant(Agent):
    multiplica_numeros = multiplica_numeros
    agendar_cita_disponibilidad = agendar_cita_disponibilidad
    cotizar_evento = cotizar_evento


# -------------------------------------------------
# Prewarm
# -------------------------------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.75,
        min_speech_duration=0.2,
        min_silence_duration=0.5,
    )


# -------------------------------------------------
# Entrypoint (Inbound)
# -------------------------------------------------

async def entrypoint(ctx: JobContext):
    logger.info("INBOUND ENTRYPOINT TRIGGERED")

    call_id = generate_call_id()
    ctx.proc.userdata["call_id"] = call_id

    call_started_at = datetime.now(tz=PST)

    transcript: list[dict[str, str]] = []

    # Connect and subscribe to audio
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 🔥 WAIT FOR THE SIP PARTICIPANT
    participant = await ctx.wait_for_participant()
    logger.info(f"SIP participant joined: {participant.identity}")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(
            model="gpt-5.2",
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        tts=google_tts.TTS(
            language="es-US",
            voice_name="es-US-Chirp3-HD-Achernar",
            model_name="chirp_3",
            speaking_rate=1.05,
            sample_rate=24000,
            pitch=0.0,
        ),
        vad=ctx.proc.userdata["vad"],
        userdata=ctx.proc.userdata,
        preemptive_generation=True,
    )

    # Transcript capture (YOUR exact style)
    def on_conversation_item(ev):
        text = "".join(
            part for part in ev.item.content if isinstance(part, str)
        ).strip()

        if text:
            transcript.append({"role": ev.item.role, "content": text})

    session.on("conversation_item_added", on_conversation_item)

    # Shutdown handler
    async def on_shutdown(reason: str):
        await handle_after_call(
            call_id=call_id,
            call_started_at=call_started_at,
            transcript=transcript,
            session_userdata=ctx.proc.userdata,
        )

    ctx.add_shutdown_callback(on_shutdown)

    # Start agent session
    await session.start(
        agent=Assistant(load_instructions()),
        room=ctx.room,
    )

    # ✅ MUST await say()
    await session.say(
        "Hola, soy Mia. Estás llamando al salón Ibargo. ¿En qué puedo ayudarte?",
        allow_interruptions=True,
    )


# -------------------------------------------------
# Entrypoint
# -------------------------------------------------

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="inbound_agent",
            prewarm_fnc=prewarm,
        )
    )
