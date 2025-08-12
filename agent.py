from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

# We will define our tool here later, but for now, we'll keep it simple
# to focus on the persona change.
# You will create a function like this for your measurement converter.
# def convert_measurements(value: float, unit_from: str, unit_to: str):
#     # ... your conversion logic goes here ...
#     return converted_value


class ChefRamsay(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an AI assistant persona of celebrity chef Gordon Ramsay. "
                "Your name is Chef Ramsay. You are a highly critical, impatient, "
                "and demanding mentor for home cooks. Your tone is often angry, "
                "sarcastic, and slightly frustrated, but your goal is to make the "
                "user a better cook. You use British slang and common Gordon Ramsay "
                "catchphrases. "
                "You will not tolerate mistakes or laziness from the user. "
                "When asked for a recipe or cooking advice, you will provide "
                "it in your characteristic tough-love style. "
                "Keep your responses direct and to the point."
            )
        )


async def entrypoint(ctx: agents.JobContext):
    # For a Gordon Ramsay persona, you might want to choose a voice that sounds
    # a bit more gruff or authoritative, if available.
    # 'c0c374aa-09be-42d9-9828-4d2d7df86962' is your current voice. 
    # You might consider finding one with a British accent for better effect,
    # or just stick with this one.
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en-US"), # Changed to 'en-US' for better English performance
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(model="sonic-2", voice="63ff761f-c1e8-414b-b969-d1833d1c870c"),
        vad=silero.VAD.load(),
        # For a single-language agent, you can use the simpler VAD without multilingual detection.
        # This will be more efficient.
        # If you want to use the multilingual, that is fine too, as it is robust.
        # turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=ChefRamsay(), # Changed the class name to reflect the new persona
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    # The initial greeting must also be in the Gordon Ramsay persona.
    # This sets the tone from the very beginning.
    await session.generate_reply(
        instructions=(
            "Greet the user in the persona of Chef Gordon Ramsay. "
            "Tell them you are here to make them a better cook and to "
            "make their meal prep less of a disaster. "
            "Be demanding and impatient, using phrases like 'don't waste my time'."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))