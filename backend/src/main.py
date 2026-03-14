from src.audio.io import AudioIO
from src.vad.silero import SileroVAD
from src.asr.whisper import ASRModel
from src.llm.client import LLMModel
from src.tts.piper import TTSModel
from src.core.pipeline import VoicePipeline

def main() -> None:
    audio_io = AudioIO(sample_rate=16000, chunk_duration_ms=32)
    vad = SileroVAD()
    asr = ASRModel()
    llm = LLMModel()
    
    tts_models = {
        "CustomerCare": TTSModel(model_name="en_GB-alba-medium"),
        "Shopper":      TTSModel(model_name="en_US-bryce-medium"),
        "OrderOps":     TTSModel(model_name="en_US-hfc_female-medium")
    }
    
    pipeline = VoicePipeline(
        audio_io=audio_io,
        vad=vad,
        asr=asr,
        llm=llm,
        tts_models=tts_models
    )
    pipeline.start()

if __name__ == "__main__":
    main()
