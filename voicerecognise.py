from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = "AQVN2zTBAsQpVdzUXigKkKzPTA8q3uys6r_rR2de"  
configure_credentials(
    yandex_credentials=creds.YandexCredentials(
        api_key=API_KEY
    )
)

def recognize_audio_with_sdk(audio_file_path: str) -> str:
    try:
        model = model_repository.recognition_model()
        model.model = 'general'
        model.language = 'ru-RU'
        model.audio_processing_type = AudioProcessingType.Full

        result = model.transcribe_file(audio_file_path)
        if result:
            return result[0].normalized_text
        else:
            logger.error("Распознавание завершилось без результата.")
            return None
    except Exception as e:
        logger.error(f"Ошибка при распознавании аудио: {e}")
        return None
