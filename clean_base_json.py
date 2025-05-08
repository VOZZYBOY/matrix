import json
import os
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Слова-маркеры для удаления (без учета регистра)
GARBAGE_MARKERS = ["Сабина Ахмедова Салмановна","Селима Муталиева  Андарбековна","Юлия Пигуль Сергеевна","Юрий Бобров Александрович"]


FILE_PATH = "/home/erik/matrixai/base/medyumed.2023-04-24.json" 

def contains_garbage(item: dict, markers: list) -> bool:
    """
    Проверяет, содержит ли какой-либо строковый ключ в словаре один из маркеров.
    """
    if not isinstance(item, dict):
        return False
    
    for value in item.values():
        if isinstance(value, str):
            value_lower = value.lower()
            for marker in markers:
                if marker.lower() in value_lower:
                    logging.debug(f"Найдена мусорная строка '{marker}' в значении '{value}' элемента: {item.get('serviceId') or item.get('employeeId') or 'N/A'}")
                    return True
    return False

def clean_json_data(file_path: str, garbage_markers: list) -> bool:
    """
    Читает JSON-файл, удаляет элементы, содержащие мусорные маркеры,
    и перезаписывает файл.
    """
    
    # Проверка существования файла
    if not os.path.exists(file_path):
        logging.error(f"Файл не найден: {file_path}")
        return False

    # Создание бэкапа (рекомендуется)
    backup_file_path = file_path + ".backup"
    try:
        if os.path.exists(backup_file_path):
            logging.warning(f"Файл бэкапа {backup_file_path} уже существует. Перезапись бэкапа не производится для безопасности.")
        else:
            import shutil
            shutil.copy2(file_path, backup_file_path)
            logging.info(f"Создан бэкап: {backup_file_path}")
    except Exception as e:
        logging.error(f"Не удалось создать бэкап файла {file_path}: {e}")
        # Можно решить, продолжать ли без бэкапа или нет
        # return False 


    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Ошибка декодирования JSON из файла {file_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"Не удалось прочитать файл {file_path}: {e}")
        return False

    if not isinstance(data, list):
        logging.error(f"Ожидалось, что {file_path} будет содержать JSON-массив (список). Найдено: {type(data)}")
        return False

    original_count = len(data)
    logging.info(f"Исходное количество записей: {original_count}")

    cleaned_data = [item for item in data if not contains_garbage(item, garbage_markers)]
    
    removed_count = original_count - len(cleaned_data)

    if removed_count > 0:
        logging.info(f"Будет удалено записей: {removed_count}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2) 
            logging.info(f"Файл {file_path} успешно очищен и перезаписан. Новое количество записей: {len(cleaned_data)}")
        except Exception as e:
            logging.error(f"Не удалось записать очищенные данные в файл {file_path}: {e}")
            if os.path.exists(backup_file_path):
                try:
                    import shutil
                    shutil.copy2(backup_file_path, file_path)
                    logging.info(f"Данные восстановлены из бэкапа: {backup_file_path}")
                except Exception as backup_restore_error:
                    logging.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось восстановить данные из бэкапа! {backup_restore_error}")
            return False
    else:
        logging.info("Мусорных записей для удаления не найдено.")

    return True

if __name__ == "__main__":
    logging.info(f"Запуск очистки файла: {FILE_PATH}")
    logging.info(f"Мусорные маркеры: {GARBAGE_MARKERS}")
    
    success = clean_json_data(FILE_PATH, GARBAGE_MARKERS)
    
    if success:
        logging.info("Очистка завершена успешно.")
    else:
        logging.error("Очистка завершилась с ошибками.") 
