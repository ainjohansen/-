import os
import re
import sys
import requests
import json 

# === НАСТРОЙКИ ===
# LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_URL = "https://localhost:8888/v1"
# В LM Studio имя модели можно оставить таким, скрипт подтянет загруженную в память модель автоматически
MODEL_NAME = "local-model"

INPUT_FILE = "Manifest-v49-rus.tex"   # Исходный файл
OUTPUT_FILE = "Manifest-v49-eng.tex"  # Файл для сохранения перевода

SYSTEM_PROMPT = (
    "You are a professional academic translator specializing in STEM. "
    "Your task is to translate LaTeX document text from Russian to formal, rigorous Academic English.\n\n"
    "CRITICAL RULES:\n"
    "1. Translate ONLY human-readable Russian text.\n"
    "2. DO NOT translate, alter, or omit any LaTeX commands, environments, or macros (e.g., \\section, \\begin{...}, \\end{...}, \\cite, \\ref, \\label).\n"
    "3. DO NOT translate or modify ANY math environments ($...$, $$...$$, \\begin{equation}, etc.), math symbols, or variables. Leave them exactly as they are.\n"
    "4. In tables, translate only the text cells. Do not touch table layout syntax, column definitions, or widths (e.g., p{4.6cm}).\n"
    "5. Maintain all braces {} and brackets [] perfectly intact.\n"
    "6. Output ONLY the translated LaTeX code. Do not include any explanations, greetings, or commentary."
)


def translate_chunk_stream(text_chunk, chunk_index, total_chunks):
    """Переводит фрагмент текста с гарантированным выводом в консоль."""
    if not text_chunk.strip():
        return ""

    print(f"\n" + "=" * 50)
    print(f" ПЕРЕВОД ФРАГМЕНТА {chunk_index}/{total_chunks}")
    print("=" * 50)
    print("Ожидание ответа от LM Studio... (если тут зависло, проверьте окно LM Studio)")

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text to translate:\n\n{text_chunk}"},
        ],
        "temperature": 0.1,
        "stream": True,  # Пробуем стриминг
    }

    translated_text = ""

    try:
        # Устанавливаем timeout, чтобы скрипт не висел вечно, если сервер упал
        response = requests.post(
            LM_STUDIO_URL, json=payload, headers=headers, stream=True, timeout=60
        )
        response.raise_for_status()

        first_token = True
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8").strip()

                # Убираем префикс API
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]

                    if data_str == "[DONE]":
                        break

                    try:
                        data_json = json.loads(data_str)
                        chunk_content = (
                            data_json.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )

                        if chunk_content:
                            if first_token:
                                print("\n--- ПОТОК ОТВЕТА НАЧАЛСЯ ---", flush=True)
                                first_token = False

                            # Прямой и принудительный вывод в консоль
                            print(chunk_content, end="", flush=True)
                            translated_text += chunk_content
                    except Exception:
                        continue

        # ПРОВЕРКА: Если стрим прошел, но текст пустой (LM Studio проигнорировал флаг stream)
        if not translated_text.strip():
            print(
                "\n⚠️ Стриминг не отдал текст. Пробую обычный запрос без стрима..."
            )
            payload["stream"] = False
            normal_resp = requests.post(
                LM_STUDIO_URL, json=payload, headers=headers, timeout=120
            )
            normal_resp.raise_for_status()
            res_json = normal_resp.json()
            translated_text = (
                res_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            print(translated_text, flush=True)

        print("\n\n✅ [Фрагмент успешно записан в файл]")
        return translated_text

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print(
            "Сохраняем оригинальный текст этого фрагмента, чтобы не потерять структуру."
        )
        return text_chunk

    except Exception as e:
        print(f"\n❌ Ошибка при связи с LM Studio: {e}")
        print("Записываем оригинальный текст фрагмента во избежание потери данных.")
        return text_chunk

    except Exception as e:
        print(f"\n❌ Ошибка при связи с LM Studio: {e}")
        print("Записываем оригинальный текст фрагмента во избежании потери данных.")
        return text_chunk


def split_by_latex_structure(file_path):
    """Разбивает текст по тегам структурных элементов (главы, секции)."""
    if not os.path.exists(file_path):
        print(f"❌ Ошибка: Файл {file_path} не найден в текущей папке!")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Ищем разделители: \chapter, \section, \subsection
    # Дополнительно можно разбивать по большим таблицам \begin{longtblr}, если секции слишком гигантские
    pattern = r"(\\chapter\*?\{.*?\}|\\section\*?\{.*?\}|\\subsection\*?\{.*?\})"
    parts = re.split(pattern, content)

    chunks = []
    current_chunk = ""

    for part in parts:
        # Если это заголовок секции, закрываем старый чанк и начинаем новый с заголовка
        if (
            part.startswith("\\chapter")
            or part.startswith("\\section")
            or part.startswith("\\subsection")
        ):
            if current_chunk.strip():
                chunks.append(current_chunk)
            current_chunk = part
        else:
            current_chunk += part

    if current_chunk.strip():
        chunks.append(current_chunk)

    return chunks


def main():
    print("--- ЗАПУСК СИСТЕМЫ ЛОКАЛЬНОГО ПЕРЕВОДА LATEX ---")
    print(f"Анализ файла: {INPUT_FILE}...")

    chunks = split_by_latex_structure(INPUT_FILE)
    total = len(chunks)
    print(f"Обнаружено фрагментов (глав/секций): {total}")

    # Очищаем или создаем чистый итоговый файл перед началом работы
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("")

    # Перевод по частям
    for idx, chunk in enumerate(chunks, 1):
        # Получаем перевод (текст будет печататься на экран на лету)
        translated_chunk = translate_chunk_stream(chunk, idx, total)

        # СРАЗУ сохраняем результат в файл на диск
        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
            out_f.write(translated_chunk + "\n\n")

    print("\n" + "=" * 50)
    print(f"🎉 ПЕРЕВОД ЗАВЕРШЕН! Результат сохранен в: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()

