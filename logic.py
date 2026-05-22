import os
import json
import re

from openai import OpenAI
from dotenv import load_dotenv

# =========================
# ENV
# =========================

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


# =========================
# STATE
# =========================

def create_empty_state():
    return {
        "room_type": None,
        "area": None,
        "budget": None,

        "current_condition": {
            "repair_age": None,
            "wall_condition": None,
            "floor_condition": None,
            "ceiling_condition": None,
            "needs_replacement": [],
            "problems": []
        },

        "preferences": {
            "colors": [],
            "materials": [],
            "likes_details": None,
            "likes_minimalism": None,
            "likes_bright": None,
            "likes_natural": None,
            "favorite_examples": []
        },

        "style": {
            "selected": None,
            "confidence": 0,
            "alternatives": []
        },

        "design_plan": {
            "layout": {"proposal": None, "approved": False},
            "colors": {"proposal": None, "approved": False},
            "furniture": {"proposal": None, "approved": False},
            "lighting": {"proposal": None, "approved": False},
            "storage": {"proposal": None, "approved": False}
        },

        "stage": "discovery"
    }


# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
Ты профессиональный AI-дизайнер интерьеров.

ТВОЯ РОЛЬ:
Ты не просто отвечаешь на вопросы.
Ты помогаешь человеку понять,
какой интерьер ему нужен.

ПРАВИЛА:

1. Общайся как дизайнер-консультант.

2. НЕ задавай сухие вопросы.

ПЛОХО:
"Какой стиль вам нравится?"

ХОРОШО:
"Вам ближе спокойные минималистичные
интерьеры или более декоративные
и выразительные?"

3. Если пользователь не знает стиль —
помоги определить его через предпочтения.

4. Постепенно согласовывай:
- планировку
- цвета
- мебель
- освещение
- хранение

5. НЕ переходи к следующему этапу,
пока пользователь не одобрит текущий.

6. Будь вовлекающим.
Помогай человеку визуализировать интерьер.

7. Всегда учитывай бюджет.

8. Узнавай текущее состояние помещения:
- как давно ремонт
- что нужно менять
- какие проблемы есть

9. Финальный текст должен быть
очень подробным и атмосферным.

10. Не пиши сухими списками.
Пиши как дизайнерскую концепцию.
"""


# =========================
# BASE LLM CALL
# =========================

def ask_llm(prompt, temperature=0.7):
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature
    )

    return response.choices[0].message.content


# =========================
# JSON HELPERS
# =========================

def clean_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        return match.group(0)

    return text


def merge_dict(old, new):
    for k, v in new.items():

        if isinstance(v, dict) and k in old and isinstance(old[k], dict):
            merge_dict(old[k], v)

        else:
            if v not in [None, "", [], {}]:
                old[k] = v


# =========================
# STATE UPDATE
# =========================

def update_state(user_message, state):
    prompt = f"""
Извлеки информацию из сообщения пользователя.

ТЕКУЩИЙ STATE:
{json.dumps(state, ensure_ascii=False)}

СООБЩЕНИЕ:
{user_message}

Верни ТОЛЬКО JSON.

Не добавляй текст.

Обновляй только те поля,
которые пользователь упомянул.
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Ты извлекаешь данные для state."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    content = clean_json(response.choices[0].message.content)

    try:
        new_state = json.loads(content)
        merge_dict(state, new_state)
        return state

    except Exception:
        print("JSON ERROR")
        print(content)
        return state


# =========================
# INTENT DETECTION
# =========================

def detect_intent(message):
    prompt = f"""
Определи intent пользователя.

Варианты:

- answer
- approval
- rejection
- uncertainty
- ask_examples
- change_direction

Сообщение:
{message}

Верни только intent.
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip().lower()


# =========================
# STAGES
# =========================

STAGES = [
    "discovery",
    "style_detection",
    "layout",
    "colors",
    "furniture",
    "lighting",
    "storage",
    "final"
]


def next_stage(state):
    current = state["stage"]
    idx = STAGES.index(current)

    if idx < len(STAGES) - 1:
        state["stage"] = STAGES[idx + 1]


# =========================
# STAGE GENERATORS
# =========================

def discovery_stage(state):
    prompt = f"""
Ты знакомишься с человеком
и его помещением.

ТЕКУЩИЙ STATE:
{json.dumps(state, ensure_ascii=False)}

Задай ОДИН следующий важный вопрос.

Тебе нужно узнать:
- тип комнаты
- площадь
- бюджет
- текущее состояние
- проблемы помещения
- что нравится человеку

Общайся тепло и естественно.
"""

    return ask_llm(prompt)


def style_stage(state):
    prompt = f"""
Помоги пользователю понять,
какой стиль ему подходит.

ТЕКУЩИЙ STATE:
{json.dumps(state, ensure_ascii=False)}

Не спрашивай напрямую стиль.

Вместо этого:
- выясняй вкусы
- предлагай варианты
- сравнивай атмосферы

Задай ОДИН вопрос.
"""

    return ask_llm(prompt)


def generate_section_proposal(state, section):
    prompt = f"""
Создай предложение дизайнера
для раздела: {section}

STATE:
{json.dumps(state, ensure_ascii=False)}

Пиши:
- подробно
- атмосферно
- как дизайнер

Не списком советов.

Человек должен визуализировать интерьер.

В конце спроси:
нравится ли ему направление.
"""

    return ask_llm(prompt)


def generate_final_design(state):
    prompt = f"""
Создай полноценную концепцию ремонта.

STATE:
{json.dumps(state, ensure_ascii=False)}

Структура:

# Планировка
# Цветовая концепция
# Мебель
# Освещение
# Хранение
# Атмосфера интерьера
# Практические советы

Пиши очень подробно.

Не сухими пунктами.

Пользователь должен буквально
представить интерьер.
"""

    return ask_llm(prompt, temperature=0.8)


# =========================
# MAIN CHAT LOGIC
# =========================

def chat(user_message, state):
    update_state(user_message, state)

    intent = detect_intent(user_message)
    stage = state["stage"]

    # DISCOVERY
    if stage == "discovery":

        required = [
            state["room_type"],
            state["area"],
            state["budget"]
        ]

        if all(required):
            next_stage(state)
            return style_stage(state)

        return discovery_stage(state)

    # STYLE DETECTION
    if stage == "style_detection":

        if state["style"]["selected"]:
            next_stage(state)
            return generate_section_proposal(state, "планировка")

        return style_stage(state)

    # SECTION UNIVERSAL LOGIC
    sections = {
        "layout": ("layout", "планировка", "цветовая концепция"),
        "colors": ("colors", "цветовая концепция", "мебель"),
        "furniture": ("furniture", "мебель", "освещение"),
        "lighting": ("lighting", "освещение", "хранение"),
        "storage": ("storage", "хранение", None)
    }

    if stage in sections:
        key, current_name, next_name = sections[stage]

        if not state["design_plan"][key]["proposal"]:
            proposal = generate_section_proposal(state, current_name)
            state["design_plan"][key]["proposal"] = proposal
            return proposal

        if intent == "approval":
            state["design_plan"][key]["approved"] = True
            next_stage(state)

            if next_name:
                return generate_section_proposal(state, next_name)
            else:
                return generate_final_design(state)

        return f"Что хотелось бы изменить в разделе: {current_name}?"

    # FINAL
    return generate_final_design(state)


# =========================
# RUN
# =========================

if __name__ == "__main__":

    state = create_empty_state()

    print("AI дизайнер запущен. Напишите 'exit' для выхода.\n")

    while True:

        msg = input("Вы: ")

        if msg.lower() in ["exit", "quit"]:
            break

        response = chat(msg, state)

        print("\nAI дизайнер:\n")
        print(response)
        print("\n")
