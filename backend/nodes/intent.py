import json
import os
import re

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")
MAX_CLARIFY_ATTEMPTS = 3

SUPPORTED_PROCEDURES = [
    "angioplasty", "appendectomy", "arthroscopy", "bypass_cabg",
    "c_section", "cataract", "colonoscopy", "ct_scan", "dialysis_single",
    "ecg_echo", "endoscopy", "gallbladder_surgery", "hernia_repair",
    "hip_replacement", "hysterectomy", "kidney_stone_removal",
    "knee_replacement", "lasik", "mri_scan", "normal_delivery",
]

INTENT_PROMPT = """
You are MedPath's clinical intake AI for hospital navigation in India.
Use clinical reasoning to understand the user's symptoms and decide whether to ask a follow-up or show hospitals.

Important boundaries:
- First behave like a careful clinical intake: collect the key missing details before showing hospitals unless there are emergency warning signs.
- Use conversation history to avoid asking the same follow-up twice. If a topic was already requested, ask about a different missing topic or proceed.
- Ask about location, duration, severity, character of pain, associated symptoms, red flags, and relevant medical context when those details are missing.
- If symptoms do not cleanly match the hospital database, still reason clinically with Gemini, list probable clinical signals, and choose the closest diagnostic search route only when ready.
- If the user describes sudden right lower abdominal pain, especially with nausea, fever, loss of appetite, or worsening with motion, flag appendicitis risk without offering a formal diagnosis.
- You are not diagnosing the user. You are identifying clinical signals for care navigation.
- Do not invent facts not present in the conversation.
- Ask at most one focused clarification question in this turn.
- The app has already asked {clarify_attempts} clarification question(s). If this is {max_clarify_attempts} or more, do not ask another question. Choose the closest hospital-search route now.
- If there are emergency warning signs, mark emergency and make the route ready immediately.
- If details are limited but enough to search after the clarification budget, choose the best supported procedure or diagnostic route.

Supported hospital-search procedures. The "procedure" field must be exactly one of these or null:
{procedures}

Common routing examples, not rigid rules:
- unclear/general diagnostic concern: ct_scan
- cardiac-type chest symptoms or cardiac red flags: ecg_echo or angioplasty depending severity
- headache/neurologic symptoms needing imaging: mri_scan or ct_scan depending urgency
- urinary/flank/kidney-stone symptoms: kidney_stone_removal
- upper digestive symptoms: endoscopy
- lower bowel/bleeding concern: colonoscopy
- pregnancy/delivery: normal_delivery or c_section
- joint injury or joint evaluation: arthroscopy; severe knee/hip replacement need: knee_replacement or hip_replacement

Clarification examples:
- Vague symptoms: ask what symptom is most concerning, where it is, and when it started.
- Pain with location known: ask severity/character plus red flags, not the same location question again.
- Right lower abdominal pain: ask about fever, nausea/vomiting, appetite loss, worsening with movement, and duration if not already asked.
- Chest pain or breathing difficulty: do not delay for clarification; mark emergency.

Return only valid JSON with exactly these fields:
{{
  "procedure": "<supported procedure or null>",
  "city": "<city name, using profile city if not mentioned>",
  "budget": <integer INR or null>,
  "deadline_days": <integer or null>,
  "is_emergency": <boolean>,
  "ambiguity_score": <float 0.0-1.0>,
  "clarifying_question": "<natural, concise question or null>",
  "possible_causes": ["<2-3 concise clinical signals, not diagnoses>"],
  "icd10_code": "<ICD-10 hint or null>",
  "symptom_summary": "<one short natural sentence>",
  "recommendation_ready": <boolean>,
  "emergency_confidence": <float 0.0-1.0>,
  "follow_up_answers": {{
    "pain_type": "<if mentioned, else null>",
    "pain_location": "<if mentioned, else null>",
    "duration": "<if mentioned, else null>",
    "additional_symptoms": ["<only symptoms the user mentioned>"]
  }}
}}

Writing standards for clarifying_question:
- Sound like a calm clinician, not a form.
- Keep it to one sentence.
- Do not list more than 4 details to answer.
- Avoid awkward phrases such as "kindly", "provide the same", or "please elaborate your concern".

USER PROFILE:
- Name: {name}
- Age: {age}
- City: {city}
- Comorbidities: {comorbidities}

CONVERSATION HISTORY:
{history}

CURRENT USER MESSAGE:
{user_input}
"""


def _history_text(history: list) -> str:
    if not history:
        return "None. This is the first message."

    lines = []
    for turn in history[-6:]:
        if turn.get("user"):
            lines.append(f"User: {turn.get('user')}")
        if turn.get("assistant"):
            lines.append(f"MedPath: {turn.get('assistant')}")
    return "\n".join(lines) or "None."


def _json_from_model(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
      parts = raw.split("```")
      raw = parts[1] if len(parts) > 1 else raw
      raw = raw[4:] if raw.strip().startswith("json") else raw
    return json.loads(raw.strip())


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", (value or "").strip().lower())


def _safe_list(value) -> list:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:3]
    return []


def _history_assistant_text(state: dict) -> str:
    return _normalize_text(
        " ".join(
            turn.get("assistant", "")
            for turn in state.get("conversation_history", [])
            if turn.get("assistant")
        )
    )


def _combined_user_text(state: dict) -> str:
    parts = [state.get("user_input", "")]
    for turn in state.get("conversation_history", [])[-4:]:
        parts.append(turn.get("user", ""))
    return _normalize_text(" ".join(parts))


def _question_was_asked(question: str | None, state: dict) -> bool:
    if not question:
        return False

    question_text = _normalize_text(question)
    history_text = _history_assistant_text(state)
    if question_text and question_text in history_text:
        return True

    question_topics = {
        "duration": ("how long", "when did", "started", "start", "duration"),
        "location": ("where", "which part", "exactly", "location"),
        "severity": ("severity", "scale", "mild", "moderate", "severe"),
        "associated": ("fever", "nausea", "vomiting", "breathing", "dizziness"),
        "movement": ("movement", "moving", "walking", "coughing"),
    }

    for markers in question_topics.values():
        if any(marker in question_text for marker in markers) and any(
            marker in history_text for marker in markers
        ):
            return True

    return False


def _extract_budget(text: str) -> int | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lac|l)", text)
    if match:
        return int(float(match.group(1)) * 100000)

    match = re.search(r"(?:under|budget|below|upto|up to)\s*(\d{4,8})", text)
    return int(match.group(1)) if match else None


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _fallback_question(state: dict, procedure: str | None = None) -> str | None:
    text = _combined_user_text(state)
    asked = _history_assistant_text(state)

    def not_asked(*markers: str) -> bool:
        return not any(marker in asked for marker in markers)

    symptom_terms = (
        "pain", "fever", "vomit", "nausea", "dizzy", "breath", "cough",
        "headache", "abdomen", "abdominal", "stomach", "chest", "weak",
        "injury", "bleeding", "vision", "urine", "stool", "swelling",
    )
    if not _has_any(text, symptom_terms):
        if not_asked("what symptoms", "what symptom", "experiencing"):
            return "What symptoms are you having, and how long has this been going on?"
        return "What is the main symptom bothering you right now?"

    abdomen_terms = ("abdomen", "abdominal", "stomach", "belly", "appendix")
    if _has_any(text, abdomen_terms):
        if not_asked("fever", "nausea", "vomiting", "appetite", "movement"):
            return (
                "Do you have fever, nausea, vomiting, loss of appetite, "
                "or pain that worsens when you move?"
            )
        if not_asked("severity", "scale", "severe"):
            return "How severe is the abdominal pain on a 1 to 10 scale, and is it constant or coming in waves?"
        if not_asked("urine", "stool", "diarrhea", "constipation"):
            return "Have you noticed any diarrhea, constipation, burning urination, or blood in stool or urine?"
        return None

    if "chest" in text and not_asked("breathing", "left arm", "sweating", "dizziness"):
        return "Are you having trouble breathing, sweating, dizziness, or pain spreading to your arm, jaw, or back?"

    if "headache" in text and not_asked("vision", "weakness", "vomiting", "sudden"):
        return "Did the headache start suddenly, and do you have vomiting, vision changes, weakness, or confusion?"

    if "pain" in text:
        if not_asked("where", "which part", "location"):
            return "Where exactly is the pain, and does it spread anywhere else?"
        if not_asked("severity", "scale", "mild", "moderate", "severe"):
            return "How severe is the pain on a 1 to 10 scale, and what does it feel like?"

    if not_asked("how long", "when did", "started"):
        return "When did this start, and has it been getting better, worse, or staying the same?"

    if not_asked("fever", "vomiting", "breathing", "dizziness"):
        return "Do you have any other symptoms like fever, vomiting, breathing trouble, dizziness, or weakness?"

    return None


def _fallback_route(state: dict) -> dict:
    text = _combined_user_text(state)

    emergency_terms = ("breath", "breathing", "left arm", "severe chest", "unconscious")
    if "chest" in text and any(term in text for term in emergency_terms):
        return {
            "procedure": "ecg_echo",
            "is_emergency": True,
            "ambiguity_score": 0.2,
            "possible_causes": ["Cardiac warning signs", "Breathing difficulty"],
            "icd10_code": "R07.9",
            "recommendation_ready": True,
            "emergency_confidence": 0.9,
        }

    if "knee replacement" in text:
        return {
            "procedure": "knee_replacement",
            "possible_causes": ["Knee joint pain", "Mobility limitation"],
        }

    if "cataract" in text:
        return {
            "procedure": "cataract",
            "possible_causes": ["Vision clouding", "Eye procedure need"],
        }

    if any(term in text for term in ("abdomen", "abdominal", "stomach", "belly")):
        procedure = "ct_scan"
        causes = ["Abdominal pain", "Digestive or surgical evaluation needed"]
        icd10_code = "R10.9"
        if (
            "bottom right" in text
            or "lower right" in text
            or "right lower" in text
            or ("right" in text and "lower" in text)
            or "appendix" in text
            or "appendicitis" in text
        ):
            procedure = "appendectomy"
            causes = ["Right lower abdominal pain", "Appendicitis warning pattern"]

        return {
            "procedure": procedure,
            "possible_causes": causes,
            "icd10_code": icd10_code,
            "clarifying_question": _fallback_question(state, procedure),
        }

    if any(term in text for term in ("mri", "headache", "migraine", "seizure")):
        return {
            "procedure": "mri_scan",
            "possible_causes": ["Neurological symptoms", "Imaging evaluation need"],
        }

    return {}


def _normalize_extraction(extracted: dict, state: dict) -> dict:
    profile = state.get("user_profile", {})
    clarify_attempts = state.get("clarify_attempts", 0)

    procedure = extracted.get("procedure")
    if procedure not in SUPPORTED_PROCEDURES:
        procedure = None

    is_emergency = bool(extracted.get("is_emergency", False))
    emergency_confidence = float(extracted.get("emergency_confidence") or 0.0)
    recommendation_ready = bool(extracted.get("recommendation_ready", False))
    clarifying_question = extracted.get("clarifying_question")
    ambiguity_score = float(extracted.get("ambiguity_score") or 0.5)

    if clarify_attempts >= MAX_CLARIFY_ATTEMPTS:
        clarifying_question = None
    elif _question_was_asked(clarifying_question, state):
        clarifying_question = _fallback_question(state, procedure)

    if not clarifying_question and not recommendation_ready:
        clarifying_question = _fallback_question(state, procedure)

    if procedure and clarify_attempts < MAX_CLARIFY_ATTEMPTS and not is_emergency:
        follow_up_question = _fallback_question(state, procedure)
        if follow_up_question and not _question_was_asked(follow_up_question, state):
            clarifying_question = follow_up_question
            recommendation_ready = False

    if is_emergency:
        recommendation_ready = True
        clarifying_question = None
        ambiguity_score = min(ambiguity_score, 0.2)
        emergency_confidence = max(emergency_confidence, 0.85)

    if clarify_attempts >= MAX_CLARIFY_ATTEMPTS:
        recommendation_ready = True
        clarifying_question = None
        ambiguity_score = min(ambiguity_score, 0.5)
        if not procedure:
            procedure = "ct_scan"

    if recommendation_ready and not procedure:
        procedure = "ct_scan"

    if clarifying_question and clarify_attempts < MAX_CLARIFY_ATTEMPTS:
        ambiguity_score = max(ambiguity_score, 0.75)
        recommendation_ready = False

    return {
        "procedure": procedure,
        "city": extracted.get("city") or profile.get("city"),
        "budget": extracted.get("budget"),
        "deadline_days": extracted.get("deadline_days"),
        "is_emergency": is_emergency,
        "ambiguity_score": ambiguity_score,
        "clarifying_question": clarifying_question,
        "possible_causes": _safe_list(extracted.get("possible_causes")),
        "icd10_code": extracted.get("icd10_code"),
        "symptom_summary": extracted.get("symptom_summary") or state.get("user_input", ""),
        "recommendation_ready": recommendation_ready,
        "emergency_confidence": emergency_confidence,
        "follow_up_answers": extracted.get("follow_up_answers") if isinstance(extracted.get("follow_up_answers"), dict) else {},
    }


def _fallback_extraction(state: dict) -> dict:
    profile = state.get("user_profile", {})
    clarify_attempts = state.get("clarify_attempts", 0)
    route = _fallback_route(state)
    procedure = route.get("procedure")
    clarifying_question = route.get("clarifying_question") or _fallback_question(state, procedure)
    ready = (bool(procedure) and not clarifying_question) or clarify_attempts >= MAX_CLARIFY_ATTEMPTS
    is_emergency = bool(route.get("is_emergency", False))

    return {
        "procedure": procedure or ("ct_scan" if ready else None),
        "city": profile.get("city"),
        "budget": _extract_budget(_combined_user_text(state)),
        "deadline_days": None,
        "is_emergency": is_emergency,
        "ambiguity_score": route.get("ambiguity_score", 0.8 if not ready else 0.5),
        "clarifying_question": None if ready else clarifying_question,
        "possible_causes": route.get("possible_causes", []),
        "icd10_code": route.get("icd10_code"),
        "symptom_summary": state.get("user_input", ""),
        "recommendation_ready": ready,
        "emergency_confidence": route.get("emergency_confidence", 0.0),
        "follow_up_answers": {},
    }


def run_intent_node(state: dict) -> dict:
    user_input = state.get("user_input", "")
    profile = state.get("user_profile", {})
    nodes_visited = state.get("nodes_visited", [])
    nodes_visited.append("intent")

    prompt = INTENT_PROMPT.format(
        procedures=", ".join(SUPPORTED_PROCEDURES),
        clarify_attempts=state.get("clarify_attempts", 0),
        max_clarify_attempts=MAX_CLARIFY_ATTEMPTS,
        name=profile.get("name", "User"),
        age=profile.get("age", "unknown"),
        city=profile.get("city", "unknown"),
        comorbidities=", ".join(profile.get("comorbidities", [])) or "none",
        history=_history_text(state.get("conversation_history", [])),
        user_input=user_input,
    )

    try:
        response = model.generate_content(prompt)
        extracted = _json_from_model(response.text)
    except Exception as e:
        print(f"Intent Gemini error: {e}")
        extracted = _fallback_extraction(state)

    extracted = _normalize_extraction(extracted, state)

    return {
        **state,
        "procedure": extracted["procedure"],
        "city": extracted["city"],
        "budget": extracted["budget"] or state.get("budget"),
        "deadline_days": extracted["deadline_days"],
        "is_emergency": extracted["is_emergency"],
        "ambiguity_score": extracted["ambiguity_score"],
        "clarifying_question": extracted["clarifying_question"],
        "possible_causes": extracted["possible_causes"],
        "icd10_code": extracted["icd10_code"],
        "symptom_summary": extracted["symptom_summary"],
        "recommendation_ready": extracted["recommendation_ready"],
        "emergency_confidence": extracted["emergency_confidence"],
        "follow_up_answers": extracted["follow_up_answers"],
        "nodes_visited": nodes_visited,
    }
