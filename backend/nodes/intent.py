import json
import os
import re

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

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
- Use clinical reasoning to choose a useful hospital-search route whenever the user's symptoms support one. Do not default to null or overly cautious wording if the information points to a specific procedure or diagnostic path.
- If the user describes sudden right lower abdominal pain, especially with nausea, fever, or worsening motion, favor appendectomy as the hospital-search route and flag appendicitis risk without offering a formal diagnosis.
- Use conversation history to avoid asking the same follow-up twice. If the previous assistant question already requested the same detail, ask a different focused question or proceed to the best supported route.
- You are not diagnosing the user. You are identifying clinical signals for care navigation.
- Do not invent facts not present in the conversation.
- Ask at most one focused clarification question in this turn.
- The app has already asked {clarify_attempts} clarification question(s). If this is 2 or more, do not ask another question. Choose the closest hospital-search route now.
- If there are emergency warning signs, mark emergency and make the route ready immediately.
- If details are limited but enough to search, choose the best supported procedure or diagnostic route.

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

    if clarifying_question and clarify_attempts > 0:
        history = state.get("conversation_history", [])
        last_assistant = next((turn.get("assistant") for turn in reversed(history) if turn.get("assistant")), None)
        if last_assistant and _normalize_text(clarifying_question) == _normalize_text(last_assistant):
            clarifying_question = None

    if is_emergency:
        recommendation_ready = True
        clarifying_question = None
        ambiguity_score = min(ambiguity_score, 0.2)
        emergency_confidence = max(emergency_confidence, 0.85)

    if clarify_attempts >= 2:
        recommendation_ready = True
        clarifying_question = None
        ambiguity_score = min(ambiguity_score, 0.5)
        if not procedure:
            procedure = "ct_scan"

    if recommendation_ready and not procedure:
        procedure = "ct_scan"

    if clarifying_question and clarify_attempts < 2 and not recommendation_ready:
        ambiguity_score = max(ambiguity_score, 0.75)

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
    ready = clarify_attempts >= 2

    return {
        "procedure": "ct_scan" if ready else None,
        "city": profile.get("city"),
        "budget": None,
        "deadline_days": None,
        "is_emergency": False,
        "ambiguity_score": 0.8 if not ready else 0.5,
        "clarifying_question": None if ready else "What symptoms are you having, and how long has this been going on?",
        "possible_causes": [],
        "icd10_code": None,
        "symptom_summary": state.get("user_input", ""),
        "recommendation_ready": ready,
        "emergency_confidence": 0.0,
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
