import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# ── Supported procedures (exactly as in your CSV) ──────────────────────────────
SUPPORTED_PROCEDURES = [
    "angioplasty", "appendectomy", "arthroscopy", "bypass_cabg",
    "c_section", "cataract", "colonoscopy", "ct_scan", "dialysis_single",
    "ecg_echo", "endoscopy", "gallbladder_surgery", "hernia_repair",
    "hip_replacement", "hysterectomy", "kidney_stone_removal",
    "knee_replacement", "lasik", "mri_scan", "normal_delivery"
]

# ── Emergency symptom keywords ─────────────────────────────────────────────────
EMERGENCY_KEYWORDS = [
    "heart attack", "stroke", "can't breathe",
    "cannot breathe", "difficulty breathing", "severe bleeding",
    "unconscious", "collapsed", "seizure", "paralysis",
    "crushing pain", "left arm pain", "jaw pain", "sudden numbness"
]

HIGH_RISK_CHEST_PAIN_TERMS = [
    "arm", "left arm", "jaw", "back", "spreading", "radiating",
    "crushing", "walking", "walk", "climb", "stairs",
    "sweating", "breath", "nausea"
]

INTENT_PROMPT = """
You are a clinical intake AI for MedPath, an Indian healthcare navigation system.
Your job is to extract structured information from a user's health query.

SUPPORTED PROCEDURES (map user's words to exactly one of these):
{procedures}

USER QUERY: "{user_input}"

USER PROFILE (already known — do not ask for these again):
- Name: {name}
- Age: {age}
- City: {city}
- Comorbidities: {comorbidities}

CONVERSATION HISTORY:
{history}

Extract and return ONLY a valid JSON object with these fields:

{{
  "procedure": "<one of the supported procedures above, or null if unknown>",
  "city": "<city name, use profile city if not mentioned>",
  "budget": <integer in INR or null>,
  "deadline_days": <integer days from today or null>,
  "is_emergency": <true if symptoms suggest urgent/life-threatening, else false>,
  "ambiguity_score": <float 0.0-1.0, where >0.6 means clarification needed>,
  "clarifying_question": "<one focused question to ask if ambiguity_score > 0.6, else null>",
  "possible_causes": ["<medical condition 1>", "<medical condition 2>"],
  "icd10_code": "<ICD-10 code for most likely condition, or null>",
  "symptom_summary": "<1 sentence summary of what user described>",
  "recommendation_ready": <true only when enough symptom detail exists to show hospitals>,
  "emergency_confidence": <float 0.0-1.0 estimating confidence of immediate emergency>,
  "follow_up_answers": {{
    "pain_type": "<if mentioned>",
    "pain_location": "<if mentioned>",
    "duration": "<if mentioned>",
    "additional_symptoms": ["<symptom1>", "<symptom2>"]
  }}
}}

RULES:
1. If user directly mentions a procedure or condition, set ambiguity_score < 0.3
2. If user only describes a symptom, do NOT recommend hospitals yet. Set recommendation_ready=false, ambiguity_score > 0.6, and ask ONE natural follow-up question.
3. For chest pain alone, recommendation_ready=false and is_emergency=false. Ask about pain type, radiation to arm/jaw/back, exertion, duration, breathlessness, sweating, and nausea.
4. Set is_emergency=true only when red flags are present with high confidence, such as chest tightness/pressure radiating to arm/jaw/back, worse on exertion, severe breathing trouble, stroke signs, collapse, or uncontrolled bleeding.
5. If is_emergency=true, set emergency_confidence >= 0.85. Otherwise keep emergency_confidence below 0.85.
6. Set recommendation_ready=true when either the user asks for a specific procedure/hospital search, or follow-up answers provide enough severity/context to identify likely causes and urgency.
7. Always use the profile city if user does not mention a city
8. possible_causes should have 2-3 entries max
9. Return ONLY the JSON. No explanation. No markdown. No preamble.
"""

def run_intent_node(state: dict) -> dict:
    """
    Takes current state, calls Gemini to extract structured intent.
    Returns updated state.
    """
    user_input   = state.get("user_input", "")
    profile      = state.get("user_profile", {})
    history      = state.get("conversation_history", [])
    nodes_visited = state.get("nodes_visited", [])
    nodes_visited.append("intent")

    # Build history string for context
    history_str = ""
    if history:
        for turn in history[-6:]:  # last 3 exchanges
            history_str += f"User: {turn.get('user', '')}\n"
            history_str += f"MedPath: {turn.get('assistant', '')}\n"
    else:
        history_str = "None — this is the first message"

    prompt = INTENT_PROMPT.format(
        procedures   = ", ".join(SUPPORTED_PROCEDURES),
        user_input   = user_input,
        name         = profile.get("name", "User"),
        age          = profile.get("age", "unknown"),
        city         = profile.get("city", "unknown"),
        comorbidities= ", ".join(profile.get("comorbidities", [])) or "none",
        history      = history_str,
    )

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown fences if Gemini adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        extracted = json.loads(raw)

    except Exception as e:
        print(f"❌ intent_node Gemini error: {e}")
        # Fallback — ask clarifying question
        extracted = {
            "procedure":           None,
            "city":                profile.get("city"),
            "budget":              None,
            "deadline_days":       None,
            "is_emergency":        False,
            "ambiguity_score":     0.9,
            "clarifying_question": "Could you tell me more about what you're experiencing or the procedure you need?",
            "possible_causes":     [],
            "icd10_code":          None,
            "symptom_summary":     user_input,
            "recommendation_ready": False,
            "emergency_confidence": 0.0,
            "follow_up_answers":   {},
        }

    # Emergency override — check keywords even if Gemini missed it
    lower_input = user_input.lower()
    history_text = " ".join(
        turn.get("user", "")
        for turn in history[-4:]
    ).lower()
    context_text = f"{history_text} {lower_input}"
    has_chest_pain = "chest pain" in context_text or "chest tightness" in context_text
    high_risk_chest = has_chest_pain and any(term in context_text for term in HIGH_RISK_CHEST_PAIN_TERMS)

    if any(kw in lower_input for kw in EMERGENCY_KEYWORDS) or high_risk_chest:
        extracted["is_emergency"] = True
        extracted["emergency_confidence"] = max(extracted.get("emergency_confidence", 0) or 0, 0.9)
        extracted["ambiguity_score"] = min(extracted.get("ambiguity_score", 0), 0.3)
        extracted["recommendation_ready"] = True

    # If emergency — don't ask clarifying questions, go straight to hospitals
    emergency_confidence = extracted.get("emergency_confidence", 0) or 0

    if extracted.get("is_emergency") and emergency_confidence >= 0.85:
        extracted["ambiguity_score"] = 0.1
        extracted["clarifying_question"] = None
        extracted["recommendation_ready"] = True

    if "chest pain" in lower_input and not high_risk_chest and len(history) == 0:
        extracted["is_emergency"] = False
        extracted["emergency_confidence"] = 0.4
        extracted["recommendation_ready"] = False
        extracted["ambiguity_score"] = 0.9
        extracted["clarifying_question"] = (
            "I'm sorry you're dealing with that. Chest pain can have several causes, so let me ask a few quick questions first. "
            "How would you describe the pain: sharp and stabbing, dull and pressure-like, burning, or tightness?"
        )

    return {
        **state,
        "procedure":            extracted.get("procedure"),
        "city":                 extracted.get("city") or profile.get("city"),
        "budget":               extracted.get("budget") or state.get("budget"),
        "deadline_days":        extracted.get("deadline_days"),
        "is_emergency":         extracted.get("is_emergency", False),
        "ambiguity_score":      extracted.get("ambiguity_score", 0.5),
        "clarifying_question":  extracted.get("clarifying_question"),
        "possible_causes":      extracted.get("possible_causes", []),
        "icd10_code":           extracted.get("icd10_code"),
        "symptom_summary":      extracted.get("symptom_summary", user_input),
        "recommendation_ready": extracted.get("recommendation_ready", False),
        "emergency_confidence": extracted.get("emergency_confidence", 0),
        "follow_up_answers":    extracted.get("follow_up_answers", {}),
        "nodes_visited":        nodes_visited,
    }
