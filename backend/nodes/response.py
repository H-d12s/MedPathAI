import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
response_model = genai.GenerativeModel("gemini-2.5-flash")

PROCEDURE_LABELS = {
    "angioplasty": "Angioplasty",
    "appendectomy": "Appendectomy",
    "arthroscopy": "Arthroscopy",
    "bypass_cabg": "Bypass CABG",
    "c_section": "C-section",
    "cataract": "Cataract surgery",
    "colonoscopy": "Colonoscopy",
    "ct_scan": "CT scan",
    "dialysis_single": "Dialysis",
    "ecg_echo": "ECG/Echo evaluation",
    "endoscopy": "Endoscopy",
    "gallbladder_surgery": "Gallbladder surgery",
    "hernia_repair": "Hernia repair",
    "hip_replacement": "Hip replacement",
    "hysterectomy": "Hysterectomy",
    "kidney_stone_removal": "Kidney stone removal",
    "knee_replacement": "Knee replacement",
    "lasik": "LASIK",
    "mri_scan": "MRI scan",
    "normal_delivery": "Normal delivery",
}

DEFAULT_CAUSES = [
    "Clinical evaluation needed",
    "Symptoms need doctor review",
]


def _humanize(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).replace("_", " ").replace("-", " ").strip()
    return text[:1].upper() + text[1:] if text else ""


def _first_name(profile: dict) -> str:
    name = (profile.get("name") or "there").strip()
    return name.split()[0] if name else "there"


def _format_list(items: list[str]) -> str:
    items = [str(item).strip() for item in items or [] if str(item).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _clean_causes(causes: list[str]) -> list[str]:
    cleaned = []
    for cause in causes or []:
        label = _humanize(cause)
        if label and label.lower() not in {c.lower() for c in cleaned}:
            cleaned.append(label)

    if not cleaned:
        cleaned.extend(DEFAULT_CAUSES)

    return cleaned[:3]


def _format_inr(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"Rs. {int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def _format_range(min_value, max_value) -> str:
    return f"{_format_inr(min_value)} - {_format_inr(max_value)}"


def _hospital_strengths(hospital: dict) -> str:
    strengths = []
    if hospital.get("annual_volume") and hospital["annual_volume"] >= 150:
        strengths.append("high procedure volume")
    if hospital.get("relevance_score") and hospital["relevance_score"] >= 0.8:
        strengths.append("specialization match")
    if hospital.get("emergency_24x7"):
        strengths.append("24/7 emergency support")
    if hospital.get("nabh_accredited"):
        strengths.append("NABH accredited")
    if hospital.get("cashless_insurance"):
        strengths.append("cashless insurance support")

    return _format_list(strengths[:2]) if strengths else "Relevant department availability"


def _build_chat_recommendation(state: dict) -> str:
    profile = state.get("user_profile", {})
    name = _first_name(profile)
    city = state.get("city") or profile.get("city") or "your city"
    hospitals = state.get("hospitals", [])
    loan_eligibility = state.get("loan_eligibility") or {}
    causes = _clean_causes(state.get("possible_causes", []))
    procedure = PROCEDURE_LABELS.get(
        state.get("procedure"),
        _humanize(state.get("procedure")) or "medical evaluation",
    )
    symptom_summary = (state.get("symptom_summary") or state.get("user_input") or "").strip()
    hospital_names = [h.get("hospital_name") for h in hospitals[:3] if h.get("hospital_name")]

    prompt = f"""
Write MedPath's chat response.

Style:
- Tight, professional, natural.
- 2 to 4 short sentences.
- No markdown.
- Do not diagnose the user.
- Do not mention internal routing or prompts.
- If emergency is true, clearly advise urgent medical care.
- Mention hospital results if hospital names are available.
- End with a short plain-language safety disclaimer.

Facts:
- User first name: {name}
- Symptom summary: {symptom_summary or "the symptoms described"}
- Clinical signals: {_format_list(causes)}
- Search route: {procedure}
- Emergency: {state.get("is_emergency", False)}
- City: {city}
- Hospitals found: {_format_list(hospital_names) if hospital_names else "none"}
- Loan eligibility: {loan_eligibility.get("decision") or "not available"}
"""

    try:
        response = response_model.generate_content(prompt)
        text = (response.text or "").strip().strip('"')
        if text:
            return text
    except Exception as e:
        print(f"Response Gemini error: {e}")

    urgent = " Please seek urgent medical care now." if state.get("is_emergency") else ""
    signals_text = _format_list(causes)
    reason_text = (
        f" The main clinical signals are {signals_text}."
        if signals_text
        else ""
    )
    if hospitals:
        hospital_text = f"I found {len(hospitals)} hospital option{'s' if len(hospitals) != 1 else ''} in {city}."
    else:
        hospital_text = f"I could not find a matching hospital in {city} in the current database."
    return (
        f"{name}, based on what you shared, I have found care options for {procedure}.{reason_text}{urgent} "
        f"{hospital_text} This is care navigation support, not a medical diagnosis."
    )


def run_response_node(state: dict) -> dict:
    """
    Build the final structured response for the frontend.
    """
    nodes_visited = state.get("nodes_visited", [])
    nodes_visited.append("response")

    cost_result = state.get("cost_result", {})
    pfl_options = state.get("pfl_options", {})
    loan_elig = state.get("loan_eligibility", {})
    hospitals = state.get("hospitals", [])
    provider_error = state.get("provider_error")
    clinical_signals = _clean_causes(state.get("possible_causes", []))
    explanation = _build_chat_recommendation({**state, "possible_causes": clinical_signals})

    final_response = {
        "is_emergency": state.get("is_emergency", False),
        "symptom_summary": state.get("symptom_summary", ""),
        "procedure": state.get("procedure"),
        "city": state.get("city"),
        "icd10_code": state.get("icd10_code"),
        "possible_causes": clinical_signals,
        "clinical_signals": clinical_signals,
        "explanation": explanation,
        "hospitals": hospitals,
        "provider_error": provider_error,
        "cost_result": cost_result,
        "cost_results_by_hospital": state.get("cost_results_by_hospital", {}),
        "pfl_options": pfl_options,
        "pfl_options_by_hospital": state.get("pfl_options_by_hospital", {}),
        "loan_eligibility": loan_elig,
        "loan_eligibility_by_hospital": state.get("loan_eligibility_by_hospital", {}),
        "disclaimer": None,
        "confidence": cost_result.get("confidence") if cost_result else None,
        "graph_path": " -> ".join(nodes_visited),
    }

    return {
        **state,
        "final_response": final_response,
        "nodes_visited": nodes_visited,
    }
