from data_loader import (
    COMORBIDITY_MULTIPLIERS,
    calculate_cost_breakdown,
    calculate_pfl_options,
    check_loan_eligibility,
)


def _estimate_for_hospital(hospital: dict, profile: dict, financials: dict) -> dict:
    """Build cost, financing, and eligibility for one hospital card."""
    procedure = {
        "min_cost_inr": hospital["cost_min"],
        "max_cost_inr": hospital["cost_max"],
        "avg_cost_inr": hospital["cost_avg"],
        "success_rate": hospital.get("success_rate", 0.9),
        "avg_recovery_days": hospital.get("avg_recovery_days", 5),
        "insurance_covered": hospital.get("insurance_covered", False),
        "procedure_waiting_time_days": hospital.get("waiting_days", 3),
    }

    comorbidities = profile.get("comorbidities", [])
    cost_result = calculate_cost_breakdown(
        procedure=procedure,
        comorbidities=comorbidities,
        age=profile.get("age"),
        insurance_coverage=profile.get("insurance_coverage", 0),
    )

    loan_amount = int((cost_result["you_pay_min"] + cost_result["you_pay_max"]) / 2)
    pfl_options = calculate_pfl_options(loan_amount)

    if financials and financials.get("monthly_income"):
        loan_eligibility = check_loan_eligibility(
            loan_amount=loan_amount,
            monthly_income=financials.get("monthly_income", 0),
            existing_emi=financials.get("existing_emi", 0),
            cibil_score=financials.get("cibil_score", 700),
            employment_years=financials.get("employment_years", 2),
        )
    else:
        loan_eligibility = {
            "decision": "UNKNOWN",
            "recommendation": "Upload your financial documents in My Documents to get instant pre-approval",
            "flags": [],
            "foir": None,
            "score": None,
        }

    comorbidity_warnings = []
    for c in comorbidities:
        m = COMORBIDITY_MULTIPLIERS.get(c.lower(), {})
        if m:
            pct = int(m.get("cost_add", 0) * 100)
            comorbidity_warnings.append(
                f"{m.get('label', c)} may add ~{pct}% to total cost"
            )

    cost_result["comorbidity_warnings"] = comorbidity_warnings
    cost_result["insurance_provider"] = profile.get("insurance_provider")
    cost_result["selected_hospital"] = {
        "hospital_id": hospital["hospital_id"],
        "hospital_name": hospital["hospital_name"],
        "procedure": hospital["procedure_name"],
        "success_rate": hospital.get("success_rate"),
        "recovery_days": hospital.get("avg_recovery_days"),
        "waiting_days": hospital.get("waiting_days"),
    }

    return {
        "cost_result": cost_result,
        "pfl_options": pfl_options,
        "loan_eligibility": loan_eligibility,
        "loan_amount": loan_amount,
    }


def run_cost_node(state: dict) -> dict:
    """
    Calculates hospital-specific cost breakdowns, PFL EMI options, and loan
    eligibility. Each returned hospital gets its own estimate, so the frontend
    can swap breakdowns instantly when the user clicks a hospital card.
    """
    nodes_visited = state.get("nodes_visited", [])
    nodes_visited.append("cost")

    profile = state.get("user_profile", {})
    financials = state.get("user_financials", {})
    selected = state.get("selected_hospital")
    hospitals = state.get("hospitals", [])

    if not hospitals:
        return {
            **state,
            "cost_result": None,
            "pfl_options": None,
            "loan_eligibility": None,
            "cost_results_by_hospital": {},
            "pfl_options_by_hospital": {},
            "loan_eligibility_by_hospital": {},
            "nodes_visited": nodes_visited,
        }

    estimates_by_hospital = {
        hospital["hospital_id"]: _estimate_for_hospital(hospital, profile, financials)
        for hospital in hospitals
    }

    hospital = hospitals[0]
    if selected:
        hospital = next(
            (h for h in hospitals if h["hospital_id"] == selected),
            hospitals[0],
        )

    for h in hospitals:
        estimate = estimates_by_hospital[h["hospital_id"]]
        h["cost_result"] = estimate["cost_result"]
        h["pfl_options"] = estimate["pfl_options"]
        h["loan_eligibility"] = estimate["loan_eligibility"]
        h["loan_amount"] = estimate["loan_amount"]

    selected_estimate = estimates_by_hospital[hospital["hospital_id"]]

    return {
        **state,
        "hospitals": hospitals,
        "cost_result": selected_estimate["cost_result"],
        "pfl_options": selected_estimate["pfl_options"],
        "loan_eligibility": selected_estimate["loan_eligibility"],
        "loan_amount": selected_estimate["loan_amount"],
        "cost_results_by_hospital": {
            hospital_id: estimate["cost_result"]
            for hospital_id, estimate in estimates_by_hospital.items()
        },
        "pfl_options_by_hospital": {
            hospital_id: estimate["pfl_options"]
            for hospital_id, estimate in estimates_by_hospital.items()
        },
        "loan_eligibility_by_hospital": {
            hospital_id: estimate["loan_eligibility"]
            for hospital_id, estimate in estimates_by_hospital.items()
        },
        "nodes_visited": nodes_visited,
    }
