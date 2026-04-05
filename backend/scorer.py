def compute_risk(tasks, blockers, decisions, communication_flags, days_to_deadline):
    score = 0
    for task in tasks:
        if task["owner_status"] in ["unknown"] and task["status"] != "done":
            score += 3
        if task["owner_status"] == "tentative":
            score += 1
        if task["deadline_status"] == "unknown" and task["status"] in ["in_progress", "blocked", "unclear"]:
            score += 1
        if task["status"] == "blocked":
            score += 3
    for blocker in blockers:
        if blocker["severity"] == "high":
            score += 3
        elif blocker["severity"] == "medium":
            score += 2
        else:
            score += 1
        if blocker["responsible_party"] is None:
            score += 2
    for decision in decisions:
        if decision["status"] == "unresolved":
            score += 2
    for flag in communication_flags:
        if flag["flag"] == "ownership_confusion":
            score += 2
        elif flag["flag"] == "weak_commitment":
            score += 1
        elif flag["flag"] == "dependency_risk":
            score += 2
        elif flag["flag"] == "scope_confusion":
            score += 2
        elif flag["flag"] == "vague_deadline":
            score += 1
    if days_to_deadline <= 3:
        score += 3
    elif days_to_deadline <= 7:
        score += 1
    return score


def risk_band(score):
    if score <= 4:
        return "low"
    elif score <= 8:
        return "moderate"
    elif score <= 13:
        return "high"
    return "deadlocked"
