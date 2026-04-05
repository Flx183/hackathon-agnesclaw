def compute_risk(tasks, blockers, decisions, communication_flags, days_to_deadline):
    """
    Compute a risk score for the project.

    Key design: "blocked" tasks score much higher than "unknown owner" or
    "not_started" tasks. This separates "at-risk" (confused but willing to
    work) from "deadlocked" (can't work even if they wanted to).
    """
    score = 0

    # --- Task scoring ---
    for task in tasks:
        status = task.get("status", "unclear")
        owner_status = task.get("owner_status", "unknown")
        deadline_status = task.get("deadline_status", "unknown")

        # Skip completed tasks
        if status == "done":
            continue

        # Owner clarity
        if owner_status == "unknown":
            score += 2
        elif owner_status == "tentative":
            score += 1

        # Deadline clarity (only for active tasks)
        if deadline_status == "unknown":
            score += 1

        # Status severity
        if status == "blocked":
            score += 4   # highest weight -- this is the deadlock signal
        elif status == "unclear":
            score += 1

    # --- Blocker scoring ---
    for blocker in blockers:
        sev = blocker.get("severity", "low")
        if sev == "high":
            score += 3
        elif sev == "medium":
            score += 2
        else:
            score += 1
        if blocker.get("responsible_party") is None:
            score += 2   # unowned blocker is much worse

    # --- Decision scoring ---
    for decision in decisions:
        if decision.get("status") == "unresolved":
            score += 2

    # --- Communication flag scoring ---
    for flag in communication_flags:
        flag_type = flag.get("flag", "")
        if flag_type == "ownership_confusion":
            score += 2
        elif flag_type == "weak_commitment":
            score += 1
        elif flag_type == "dependency_risk":
            score += 2
        elif flag_type == "scope_confusion":
            score += 2
        elif flag_type == "vague_deadline":
            score += 1
        elif flag_type == "repeated_uncertainty":
            score += 1

    # --- Deadline proximity ---
    if days_to_deadline <= 2:
        score += 5
    elif days_to_deadline <= 3:
        score += 3
    elif days_to_deadline <= 5:
        score += 2
    elif days_to_deadline <= 7:
        score += 1

    return score


def risk_band(score):
    """
    Map score to risk level.

    Bands are calibrated so that:
    - Healthy group with confirmed owners and progress = low (0-8)
    - At-risk group with ownership confusion but no explicit blocks = moderate/high (9-35)
    - Deadlocked group with blocked tasks and high-severity blockers = deadlocked (36+)
    """
    if score <= 8:
        return "low"
    elif score <= 20:
        return "moderate"
    elif score <= 35:
        return "high"
    return "deadlocked"
