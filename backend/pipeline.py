import json
import logging
import os
import re
from datetime import date, datetime

from openai import OpenAI

from parser import parse_chat, preprocess_text
from prompts import (
    CONTEXT_EXTRACTION_PROMPT,
    CONTRADICTION_PROMPT,
    DEADLOCK_PROMPT,
    EXTRACTION_PROMPT,
    FOLLOWUP_PROMPT,
    NORMALIZATION_PROMPT,
)
from scorer import compute_risk, risk_band

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_mock_mode() -> bool:
    if os.environ.get("MOCK_MODE", "").lower() in ("1", "true", "yes"):
        return True
    return not bool(os.environ.get("OPENCLAW_API_KEY", "").strip())


def _extract_json_from_text(text: str):
    """Find and parse JSON from LLM output using multiple strategies."""
    if not text or not text.strip():
        return None

    text = text.strip()

    # Strategy 1: strip leading/trailing fences and parse directly
    stripped = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    stripped = re.sub(r'\s*```$', '', stripped).strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass

    # Strategy 2: find ```json ... ``` block anywhere in text
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except Exception:
            pass

    # Strategy 3: brace-match the first top-level { ... } or [ ... ]
    start = text.find('{')
    if start == -1:
        start = text.find('[')
    if start == -1:
        return None

    open_char = text[start]
    close_char = '}' if open_char == '{' else ']'
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _safe_json(text: str, default):
    result = _extract_json_from_text(text)
    if result is not None:
        return result
    logger.warning("JSON parse failed. Raw (first 500): %s", (text or "")[:500])
    return default


def _make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENCLAW_API_KEY", "mock"),
        base_url=os.environ.get("OPENCLAW_BASE_URL", "https://zenmux.ai/api/v1"),
    )


def _call_llm(client: OpenAI, system: str, user: str) -> str:
    model = os.environ.get("OPENCLAW_MODEL", "sapiens-ai/agnes-1.5-pro")
    logger.info("LLM call: model=%s, sys=%d, user=%d", model, len(system), len(user))
    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or ""
        logger.info("LLM response: %d chars", len(content))
        return content
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return ""


def _call_vision(client: OpenAI, prompt: str, mime: str, b64: str) -> str:
    model = os.environ.get("OPENCLAW_MODEL", "sapiens-ai/agnes-1.5-pro")
    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=1024,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error("Vision failed: %s", e)
        return f"[Vision failed: {e}]"


# ---------------------------------------------------------------------------
# Heuristic fallbacks
# ---------------------------------------------------------------------------

def _heuristic_normalize(parsed_messages: list[dict]) -> list[dict]:
    normalized = []
    for m in parsed_messages:
        raw = m.get("raw_text", "")
        expanded = preprocess_text(raw)
        tl = expanded.lower()
        signals = {
            "contains_task_reference": any(w in tl for w in
                ["do", "finish", "complete", "handle", "prepare", "compile",
                 "write", "submit", "send", "start", "work on", "slides",
                 "report", "survey", "data", "code", "design", "cleaning",
                 "presentation", "intro", "section", "angle", "decide",
                 "pushed", "github", "review", "queries", "api", "backend"]),
            "contains_owner_reference": any(w in tl for w in
                ["i will", "i'll", "i can", "you", "who", "someone",
                 "i thought", "was doing", "responsible", "i think",
                 "anyone", "no one", "i tot", "did anyone"]),
            "contains_deadline_reference": any(w in tl for w in
                ["tomorrow", "today", "tonight", "by", "deadline",
                 "friday", "monday", "tuesday", "wednesday", "thursday",
                 "saturday", "sunday", "next week", "as soon as possible",
                 "soon", "morning", "evening"]),
            "contains_blocker_reference": any(w in tl for w in
                ["can't", "cannot", "cant", "waiting", "blocked",
                 "have not", "haven't", "need", "stuck", "depends", "haven"]),
            "contains_uncertainty": any(w in tl for w in
                ["maybe", "probably", "i think", "not sure", "i thought",
                 "see first", "i guess", "i do not know", "don't know",
                 "what", "prob", "did anyone"]),
            "contains_commitment": any(w in tl for w in
                ["i will", "i'll", "done", "finished", "completed",
                 "already", "confirmed", "yes", "i can"]),
            "contains_dependency": any(w in tl for w in
                ["waiting", "need", "depends", "before", "after",
                 "first", "then", "once", "until", "blocked"]),
        }
        normalized.append({
            "speaker": m.get("speaker", "Unknown"),
            "raw_text": raw,
            "normalized_text": expanded,
            "confidence": "medium",
            "signals": signals,
        })
    return normalized


def _heuristic_extract(parsed_messages, normalized_messages, members):
    """Extract tasks, blockers, decisions, flags using regex. Safety net for Agnes."""
    tasks = []
    blockers = []
    decisions = []
    questions = []
    communication_flags = []

    members_lower = {m.lower(): m for m in members}
    task_counter = 0
    seen_tasks = set()

    TASK_PATTERNS = [
        (r'\b(?:data\s*clean(?:ing)?)\b', 'Data cleaning'),
        (r'\b(?:presentation|slides?|deck)\b', 'Prepare presentation slides'),
        (r'\b(?:intro(?:duction)?\s*(?:section)?)\b', 'Write introduction section'),
        (r'\b(?:survey\s*results?)\b', 'Compile survey results'),
        (r'\b(?:final\s*(?:angle|direction)|thesis)\b', 'Decide final project angle'),
        (r'\b(?:report|write[\s-]?up)\b', 'Write report'),
        (r'\b(?:code|coding|program(?:ming)?|script)\b', 'Complete coding work'),
        (r'\b(?:design|mockup|wireframe|prototype)\b', 'Create design'),
        (r'\b(?:research|literature\s*review|reading)\b', 'Conduct research'),
        (r'\b(?:edit(?:ing)?|proofread(?:ing)?|review\s+pass)\b', 'Edit and review'),
        (r'\b(?:refs?|references?|bibliography|citations?)\b', 'Compile references'),
        (r'\b(?:er\s*diagram|schema|database)\b', 'Database work'),
        (r'\b(?:sql|quer(?:y|ies))\b', 'SQL queries'),
        (r'\b(?:backend|api|frontend)\b', 'Application development'),
    ]

    CONFUSION_PATTERNS = [
        r'i\s*(?:thought|tot)\s+(?:you|u|he|she|they|\w+)\s+(?:was|were|is|doing|handling|compiling)',
        r'(?:who(?:\'?s)?|anyone)\s+(?:is\s+)?(?:doing|handling|responsible|starting)',
        r'(?:someone|anybody)\s+needs?\s+to',
        r'(?:did\s+anyone|not\s+sure\s+who)',
    ]

    WEAK_PATTERNS = [
        r'\bok\s+i\s+see\s+first\b',
        r'\bsee\s+first\b',
        r'\bi\s+can\s+(?:maybe|prob(?:ably)?)\b',
        r'\b(?:maybe|might)\s+(?:i|we|can)\b',
        r'\bprob(?:ably)?\s',
    ]

    BLOCKER_PATTERNS = [
        r"\b(?:can'?t|cannot|cant)\s+(?:do|start|continue|finish|proceed)",
        r'\b(?:waiting|stuck|blocked)\b',
        r"\b(?:have\s*n(?:ot|'t)|haven)\s+(?:send|sent|done|finish|start)",
        r'\bneed\s+(?:the|that|this|it|to\s+get|results?|material)',
    ]

    DECISION_PATTERNS = [
        r'\bwhat\s+(?:are|is|should)\s+we\b',
        r'\bshould\s+we\b',
        r'\bwho\s+(?:is|should)\s+(?:do|handle|doing)\b',
        r'\b(?:decide|decision|agree\s+on|settle)\b',
        r'\bwhat\s+(?:are|is)\s+(?:we|the|our)\s+(?:using|doing|going)\b',
    ]

    ownership_claims = {}
    uncertain_speakers = set()
    weak_speakers = set()
    blocker_speakers_set = set()
    dependency_speakers = set()
    vague_deadline_speakers = set()

    for nm in normalized_messages:
        speaker = nm.get("speaker", "Unknown")
        raw = nm.get("raw_text", "")
        text = nm.get("normalized_text", raw)
        tl = text.lower()
        rl = raw.lower()
        signals = nm.get("signals", {})

        # --- Tasks ---
        for pattern, task_name in TASK_PATTERNS:
            if re.search(pattern, tl) or re.search(pattern, rl):
                if task_name not in seen_tasks:
                    seen_tasks.add(task_name)
                    task_counter += 1

                    owner = None
                    owner_status = "unknown"
                    if re.search(r"\bi(?:'ll| will| am| can)\b", tl):
                        owner = speaker
                        owner_status = "tentative" if signals.get("contains_uncertainty") else "confirmed"
                    if any(w in tl for w in ["done", "finished", "completed", "already pushed"]):
                        owner = speaker
                        owner_status = "confirmed"
                    if not owner:
                        for ml, m_orig in members_lower.items():
                            if ml in tl and ml != speaker.lower():
                                owner = m_orig
                                owner_status = "tentative"
                                break

                    status = "not_started"
                    if any(re.search(bp, tl) for bp in BLOCKER_PATTERNS):
                        status = "blocked"
                    elif any(w in tl for w in ["done", "finished", "completed", "already pushed"]):
                        status = "done"
                    elif any(w in tl for w in ["doing", "working on", "started", "in progress", "80%", "starting"]):
                        status = "in_progress"
                    elif signals.get("contains_uncertainty"):
                        status = "unclear"

                    deadline_val = None
                    deadline_status = "unknown"
                    dl_match = re.search(r'\b(?:by|before)\s+(\w+(?:\s+\w+)?)\b', tl)
                    if dl_match:
                        deadline_val = dl_match.group(1).strip().capitalize()
                        deadline_status = "known"
                    elif re.search(r'\b(?:tomorrow|tonight)\b', tl):
                        deadline_val = "Tomorrow"
                        deadline_status = "known"

                    tasks.append({
                        "task_id": f"t{task_counter}",
                        "task_name": task_name,
                        "owner": owner,
                        "owner_status": owner_status,
                        "deadline": deadline_val,
                        "deadline_status": deadline_status,
                        "status": status,
                        "depends_on": [],
                        "evidence": [f"{speaker}: {raw}"],
                    })

        # --- Blockers ---
        for bp in BLOCKER_PATTERNS:
            if re.search(bp, tl) or re.search(bp, rl):
                blocker_speakers_set.add(speaker)
                responsible = None
                for ml, m_orig in members_lower.items():
                    if ml in tl and ml != speaker.lower():
                        responsible = m_orig
                        break
                blockers.append({
                    "blocker": text if len(text) < 120 else text[:117] + "...",
                    "blocking_task": None,
                    "responsible_party": responsible,
                    "severity": "high" if re.search(BLOCKER_PATTERNS[0], tl) else "medium",
                    "evidence": [f"{speaker}: {raw}"],
                })
                break

        # --- Ownership confusion ---
        for cp in CONFUSION_PATTERNS:
            if re.search(cp, tl) or re.search(cp, rl):
                ownership_claims.setdefault(speaker, []).append(raw)
                break

        # --- Decisions ---
        for dp in DECISION_PATTERNS:
            if re.search(dp, tl) or re.search(dp, rl):
                desc = text if len(text) < 150 else text[:147] + "..."
                decisions.append({
                    "decision": desc, "status": "unresolved",
                    "evidence": [f"{speaker}: {raw}"],
                })
                questions.append(desc)
                break

        # --- Track signals ---
        if signals.get("contains_uncertainty"):
            uncertain_speakers.add(speaker)
        for wp in WEAK_PATTERNS:
            if re.search(wp, tl) or re.search(wp, rl):
                weak_speakers.add(speaker)
                break
        if re.search(r'\b(?:tomorrow|tmr|soon|later|eventually)\b', rl):
            if not re.search(r'\d{4}|\d{1,2}(?:am|pm|:\d{2})', rl):
                vague_deadline_speakers.add(speaker)
        if signals.get("contains_dependency"):
            dependency_speakers.add(speaker)

    # --- Communication flags ---
    if len(ownership_claims) >= 2:
        communication_flags.append({
            "flag": "ownership_confusion",
            "description": f"Multiple members confused about task ownership: {', '.join(ownership_claims.keys())}.",
            "involved_members": list(ownership_claims.keys()),
        })
    elif len(ownership_claims) == 1:
        s = list(ownership_claims.keys())[0]
        communication_flags.append({
            "flag": "ownership_confusion",
            "description": f"{s} expressed confusion about who owns a task.",
            "involved_members": [s],
        })

    if weak_speakers:
        communication_flags.append({
            "flag": "weak_commitment",
            "description": f"Non-committal language from: {', '.join(weak_speakers)}.",
            "involved_members": list(weak_speakers),
        })

    if vague_deadline_speakers:
        communication_flags.append({
            "flag": "vague_deadline",
            "description": f"Vague deadline references from: {', '.join(vague_deadline_speakers)}.",
            "involved_members": list(vague_deadline_speakers),
        })

    if dependency_speakers:
        communication_flags.append({
            "flag": "dependency_risk",
            "description": f"Dependency/blocking language from: {', '.join(dependency_speakers)}.",
            "involved_members": list(dependency_speakers),
        })

    if len(uncertain_speakers) >= 3:
        communication_flags.append({
            "flag": "repeated_uncertainty",
            "description": f"Widespread uncertainty: {', '.join(uncertain_speakers)}.",
            "involved_members": list(uncertain_speakers),
        })

    scope_speakers = [
        nm["speaker"] for nm in normalized_messages
        if re.search(r'what\s+(?:are|is)\s+(?:we|the|our)', nm.get("raw_text", "").lower())
    ]
    if scope_speakers:
        communication_flags.append({
            "flag": "scope_confusion",
            "description": f"Members questioning project direction: {', '.join(set(scope_speakers))}.",
            "involved_members": list(set(scope_speakers)),
        })

    # Deduplicate
    seen_b = set()
    unique_blockers = [b for b in blockers if b["blocker"][:50].lower() not in seen_b and not seen_b.add(b["blocker"][:50].lower())]
    seen_d = set()
    unique_decisions = [d for d in decisions if d["decision"][:50].lower() not in seen_d and not seen_d.add(d["decision"][:50].lower())]

    # Link blockers to tasks
    for b in unique_blockers:
        for t in tasks:
            tw = set(t["task_name"].lower().split())
            bw = set(b["blocker"].lower().split())
            if len(tw & bw) >= 2:
                b["blocking_task"] = t["task_name"]
                if t["status"] != "done":
                    t["status"] = "blocked"
                break

    return {
        "tasks": tasks, "decisions": unique_decisions,
        "questions": questions, "blockers": unique_blockers,
        "communication_flags": communication_flags,
    }


def _merge_extracted(llm_result: dict, heuristic_result: dict) -> dict:
    """
    Merge LLM extraction results with heuristic results.
    LLM results take priority; heuristic fills gaps.
    """
    merged = {}

    # For tasks: use heuristic if LLM returned none
    merged["tasks"] = llm_result.get("tasks") or heuristic_result.get("tasks", [])

    # For blockers: use heuristic if LLM returned none
    merged["blockers"] = llm_result.get("blockers") or heuristic_result.get("blockers", [])

    # For decisions: merge both, deduplicate
    llm_decisions = llm_result.get("decisions", [])
    h_decisions = heuristic_result.get("decisions", [])
    seen = set()
    merged_decisions = []
    for d in llm_decisions + h_decisions:
        key = d.get("decision", "")[:50].lower()
        if key not in seen:
            seen.add(key)
            merged_decisions.append(d)
    merged["decisions"] = merged_decisions

    # For communication flags: merge both, deduplicate by flag type
    llm_flags = llm_result.get("communication_flags", [])
    h_flags = heuristic_result.get("communication_flags", [])
    seen_flags = set()
    merged_flags = []
    for f in llm_flags + h_flags:
        key = f.get("flag", "")
        if key not in seen_flags:
            seen_flags.add(key)
            merged_flags.append(f)
    merged["communication_flags"] = merged_flags

    # Questions: merge
    llm_q = llm_result.get("questions", [])
    h_q = heuristic_result.get("questions", [])
    seen_q = set()
    merged_q = []
    for q in llm_q + h_q:
        key = q[:50].lower()
        if key not in seen_q:
            seen_q.add(key)
            merged_q.append(q)
    merged["questions"] = merged_q

    return merged


# ---------------------------------------------------------------------------
# Source processing
# ---------------------------------------------------------------------------

def _process_source(client, source):
    file_type = source.get("file_type", "text")
    raw_content = source.get("extracted_text", "")
    filename = source.get("filename", "document")

    if file_type == "image":
        try:
            img_data = json.loads(raw_content)
            readable_text = _call_vision(client,
                "Extract ALL text, requirements, deadlines from this image.",
                img_data["mime"], img_data["b64"])
        except Exception:
            readable_text = "[Image: could not extract text]"
    elif file_type == "video":
        readable_text = raw_content
    else:
        readable_text = raw_content

    context_raw = _call_llm(client, CONTEXT_EXTRACTION_PROMPT, json.dumps({
        "filename": filename, "file_type": file_type,
        "reliability_label": source.get("reliability_label", "auto"),
        "content": readable_text[:8000],
    }))
    context = _safe_json(context_raw, {
        "source_summary": readable_text[:200], "document_type": "unknown",
        "deliverables": [], "requirements": [], "deadlines": [],
        "evaluation_criteria": [], "constraints": [], "key_topics": [],
        "uncertainty_notes": [],
    })
    return {**source, "readable_text": readable_text, "context": context}


def _detect_contradictions(client, processed_sources, extracted_chat):
    sources_summary = [{
        "filename": s.get("filename"),
        "reliability_score": s.get("reliability_score", 6),
        "reliability_label": s.get("reliability_label", "auto"),
        "context": s.get("context", {}),
    } for s in processed_sources]
    raw = _call_llm(client, CONTRADICTION_PROMPT, json.dumps({
        "document_sources": sources_summary,
        "chat_state": {
            "tasks": extracted_chat.get("tasks", []),
            "decisions": extracted_chat.get("decisions", []),
            "blockers": extracted_chat.get("blockers", []),
            "communication_flags": extracted_chat.get("communication_flags", []),
        },
    }))
    return _safe_json(raw, {
        "contradictions": [], "requirement_gaps": [], "source_conflicts": [],
        "overall_alignment": "aligned", "alignment_summary": "No sources provided.",
    })


# ---------------------------------------------------------------------------
# Mock response
# ---------------------------------------------------------------------------

def _mock_analyze(project_name, deadline, members, raw_chat, tone):
    parsed = parse_chat(raw_chat, members)
    days = _days_to_deadline(deadline)

    tasks = [
        {"task_id": "t1", "task_name": "Compile survey results", "owner": None,
         "owner_status": "unknown", "deadline": None, "deadline_status": "unknown",
         "status": "blocked", "depends_on": [], "evidence": ["Ben has not sent the required material."]},
        {"task_id": "t2", "task_name": "Prepare slides", "owner": "Alex",
         "owner_status": "tentative", "deadline": None, "deadline_status": "unknown",
         "status": "blocked", "depends_on": ["t1"], "evidence": ["Alex cannot continue slides until survey results are compiled."]},
        {"task_id": "t3", "task_name": "Decide final project angle", "owner": None,
         "owner_status": "unknown", "deadline": None, "deadline_status": "unknown",
         "status": "not_started", "depends_on": [], "evidence": ["Dion raised an unresolved question about the final direction."]},
    ]
    blockers = [
        {"blocker": "Survey results have not been sent by Ben", "blocking_task": "Compile survey results",
         "responsible_party": "Ben", "severity": "high", "evidence": ["Alex: ben haven send survey leh"]},
        {"blocker": "Final project angle is undecided", "blocking_task": "Prepare slides",
         "responsible_party": None, "severity": "high", "evidence": ["Dion: what are we even using for final angle"]},
    ]
    decisions = [
        {"decision": "Who is responsible for compiling the survey results?", "status": "unresolved",
         "evidence": ["Ben: i tot alex compiling", "Alex: ben haven send survey leh"]},
        {"decision": "What is the final project angle or thesis?", "status": "unresolved",
         "evidence": ["Dion: what are we even using for final angle"]},
    ]
    communication_flags = [
        {"flag": "ownership_confusion", "description": "Both Alex and Ben believe the other is responsible for survey results.", "involved_members": ["Alex", "Ben"]},
        {"flag": "weak_commitment", "description": "Ben responded 'ok i see first' with no clear action.", "involved_members": ["Ben"]},
        {"flag": "vague_deadline", "description": "Cara proposed 'tmr settle' without specifying time or owner.", "involved_members": ["Cara"]},
        {"flag": "scope_confusion", "description": "Final project angle is unresolved.", "involved_members": ["Dion"]},
    ]

    rule_score = compute_risk(tasks, blockers, decisions, communication_flags, days)
    rule_level = risk_band(rule_score)

    tone_messages = {
        "diplomatic": f"Hey everyone, two things need sorting: (1) who is compiling the survey results -- Alex and Ben, can you confirm? And (2) Dion's point about the final angle -- can we agree today? Once cleared, Alex can continue slides. Lock both by 9pm tonight?",
        "direct": f"Two blockers: (1) Ben or Alex -- who is sending survey results? Decide now. (2) What is the final angle? Nothing moves until these are done. Reply by 9pm.",
        "formal": f"Hi team, to remain on track for {deadline}, two items need resolution: survey result ownership (Alex/Ben) and the project angle (Dion's question). Please confirm by 9pm today.",
        "casual": f"hey guys -- who's doing the survey results, alex or ben? also dion's q about the final angle is still hanging. sort both by 9pm tonight?",
    }

    return {
        "project_name": project_name, "deadline": deadline, "members": members,
        "days_to_deadline": days,
        "normalized_messages": [
            {"speaker": m["speaker"], "raw_text": m["raw_text"],
             "normalized_text": f"[MOCK] {m['raw_text']}", "confidence": "medium",
             "signals": {"contains_task_reference": True, "contains_owner_reference": False,
                         "contains_deadline_reference": False, "contains_blocker_reference": False,
                         "contains_uncertainty": True, "contains_commitment": False, "contains_dependency": False}}
            for m in parsed],
        "tasks": tasks, "decisions": decisions,
        "questions": ["Who is compiling the survey results?", "What is the final project angle?"],
        "blockers": blockers, "communication_flags": communication_flags,
        "rule_based_score": rule_score, "rule_based_level": rule_level,
        "risk_score": rule_score, "risk_level": rule_level,
        "status_summary": f"[MOCK] Project at {rule_level} risk. Two blockers: unresolved survey ownership and undecided angle.",
        "top_causes": [
            {"cause": "Ownership confusion over survey compilation", "severity": "high", "why_it_matters": "Alex and Ben each think the other is responsible."},
            {"cause": "Final angle unresolved", "severity": "high", "why_it_matters": "All content work depends on this."},
            {"cause": "Weak commitment from Ben", "severity": "medium", "why_it_matters": "'Ok I see first' confirms nothing."},
        ],
        "clarifications_needed": ["Who compiles survey results?", "What is the final angle?", "Check-in deadline?"],
        "next_actions": [
            {"priority": 1, "action": "Ben and Alex confirm who compiles survey results.", "owner": "Ben", "deadline": "Today"},
            {"priority": 2, "action": "Team agrees on final project angle.", "owner": "Dion", "deadline": "Today"},
            {"priority": 3, "action": "Alex resumes slides once blockers cleared.", "owner": "Alex", "deadline": deadline},
        ],
        "followup_message": tone_messages.get(tone, tone_messages["diplomatic"]),
        "sources": [],
        "contradiction_analysis": {"contradictions": [], "requirement_gaps": [], "source_conflicts": [], "overall_alignment": "aligned", "alignment_summary": "No sources uploaded."},
    }


def _days_to_deadline(deadline_str):
    try:
        return max((datetime.strptime(deadline_str, "%Y-%m-%d").date() - date.today()).days, 0)
    except Exception:
        return 7


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def analyze_project(
    project_name: str,
    deadline: str,
    members: list[str],
    raw_chat: str,
    tone: str,
    sources: list[dict] | None = None,
) -> dict:
    if _is_mock_mode():
        return _mock_analyze(project_name, deadline, members, raw_chat, tone)

    client = _make_client()

    # 1. Parse
    parsed_messages = parse_chat(raw_chat, members)
    logger.info("Parsed %d messages", len(parsed_messages))
    if not parsed_messages:
        return _mock_analyze(project_name, deadline, members, raw_chat, tone)

    # 2. Normalize via LLM (heuristic fallback)
    norm_raw = _call_llm(client, NORMALIZATION_PROMPT,
        json.dumps({"members": members, "messages": parsed_messages}))
    norm_result = _safe_json(norm_raw, {"normalized_messages": []})
    normalized_messages = norm_result.get("normalized_messages", [])
    if not normalized_messages:
        logger.warning("LLM normalization empty -- heuristic fallback")
        normalized_messages = _heuristic_normalize(parsed_messages)
    logger.info("Normalized: %d messages", len(normalized_messages))

    # 3. Extract via LLM
    ext_raw = _call_llm(client, EXTRACTION_PROMPT, json.dumps({
        "normalized_messages": normalized_messages,
        "members": members, "project_name": project_name, "deadline": deadline,
    }))
    llm_extracted = _safe_json(ext_raw, {
        "tasks": [], "decisions": [], "questions": [],
        "blockers": [], "communication_flags": [],
    })

    # 4. ALWAYS run heuristic extraction as well
    #    Then merge: LLM results take priority, heuristic fills gaps
    heuristic_result = _heuristic_extract(parsed_messages, normalized_messages, members)

    merged = _merge_extracted(llm_extracted, heuristic_result)

    tasks = merged["tasks"]
    blockers = merged["blockers"]
    decisions = merged["decisions"]
    communication_flags = merged["communication_flags"]
    all_questions = merged["questions"]

    logger.info("Merged: %d tasks, %d blockers, %d decisions, %d flags",
                len(tasks), len(blockers), len(decisions), len(communication_flags))

    # 5. Process sources
    processed_sources = []
    contradiction_analysis = {
        "contradictions": [], "requirement_gaps": [], "source_conflicts": [],
        "overall_alignment": "aligned", "alignment_summary": "No sources uploaded.",
    }
    if sources:
        for src in sorted(sources, key=lambda s: s.get("reliability_score", 6), reverse=True):
            processed_sources.append(_process_source(client, src))
        contradiction_analysis = _detect_contradictions(client, processed_sources, {
            "tasks": tasks, "decisions": decisions,
            "blockers": blockers, "communication_flags": communication_flags,
        })
        high_c = [c for c in contradiction_analysis.get("contradictions", []) if c.get("severity") == "high"]
        if high_c:
            communication_flags.append({
                "flag": "scope_confusion",
                "description": f"{len(high_c)} high-severity contradiction(s) with documents.",
                "involved_members": [],
            })

    # 6. Score
    days_to_deadline = _days_to_deadline(deadline)
    rule_score = compute_risk(tasks, blockers, decisions, communication_flags, days_to_deadline)
    rule_level = risk_band(rule_score)
    logger.info("Rule score: %d (%s)", rule_score, rule_level)

    # 7. Deadlock analysis via LLM
    deadlock_raw = _call_llm(client, DEADLOCK_PROMPT, json.dumps({
        "project_name": project_name, "deadline": deadline,
        "days_to_deadline": days_to_deadline,
        "tasks": tasks, "blockers": blockers,
        "decisions": decisions, "communication_flags": communication_flags,
        "rule_based_risk_score": rule_score,
        "rule_based_risk_level": rule_level,
        "document_contradictions": contradiction_analysis.get("contradictions", []),
        "requirement_gaps": contradiction_analysis.get("requirement_gaps", []),
    }))
    deadlock_result = _safe_json(deadlock_raw, {
        "risk_score": rule_score, "risk_level": rule_level,
        "status_summary": "Unable to generate detailed analysis.",
        "top_causes": [], "clarifications_needed": [], "next_actions": [],
    })

    # Ensure LLM doesn't downgrade below rule-based score
    final_score = max(deadlock_result.get("risk_score", rule_score), rule_score)
    final_level = risk_band(final_score)

    # 8. Follow-up message via LLM
    followup_raw = _call_llm(client, FOLLOWUP_PROMPT, json.dumps({
        "project_name": project_name, "deadline": deadline, "tone": tone,
        "top_causes": deadlock_result.get("top_causes", []),
        "clarifications_needed": deadlock_result.get("clarifications_needed", []),
        "next_actions": deadlock_result.get("next_actions", []),
        "members": members,
        "critical_contradictions": [c for c in contradiction_analysis.get("contradictions", []) if c.get("severity") == "high"],
    }))
    followup_result = _safe_json(followup_raw, {"message": "Could not generate follow-up message."})

    return {
        "project_name": project_name, "deadline": deadline, "members": members,
        "days_to_deadline": days_to_deadline,
        "normalized_messages": normalized_messages,
        "tasks": tasks, "decisions": decisions,
        "questions": all_questions,
        "blockers": blockers, "communication_flags": communication_flags,
        "rule_based_score": rule_score, "rule_based_level": rule_level,
        "risk_score": final_score, "risk_level": final_level,
        "status_summary": deadlock_result.get("status_summary", ""),
        "top_causes": deadlock_result.get("top_causes", []),
        "clarifications_needed": deadlock_result.get("clarifications_needed", []),
        "next_actions": deadlock_result.get("next_actions", []),
        "followup_message": followup_result.get("message", ""),
        "sources": [{k: v for k, v in s.items() if k not in ("extracted_text", "b64")} for s in processed_sources],
        "contradiction_analysis": contradiction_analysis,
    }
