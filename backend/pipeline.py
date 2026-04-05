import json
import os
import re
from datetime import date, datetime

from openai import OpenAI

from parser import parse_chat
from prompts import (
    CONTEXT_EXTRACTION_PROMPT,
    CONTRADICTION_PROMPT,
    DEADLOCK_PROMPT,
    EXTRACTION_PROMPT,
    FOLLOWUP_PROMPT,
    NORMALIZATION_PROMPT,
)
from scorer import compute_risk, risk_band

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_mock_mode() -> bool:
    if os.environ.get("MOCK_MODE", "").lower() in ("1", "true", "yes"):
        return True
    return not bool(os.environ.get("OPENCLAW_API_KEY", "").strip())


def _safe_json(text: str, default):
    """
    Robustly extract JSON from LLM output using multiple strategies.

    Handles all of these real Agnes response formats:
      - Clean JSON                          {"key": ...}
      - Code-fenced anywhere in prose       ```json\n{...}\n```
      - JSON buried after introductory text Sure! Here is the result:\n{...}
      - Trailing commentary after JSON      {...}\n\nHope that helps!
    """
    if not text:
        return default

    text = text.strip()

    # Strategy 1: direct parse (ideal case)
    try:
        return json.loads(text)
    except Exception:
        pass

    # Strategy 2: extract from code fence anywhere in the response
    fence = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text, re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except Exception:
            pass

    # Strategy 3: find the outermost { ... } block (handles prose before/after)
    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return default


def _make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENCLAW_API_KEY", "mock"),
        base_url=os.environ.get("OPENCLAW_BASE_URL", "https://zenmux.ai/api/v1"),
    )


def _call_llm(client: OpenAI, system: str, user: str) -> str:
    """Call the OpenClaw/Agnes API and return the text content."""
    model = os.environ.get("OPENCLAW_MODEL", "sapiens-ai/agnes-1.5-pro")
    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def _call_vision(client: OpenAI, prompt: str, mime: str, b64: str) -> str:
    """Call the LLM with an image attachment (vision). Falls back gracefully."""
    model = os.environ.get("OPENCLAW_MODEL", "agnes-1.5-pro")
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Vision call failed: {e}. Please describe the image content manually.]"


def _process_source(client: OpenAI, source: dict) -> dict:
    """
    Extract structured context from a single uploaded source.
    Handles text-based files and images (via vision LLM).
    Returns source dict enriched with 'context' and 'readable_text' keys.
    """
    file_type = source.get("file_type", "text")
    raw_content = source.get("extracted_text", "")
    filename = source.get("filename", "document")

    # For images: run vision call first to get readable text
    if file_type == "image":
        try:
            img_data = json.loads(raw_content)
            readable_text = _call_vision(
                client,
                "Extract ALL text, requirements, deadlines, task descriptions, and key information "
                "from this image. Be thorough and precise.",
                img_data["mime"],
                img_data["b64"],
            )
        except Exception:
            readable_text = "[Image: could not extract text]"
    elif file_type == "video":
        readable_text = raw_content  # already a guidance message
    else:
        readable_text = raw_content

    # Run context extraction on the readable text
    context_user = json.dumps({
        "filename": filename,
        "file_type": file_type,
        "reliability_label": source.get("reliability_label", "auto"),
        "content": readable_text[:8000],  # cap to avoid token overflow
    })
    context_raw = _call_llm(client, CONTEXT_EXTRACTION_PROMPT, context_user)
    context = _safe_json(context_raw, {
        "source_summary": readable_text[:200],
        "document_type": "unknown",
        "deliverables": [],
        "requirements": [],
        "deadlines": [],
        "evaluation_criteria": [],
        "constraints": [],
        "key_topics": [],
        "uncertainty_notes": [],
    })

    return {
        **source,
        "readable_text": readable_text,
        "context": context,
    }


def _detect_contradictions(
    client: OpenAI,
    processed_sources: list[dict],
    extracted_chat: dict,
) -> dict:
    """Compare document requirements against chat-derived project state."""
    sources_summary = [
        {
            "filename": s.get("filename"),
            "reliability_score": s.get("reliability_score", 6),
            "reliability_label": s.get("reliability_label", "auto"),
            "context": s.get("context", {}),
        }
        for s in processed_sources
    ]
    contradiction_user = json.dumps({
        "document_sources": sources_summary,
        "chat_state": {
            "tasks": extracted_chat.get("tasks", []),
            "decisions": extracted_chat.get("decisions", []),
            "blockers": extracted_chat.get("blockers", []),
            "communication_flags": extracted_chat.get("communication_flags", []),
        },
    })
    raw = _call_llm(client, CONTRADICTION_PROMPT, contradiction_user)
    return _safe_json(raw, {
        "contradictions": [],
        "requirement_gaps": [],
        "source_conflicts": [],
        "overall_alignment": "aligned",
        "alignment_summary": "No document sources provided — analysis based on chat only.",
    })


# ---------------------------------------------------------------------------
# Bug 1 fix: heuristic fallback when LLM normalization returns bad/empty JSON
# ---------------------------------------------------------------------------

_UNCERTAINTY   = {"i think", "maybe", "probably", "not sure", "idk", "i see first",
                  "ok i see", "see first", "i guess", "perhaps", "prob"}
_BLOCKER       = {"haven", "haven't", "waiting", "can't", "cannot", "stuck",
                  "blocked", "need", "need first", "not sent", "haven send"}
_OWNERSHIP     = {"i thought", "i tot", "who is", "who's doing", "someone should",
                  "u doing", "you doing", "i thought u", "tot u"}
_WEAK_COMMIT   = {"see first", "ok i see", "maybe later", "i try", "will see",
                  "prob", "probably", "idk", "later"}
_TASK_WORDS    = {"do", "doing", "done", "finish", "complete", "send", "compile",
                  "slides", "report", "write", "prepare", "submit", "survey", "results"}
_DEADLINE_WORDS = {"tmr", "tomorrow", "tonight", "today", "deadline", "by when",
                   "asap", "settle", "due"}
_DEP_WORDS     = {"need", "waiting", "first", "before", "after", "results", "then",
                  "once", "until"}
_SINGLISH      = {"lah", "leh", "lor", "ah", "bro", "sia", "leh", "hor"}


def _build_fallback_normalized(parsed_messages: list, members: list) -> list:
    """
    Bug 1 fix: keyword-heuristic normalization when the LLM fails.
    Produces usable normalized messages so extraction has real input.
    Singlish patterns are explicitly covered.
    """
    member_lower = {m.lower() for m in members}
    result = []
    for msg in parsed_messages:
        raw = msg["raw_text"]
        t = raw.lower()
        words = set(t.split())

        has_uncertainty   = bool(any(p in t for p in _UNCERTAINTY) or words & _SINGLISH)
        has_blocker       = bool(any(p in t for p in _BLOCKER))
        has_ownership     = bool(any(p in t for p in _OWNERSHIP))
        has_task          = bool(words & _TASK_WORDS)
        has_deadline      = bool(any(p in t for p in _DEADLINE_WORDS))
        has_dependency    = bool(any(p in t for p in _DEP_WORDS))
        has_commitment    = bool(any(w in t for w in ("i will", "i'll", "will do",
                                                       "i confirm", "confirmed", "done")))
        has_owner_ref     = bool(has_ownership or words & member_lower)
        has_weak_commit   = bool(any(p in t for p in _WEAK_COMMIT))

        # Build a plain-English normalized version
        parts = []
        if has_blocker:
            parts.append("The speaker is blocked or waiting on something.")
        if has_ownership:
            parts.append("There is confusion about who owns a task.")
        if has_weak_commit:
            parts.append("The speaker gives a weak, non-committal response.")
        if has_task and not parts:
            parts.append("The speaker references a task.")
        if has_dependency:
            parts.append("Progress depends on another person or deliverable.")
        if not parts:
            parts.append(f"Informal message: \"{raw}\"")

        result.append({
            "speaker": msg["speaker"],
            "raw_text": raw,
            "normalized_text": " ".join(parts),
            "confidence": "low",
            "signals": {
                "contains_task_reference":     has_task,
                "contains_owner_reference":    has_owner_ref,
                "contains_deadline_reference": has_deadline,
                "contains_blocker_reference":  has_blocker,
                "contains_uncertainty":        has_uncertainty or has_weak_commit,
                "contains_commitment":         has_commitment,
                "contains_dependency":         has_dependency,
            },
        })
    return result


def _build_heuristic_flags(normalized_messages: list) -> list:
    """
    Bug 2 fix: derive communication_flags from signal counts when extraction
    returns empty — so the scorer still has something meaningful to work with.
    """
    flags = []
    n = len(normalized_messages)
    if n == 0:
        return flags

    uncertainty_count = sum(1 for m in normalized_messages
                            if m.get("signals", {}).get("contains_uncertainty"))
    blocker_count     = sum(1 for m in normalized_messages
                            if m.get("signals", {}).get("contains_blocker_reference"))
    owner_count       = sum(1 for m in normalized_messages
                            if m.get("signals", {}).get("contains_owner_reference"))
    commit_count      = sum(1 for m in normalized_messages
                            if m.get("signals", {}).get("contains_commitment"))
    dep_count         = sum(1 for m in normalized_messages
                            if m.get("signals", {}).get("contains_dependency"))

    if uncertainty_count >= max(1, n // 2):
        flags.append({
            "flag": "weak_commitment",
            "description": f"{uncertainty_count} of {n} messages contain uncertain or non-committal language.",
            "involved_members": [],
        })
    if owner_count > 0 and commit_count == 0:
        flags.append({
            "flag": "ownership_confusion",
            "description": "Team members are referenced but no one has explicitly confirmed ownership of any task.",
            "involved_members": [],
        })
    if blocker_count > 0:
        flags.append({
            "flag": "dependency_risk",
            "description": f"{blocker_count} message(s) indicate blocked progress or unmet dependencies.",
            "involved_members": [],
        })
    if dep_count > 0:
        flags.append({
            "flag": "dependency_risk",
            "description": "Messages reference waiting on others or sequential dependencies.",
            "involved_members": [],
        })
    return flags


def _heuristic_extract(parsed_messages: list, members: list) -> dict:
    """
    Local keyword-based extraction fallback used when Agnes fails completely.
    Produces real tasks, blockers, decisions, and communication_flags so
    the UI tabs are never empty after a total LLM failure.
    """
    tasks: list = []
    blockers: list = []
    decisions: list = []
    task_counter = 0

    # Commitment → task: speaker declares they will do something
    _COMMIT_PATS = [
        r"\bi(?:'ll| will| am| 'm)\s+(?:do|doing|compil|send|writ|prepar|submit|finish|handl|work)",
        r"\bwill\s+(?:do|handle|send|finish|prepare|compile|write|submit)\b",
        r"\bi(?:'ve| have)\s+(?:done|finished|sent|submitted|completed)\b",
    ]
    # Blocker: something is waiting, can't proceed, or hasn't been done
    _BLOCK_PATS = [
        r"\bhaven[''']?t?\s+(?:send|sent|done|finish|submit|upload)",
        r"\bwaiting\s+(?:for|on)\b",
        r"\bcan[''']?t\s+(?:do|continue|proceed|finish|start)\b",
        r"\bnot\s+(?:yet|done|sent|received)\b",
        r"\bblocked\b",
        r"\bstuck\b",
        r"\bneed\s+\w+\s+(?:first|before)\b",
    ]
    # Unresolved decision: ownership confusion or scope question
    _DECISION_PATS = [
        r"\bi\s+(?:thought|tot)\s+(?:u|you|\w+)\s+(?:was|were|is|will|would)\s+(?:do|doing|handl|compil)",
        r"\bwho[''']?s?\s+(?:doing|handling|responsible\s+for|in\s+charge)",
        r"\bwho\s+(?:is|will|should)\s+(?:do|send|compil|write|prepar|submit)",
        r"\bsomeone\s+(?:should|need|must)\s+(?:do|handle|take\s+care)",
        r"\bwhat\s+are\s+we\s+(?:using|doing|going\s+with)\b",
    ]

    seen_tasks: set = set()
    seen_blockers: set = set()
    seen_decisions: set = set()

    for msg in parsed_messages:
        raw = msg["raw_text"]
        speaker = msg["speaker"]
        t = raw.lower()

        # --- Task extraction ---
        for pat in _COMMIT_PATS:
            m = re.search(pat, t)
            if m:
                rest = t[m.end():].strip().split()
                task_frag = " ".join(rest[:4]).strip(" .,!?")
                if not task_frag or task_frag in {"a", "the", "it", "ok", "sure"}:
                    task_frag = "task"
                task_name = task_frag.capitalize()
                key = f"{speaker.lower()}:{task_name.lower()}"
                if key not in seen_tasks:
                    seen_tasks.add(key)
                    task_counter += 1
                    tasks.append({
                        "task_id": f"h{task_counter}",
                        "task_name": task_name,
                        "owner": speaker,
                        "owner_status": "tentative",
                        "deadline": None,
                        "deadline_status": "unknown",
                        "status": "in_progress",
                        "depends_on": [],
                        "evidence": [f"{speaker}: {raw}"],
                    })
                break

        # --- Blocker extraction ---
        for pat in _BLOCK_PATS:
            if re.search(pat, t):
                key = f"{speaker.lower()}:{t[:30]}"
                if key not in seen_blockers:
                    seen_blockers.add(key)
                    blockers.append({
                        "blocker": f"Blocked/waiting: \"{raw}\"",
                        "blocking_task": None,
                        "responsible_party": speaker,
                        "severity": "medium",
                        "evidence": [f"{speaker}: {raw}"],
                    })
                break

        # --- Unresolved decision / ownership confusion ---
        for pat in _DECISION_PATS:
            if re.search(pat, t):
                key = f"{speaker.lower()}:{t[:30]}"
                if key not in seen_decisions:
                    seen_decisions.add(key)
                    decisions.append({
                        "decision": f"Unresolved: \"{raw}\"",
                        "status": "unresolved",
                        "evidence": [f"{speaker}: {raw}"],
                    })
                break

    # If blockers were found but no tasks, synthesize blocked tasks from blockers
    if blockers and not tasks:
        _TASK_NOUNS = ["slides", "report", "survey", "write", "compile", "send",
                       "submit", "prepare", "finish", "design", "code"]
        for i, b in enumerate(blockers[:3], 1):
            evid = (b["evidence"][0] if b["evidence"] else "").lower()
            noun = next((tw for tw in _TASK_NOUNS if tw in evid), "task")
            tasks.append({
                "task_id": f"h{i}",
                "task_name": noun.capitalize() + " (blocked)",
                "owner": None,
                "owner_status": "unknown",
                "deadline": None,
                "deadline_status": "unknown",
                "status": "blocked",
                "depends_on": [],
                "evidence": b["evidence"],
            })

    # Build communication flags via existing heuristics
    norm = _build_fallback_normalized(parsed_messages, members)
    flags = _build_heuristic_flags(norm)

    return {
        "tasks": tasks,
        "blockers": blockers,
        "decisions": decisions,
        "questions": [d["decision"] for d in decisions[:3]],
        "communication_flags": flags,
    }


# ---------------------------------------------------------------------------
# Mock response — used when OPENCLAW_API_KEY is not set or MOCK_MODE=true
# ---------------------------------------------------------------------------

def _mock_analyze(project_name: str, deadline: str, members: list, raw_chat: str, tone: str) -> dict:
    parsed = parse_chat(raw_chat, members)
    days_to_deadline = _days_to_deadline(deadline)

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
         "responsible_party": "Ben", "severity": "high",
         "evidence": ["Alex: ben haven send survey leh i cant do slides"]},
        {"blocker": "Final project angle is undecided — blocks all content work", "blocking_task": "Prepare slides",
         "responsible_party": None, "severity": "high",
         "evidence": ["Dion: what are we even using for final angle"]},
    ]
    decisions = [
        {"decision": "Who is responsible for compiling the survey results?", "status": "unresolved",
         "evidence": ["Ben: i tot alex compiling", "Alex: ben haven send survey leh"]},
        {"decision": "What is the final project angle or thesis?", "status": "unresolved",
         "evidence": ["Dion: what are we even using for final angle"]},
    ]
    communication_flags = [
        {"flag": "ownership_confusion", "description": "Both Alex and Ben believe the other person is responsible for compiling survey results.", "involved_members": ["Alex", "Ben"]},
        {"flag": "weak_commitment", "description": "Ben responded 'ok i see first' — a non-committal response with no clear action.", "involved_members": ["Ben"]},
        {"flag": "vague_deadline", "description": "Cara proposed 'tmr settle' without specifying time or owner.", "involved_members": ["Cara"]},
        {"flag": "scope_confusion", "description": "The final project angle is unresolved, blocking content work.", "involved_members": ["Dion"]},
    ]

    rule_score = compute_risk(tasks, blockers, decisions, communication_flags, days_to_deadline)
    rule_level = risk_band(rule_score)

    top_causes = [
        {"cause": "Ownership confusion over survey compilation", "severity": "high",
         "why_it_matters": "Alex and Ben each believe the other is responsible. No one is actually doing it."},
        {"cause": "Final project angle unresolved", "severity": "high",
         "why_it_matters": "All content work depends on agreeing on the angle first. Dion raised this but got no answer."},
        {"cause": "Weak commitment from Ben", "severity": "medium",
         "why_it_matters": "'Ok I see first' does not confirm any action will be taken."},
    ]
    clarifications = [
        "Who is compiling the survey results — Alex or Ben?",
        "What is the final angle or thesis for the project?",
        "What is the check-in deadline for resolving these two items?",
    ]
    next_actions = [
        {"priority": 1, "action": "Ben and Alex must confirm right now who is sending/compiling the survey results.", "owner": "Ben", "deadline": "Today"},
        {"priority": 2, "action": "Team to agree on the final project angle before any more content is written.", "owner": "Dion", "deadline": "Today"},
        {"priority": 3, "action": "Alex to resume slides once survey results are confirmed.", "owner": "Alex", "deadline": deadline},
    ]

    tone_messages = {
        "diplomatic": f"Hey everyone, just want to make sure we're aligned before we get too far. Two things need to be sorted: (1) who is actually compiling the survey results — Alex and Ben, can you two confirm between yourselves? And (2) Dion raised a good point about the final angle — can we all agree on that today? Once those are cleared, Alex can continue the slides. Can we lock both by 9pm tonight?",
        "direct": f"Two blockers need to be resolved now: (1) Ben or Alex — who is sending the survey results? Decide and confirm. (2) What is the final angle? Dion flagged this, team needs to answer. Nothing else can move until these are done. Reply by 9pm.",
        "formal": f"Hi team, to ensure we remain on track for the {deadline} deadline, I would like to flag two unresolved items: first, the ownership of survey result compilation between Alex and Ben; second, the undecided project angle raised by Dion. Could both items be confirmed by 9pm today so that Alex can proceed with the slides? Thank you.",
        "casual": f"hey guys, quick check — who's doing the survey results, alex or ben? also dion's q about the final angle is still hanging. can we sort both by 9pm tonight? alex is waiting on both to do the slides 🙏",
    }

    # Mock contradiction data — shown when sources are uploaded
    mock_contradictions = {
        "contradictions": [],
        "requirement_gaps": [],
        "source_conflicts": [],
        "overall_alignment": "aligned",
        "alignment_summary": "No document sources uploaded — analysis based on chat only.",
    }

    return {
        "project_name": project_name,
        "deadline": deadline,
        "members": members,
        "days_to_deadline": days_to_deadline,
        "normalized_messages": [
            {"speaker": m["speaker"], "raw_text": m["raw_text"],
             "normalized_text": f"[MOCK] {m['raw_text']}", "confidence": "medium",
             "signals": {"contains_task_reference": True, "contains_owner_reference": False,
                         "contains_deadline_reference": False, "contains_blocker_reference": False,
                         "contains_uncertainty": True, "contains_commitment": False, "contains_dependency": False}}
            for m in parsed
        ],
        "tasks": tasks,
        "decisions": decisions,
        "questions": ["Who is compiling the survey results?", "What is the final project angle?"],
        "blockers": blockers,
        "communication_flags": communication_flags,
        "rule_based_score": rule_score,
        "rule_based_level": rule_level,
        "risk_score": rule_score,
        "risk_level": rule_level,
        "status_summary": f"[MOCK MODE] The project is at {rule_level} risk. Two critical blockers exist: unresolved ownership of survey compilation and an undecided final angle. No progress can be made on slides until both are resolved.",
        "top_causes": top_causes,
        "clarifications_needed": clarifications,
        "next_actions": next_actions,
        "followup_message": tone_messages.get(tone, tone_messages["diplomatic"]),
        "sources": [],
        "contradiction_analysis": mock_contradictions,
    }


def _days_to_deadline(deadline_str: str) -> int:
    """Compute days from today to the given deadline string."""
    try:
        deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d").date()
        delta = (deadline_date - date.today()).days
        return max(delta, 0)
    except Exception:
        return 7  # safe default


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

    # 1. Parse chat
    parsed_messages = parse_chat(raw_chat, members)

    # 2. Build initial data model
    data_model = {
        "project_name": project_name,
        "deadline": deadline,
        "members": members,
        "raw_messages": parsed_messages,
    }

    # 3. Normalization
    norm_user = json.dumps({
        "members": members,
        "messages": parsed_messages,
    })
    norm_raw = _call_llm(client, NORMALIZATION_PROMPT, norm_user)
    norm_result = _safe_json(norm_raw, {"normalized_messages": []})
    normalized_messages = norm_result.get("normalized_messages", [])

    # Bug 1 fix: if LLM returned bad/empty JSON, use heuristic fallback
    if not normalized_messages and parsed_messages:
        normalized_messages = _build_fallback_normalized(parsed_messages, members)

    # 4. Extraction
    ext_user = json.dumps({"normalized_messages": normalized_messages})
    ext_raw = _call_llm(client, EXTRACTION_PROMPT, ext_user)
    extracted = _safe_json(ext_raw, None)

    # Bug 2 fix: retry once with an explicit nudge if extraction returned empty
    extraction_empty = (
        not extracted
        or (not extracted.get("tasks") and not extracted.get("blockers")
            and not extracted.get("decisions"))
    )
    if extraction_empty:
        retry_user = (
            ext_user.rstrip("}")
            + ', "nudge": "The chat is informal or uses slang. Extract ALL tasks, blockers, '
            'decisions, and communication flags even if ownership or deadlines are only implied. '
            'Do not return empty arrays if the conversation contains coordination activity."}'
        )
        ext_raw2 = _call_llm(client, EXTRACTION_PROMPT, retry_user)
        extracted = _safe_json(ext_raw2, {
            "tasks": [], "decisions": [], "questions": [],
            "blockers": [], "communication_flags": [],
        })

    tasks              = extracted.get("tasks", [])
    blockers           = extracted.get("blockers", [])
    decisions          = extracted.get("decisions", [])
    communication_flags = extracted.get("communication_flags", [])

    # Bug 2 fix (cont): if still empty, use full heuristic extraction as final fallback
    if not tasks and not blockers and not decisions and not communication_flags:
        heuristic = _heuristic_extract(parsed_messages, members)
        tasks               = heuristic["tasks"]
        blockers            = heuristic["blockers"]
        decisions           = heuristic["decisions"]
        communication_flags = heuristic["communication_flags"]

    # 5. Process uploaded sources (if any)
    processed_sources = []
    contradiction_analysis = {
        "contradictions": [],
        "requirement_gaps": [],
        "source_conflicts": [],
        "overall_alignment": "aligned",
        "alignment_summary": "No document sources uploaded — analysis based on chat only.",
    }

    if sources:
        # Sort by reliability descending so highest-authority sources are processed first
        sorted_sources = sorted(sources, key=lambda s: s.get("reliability_score", 6), reverse=True)
        for src in sorted_sources:
            processed = _process_source(client, src)
            processed_sources.append(processed)

        # Contradiction detection — compare docs vs chat state
        contradiction_analysis = _detect_contradictions(client, processed_sources, extracted)

        # Elevate risk score if contradictions are found
        high_contradictions = [c for c in contradiction_analysis.get("contradictions", [])
                               if c.get("severity") == "high"]
        if high_contradictions:
            extracted["communication_flags"] = communication_flags + [
                {"flag": "scope_confusion",
                 "description": f"{len(high_contradictions)} high-severity contradiction(s) with uploaded documents.",
                 "involved_members": []}
            ]
            communication_flags = extracted["communication_flags"]

    # 6. Compute days to deadline
    days_to_deadline = _days_to_deadline(deadline)

    # 7. Rule-based risk score
    rule_score = compute_risk(tasks, blockers, decisions, communication_flags, days_to_deadline)

    # Bug 3 fix: a chat with messages but zero extracted structure is itself a red flag.
    # Enforce a minimum score floor so the scorer doesn't silently report "healthy".
    if not tasks and not blockers and len(parsed_messages) >= 3:
        if days_to_deadline <= 5:
            rule_score = max(rule_score, 9)   # floor at "high"
        elif days_to_deadline <= 7:
            rule_score = max(rule_score, 5)   # floor at "moderate"

    rule_level = risk_band(rule_score)

    # 8. Deadlock analysis — include contradiction context if available
    deadlock_user = json.dumps({
        "project_name": project_name,
        "deadline": deadline,
        "days_to_deadline": days_to_deadline,
        "tasks": tasks,
        "blockers": blockers,
        "decisions": decisions,
        "communication_flags": communication_flags,
        "rule_based_risk_score": rule_score,
        "rule_based_risk_level": rule_level,
        "document_contradictions": contradiction_analysis.get("contradictions", []),
        "requirement_gaps": contradiction_analysis.get("requirement_gaps", []),
    })
    deadlock_raw = _call_llm(client, DEADLOCK_PROMPT, deadlock_user)
    deadlock_result = _safe_json(deadlock_raw, {
        "risk_score": rule_score,
        "risk_level": rule_level,
        "status_summary": "Unable to generate analysis.",
        "top_causes": [],
        "clarifications_needed": [],
        "next_actions": [],
    })

    # 9. Follow-up message
    followup_user = json.dumps({
        "project_name": project_name,
        "deadline": deadline,
        "tone": tone,
        "top_causes": deadlock_result.get("top_causes", []),
        "clarifications_needed": deadlock_result.get("clarifications_needed", []),
        "next_actions": deadlock_result.get("next_actions", []),
        "members": members,
        "critical_contradictions": [c for c in contradiction_analysis.get("contradictions", [])
                                    if c.get("severity") == "high"],
    })
    followup_raw = _call_llm(client, FOLLOWUP_PROMPT, followup_user)
    followup_result = _safe_json(followup_raw, {"message": "Could not generate follow-up message."})

    # Bug 4 fix: the rule engine is the floor — the LLM can only raise the score, never lower it.
    final_score = max(rule_score, deadlock_result.get("risk_score", 0))
    final_level = risk_band(final_score)

    # 10. Return combined result
    return {
        "project_name": project_name,
        "deadline": deadline,
        "members": members,
        "days_to_deadline": days_to_deadline,
        "normalized_messages": normalized_messages,
        "tasks": tasks,
        "decisions": decisions,
        "questions": extracted.get("questions", []),
        "blockers": blockers,
        "communication_flags": communication_flags,
        "rule_based_score": rule_score,
        "rule_based_level": rule_level,
        "risk_score": final_score,
        "risk_level": final_level,
        "status_summary": deadlock_result.get("status_summary", ""),
        "top_causes": deadlock_result.get("top_causes", []),
        "clarifications_needed": deadlock_result.get("clarifications_needed", []),
        "next_actions": deadlock_result.get("next_actions", []),
        "followup_message": followup_result.get("message", ""),
        "sources": [
            {k: v for k, v in s.items() if k not in ("extracted_text", "b64")}
            for s in processed_sources
        ],
        "contradiction_analysis": contradiction_analysis,
    }
