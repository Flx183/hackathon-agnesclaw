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


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def _safe_json(text: str, default):
    """Parse JSON robustly, returning default on failure."""
    try:
        return json.loads(_strip_fences(text))
    except Exception:
        return default


def _make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENCLAW_API_KEY", "mock"),
        base_url=os.environ.get("OPENCLAW_BASE_URL", "http://localhost:18789/v1"),
    )


def _call_llm(client: OpenAI, system: str, user: str) -> str:
    """Call the OpenClaw/Agnes API and return the text content."""
    model = os.environ.get("OPENCLAW_MODEL", "agnes-1.5-pro")
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

    # 4. Extraction
    ext_user = json.dumps({"normalized_messages": normalized_messages})
    ext_raw = _call_llm(client, EXTRACTION_PROMPT, ext_user)
    extracted = _safe_json(ext_raw, {
        "tasks": [],
        "decisions": [],
        "questions": [],
        "blockers": [],
        "communication_flags": [],
    })

    tasks = extracted.get("tasks", [])
    blockers = extracted.get("blockers", [])
    decisions = extracted.get("decisions", [])
    communication_flags = extracted.get("communication_flags", [])

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
        "risk_score": deadlock_result.get("risk_score", rule_score),
        "risk_level": deadlock_result.get("risk_level", rule_level),
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
