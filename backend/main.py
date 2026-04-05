import os
import re
from datetime import date, timedelta
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

load_dotenv()

from file_processor import extract_text, build_source_meta, RELIABILITY_SCORES
from pipeline import analyze_project

app = FastAPI(title="Project Pulse API")

# Allow all origins for hackathon demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    project_name: str
    deadline: str          # ISO date string: YYYY-MM-DD
    members: list[str]
    raw_chat: str
    tone: str = "diplomatic"
    sources: list[dict] = []   # pre-processed source objects from /upload


class CorrectRequest(BaseModel):
    project_name: str
    deadline: str
    members: list[str]
    raw_chat: str
    tone: str = "diplomatic"
    corrections: Optional[dict] = None  # overrides for tasks/owners/deadlines


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "Project Pulse API"}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    label: str = Form("auto"),
):
    """
    Upload a project resource file. Returns extracted content + metadata.
    label: official | reference | notes | auto
    """
    if label not in RELIABILITY_SCORES:
        label = "auto"
    file_bytes = await file.read()
    extracted_text, file_type = extract_text(file_bytes, file.filename)
    meta = build_source_meta(file.filename, file_type, label)
    return {
        **meta,
        "extracted_text": extracted_text,
        "char_count": len(extracted_text),
    }


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    result = await analyze_project(
        project_name=request.project_name,
        deadline=request.deadline,
        members=request.members,
        raw_chat=request.raw_chat,
        tone=request.tone,
        sources=request.sources or [],
    )
    return result


@app.post("/correct")
async def correct(request: CorrectRequest):
    """
    Apply user corrections to the analysis and re-run the pipeline.
    Corrections dict can contain overrides like:
      { "tasks": [...patched task objects...] }
    For MVP this re-runs the full pipeline; corrections are noted in the chat.
    """
    correction_notes = ""
    if request.corrections:
        correction_notes = f"\n[USER CORRECTIONS: {request.corrections}]"

    result = await analyze_project(
        project_name=request.project_name,
        deadline=request.deadline,
        members=request.members,
        raw_chat=request.raw_chat + correction_notes,
        tone=request.tone,
    )
    return result


# ---------------------------------------------------------------------------
# WhatsApp webhook (Twilio)
# Session state per sender phone number
# ---------------------------------------------------------------------------

_wa_sessions: dict[str, dict] = {}   # phone -> session dict

RISK_EMOJI = {"low": "🟢", "moderate": "🟡", "high": "🔴", "deadlocked": "🚨"}


def _wa_fmt_full(data: dict) -> str:
    level = (data.get("risk_level") or "unknown").lower()
    emoji = RISK_EMOJI.get(level, "⚪")
    score = data.get("risk_score", 0)
    tasks = data.get("tasks") or []
    blockers = data.get("blockers") or []
    actions = (data.get("next_actions") or [])[:3]
    msg = data.get("followup_message", "")

    lines = [
        f"📊 *Project Pulse — {data.get('project_name','')}*",
        f"Deadline: {data.get('deadline','')}",
        "",
        f"{emoji} *Risk: {level.upper()}* | Score: {score}",
        data.get("status_summary", ""),
        "",
        "*Tasks:*",
    ]
    dots = {"confirmed": "🟢", "tentative": "🟡", "unknown": "🔴"}
    for t in tasks:
        d = dots.get(t.get("owner_status", "unknown"), "🔴")
        lines.append(f"{d} {t.get('task_name','?')} — {t.get('owner') or 'Unassigned'}")

    lines += ["", "*Blockers:*"]
    if blockers:
        for b in blockers:
            lines.append(f"🔒 {b.get('blocker','?')}")
    else:
        lines.append("None")

    lines += ["", "*Next Actions:*"]
    for i, a in enumerate(actions, 1):
        lines.append(f"{i}. {a.get('action','?')}")

    lines += ["", "*Suggested Message:*", msg]
    return "\n".join(lines)


def _wa_fmt_part(data: dict, part: str) -> str:
    if part == "score":
        level = (data.get("risk_level") or "unknown").lower()
        return f"{RISK_EMOJI.get(level,'⚪')} Risk: {level.upper()} | Score: {data.get('risk_score',0)}\n{data.get('status_summary','')}"
    if part == "tasks":
        tasks = data.get("tasks") or []
        if not tasks:
            return "No tasks identified."
        dots = {"confirmed": "🟢", "tentative": "🟡", "unknown": "🔴"}
        lines = ["*Tasks:*"]
        for t in tasks:
            d = dots.get(t.get("owner_status", "unknown"), "🔴")
            lines.append(f"{d} {t.get('task_name','?')} — {t.get('owner') or 'Unassigned'} [{t.get('status','?')}]")
        return "\n".join(lines)
    if part == "blockers":
        blockers = data.get("blockers") or []
        if not blockers:
            return "✅ No blockers."
        return "*Blockers:*\n" + "\n".join(f"🔒 {b.get('blocker','?')}" for b in blockers)
    if part == "suggest":
        return f"✉️ *Suggested message:*\n\n{data.get('followup_message','No message.')}"
    return _wa_fmt_full(data)


def _wa_reply(to: str, body: str) -> None:
    """Send a WhatsApp message via Twilio REST API."""
    import urllib.request, urllib.parse
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_number = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    if not sid or not token:
        return  # no credentials configured
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    data = urllib.parse.urlencode({"From": from_number, "To": to, "Body": body}).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    import base64
    creds = base64.b64encode(f"{sid}:{token}".encode()).decode()
    req.add_header("Authorization", f"Basic {creds}")
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...),
):
    """
    Twilio WhatsApp webhook. Twilio POSTs form data here when a message arrives.
    Configure your Twilio sandbox webhook URL to: https://<your-host>/whatsapp/webhook
    """
    sender = From  # e.g. "whatsapp:+6512345678"
    text = Body.strip()
    text_lower = text.lower()

    session = _wa_sessions.setdefault(sender, {"state": "idle", "data": {}})
    state = session["state"]

    # ---- Quick commands on cached result ----
    QUICK = {"pulse score": "score", "pulse tasks": "tasks",
             "pulse blockers": "blockers", "pulse suggest": "suggest",
             "pulse overall": "overall"}

    if state == "idle":
        if text_lower in QUICK or text_lower == "pulse":
            part = QUICK.get(text_lower, "overall")
            cached = session.get("last_analysis")
            if cached:
                _wa_reply(sender, _wa_fmt_part(cached, part))
            else:
                _wa_reply(sender, "No analysis yet. Send *pulse analyze* to start.")
            return PlainTextResponse("OK")

        if text_lower in ("pulse analyze", "/pulse"):
            session["state"] = "ask_name"
            session["data"] = {}
            _wa_reply(sender, "🔍 *Project Pulse*\n\nWhat is the *project name*?")
            return PlainTextResponse("OK")

        # Unknown — show help
        if "pulse" in text_lower:
            _wa_reply(
                sender,
                "👋 *Project Pulse commands:*\n"
                "• *pulse analyze* — start full analysis\n"
                "• *pulse score* — last risk score\n"
                "• *pulse tasks* — last task list\n"
                "• *pulse blockers* — last blockers\n"
                "• *pulse suggest* — last follow-up message\n"
                "• *pulse overall* — full last result",
            )
        return PlainTextResponse("OK")

    # ---- Conversation flow ----
    if text_lower == "cancel":
        session["state"] = "idle"
        _wa_reply(sender, "Cancelled.")
        return PlainTextResponse("OK")

    d = session["data"]

    if state == "ask_name":
        d["project_name"] = text
        session["state"] = "ask_deadline"
        _wa_reply(sender, "📅 What is the *deadline*? (YYYY-MM-DD)")

    elif state == "ask_deadline":
        if not re.match(r"\d{4}-\d{2}-\d{2}", text):
            _wa_reply(sender, "Please use YYYY-MM-DD format, e.g. 2026-04-10")
        else:
            d["deadline"] = text
            session["state"] = "ask_members"
            _wa_reply(sender, "👥 Who are the *teammates*? (comma-separated)")

    elif state == "ask_members":
        members = [m.strip() for m in text.split(",") if m.strip()]
        if not members:
            _wa_reply(sender, "Please enter at least one name.")
        else:
            d["members"] = members
            session["state"] = "ask_chat"
            _wa_reply(
                sender,
                "💬 Now *paste the group chat or meeting notes*.\n\n"
                "Format: Name: message (one per line)\n"
                "Optionally add on first line: tone: casual\n"
                "(options: casual, diplomatic, direct, formal)"
            )

    elif state == "ask_chat":
        tone = "diplomatic"
        lines = text.split("\n")
        if lines and re.match(r"^tone\s*:", lines[0], re.IGNORECASE):
            tone_val = lines[0].split(":", 1)[1].strip().lower()
            if tone_val in ("casual", "diplomatic", "direct", "formal"):
                tone = tone_val
            text = "\n".join(lines[1:]).strip()

        d["raw_chat"] = text
        d["tone"] = tone
        session["state"] = "idle"

        _wa_reply(sender, "⏳ Analyzing… please wait.")
        try:
            result = await analyze_project(
                project_name=d["project_name"],
                deadline=d["deadline"],
                members=d["members"],
                raw_chat=d["raw_chat"],
                tone=d["tone"],
            )
            session["last_analysis"] = result
            reply = _wa_fmt_full(result)
            # Split into chunks if too long
            for i in range(0, len(reply), 1500):
                _wa_reply(sender, reply[i:i + 1500])
        except Exception as e:
            _wa_reply(sender, f"❌ Error: {e}")

    return PlainTextResponse("OK")


@app.get("/whatsapp/webhook")
async def whatsapp_verify(request: Request):
    """Twilio webhook URL verification (GET)."""
    return PlainTextResponse("OK")
