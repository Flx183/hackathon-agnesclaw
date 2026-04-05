"""
Project Pulse — Telegram Bot
Run with: python bot_telegram.py

Commands (DM or group):
  /pulse    — start full analysis (conversation flow)
  /score    — risk score from last analysis
  /tasks    — task list from last analysis
  /blockers — blockers from last analysis
  /suggest  — follow-up message from last analysis
  /cancel   — cancel ongoing analysis
  /help     — show commands

In groups, also responds to:
  pulse score | pulse tasks | pulse blockers | pulse suggest
"""

import asyncio
import os
import re
import sys
from datetime import date

from dotenv import load_dotenv
load_dotenv()

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# Import pipeline directly — no HTTP overhead
from pipeline import analyze_project

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Conversation states
ASK_NAME, ASK_DEADLINE, ASK_MEMBERS, ASK_CHAT, ASK_TONE = range(5)

# Cache: chat_id -> last analysis result
_cache: dict[int, dict] = {}

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

RISK_EMOJI = {
    "low": "🟢",
    "moderate": "🟡",
    "high": "🔴",
    "deadlocked": "🚨",
}

def _esc(text: str) -> str:
    """Escape HTML special chars."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_risk(data: dict) -> str:
    level = (data.get("risk_level") or "unknown").lower()
    emoji = RISK_EMOJI.get(level, "⚪")
    score = data.get("risk_score", 0)
    summary = data.get("status_summary", "")
    return (
        f"{emoji} <b>Risk Level: {level.upper()}</b>  |  Score: <b>{score}</b>\n"
        f"<i>{_esc(summary)}</i>"
    )


def fmt_tasks(data: dict) -> str:
    tasks = data.get("tasks") or []
    if not tasks:
        return "No tasks identified."
    lines = ["<b>📋 Tasks</b>"]
    for t in tasks:
        owner = t.get("owner") or "Unassigned"
        status = (t.get("status") or "unclear").replace("_", " ")
        os_ = t.get("owner_status", "unknown")
        dot = "🟢" if os_ == "confirmed" else "🟡" if os_ == "tentative" else "🔴"
        lines.append(f"{dot} <b>{_esc(t.get('task_name','?'))}</b> — {_esc(owner)} [{_esc(status)}]")
    return "\n".join(lines)


def fmt_blockers(data: dict) -> str:
    blockers = data.get("blockers") or []
    if not blockers:
        return "✅ No blockers identified."
    lines = ["<b>🔒 Blockers</b>"]
    sev_emoji = {"high": "🔴", "medium": "🟡", "low": "⚪"}
    for b in blockers:
        sev = b.get("severity", "low")
        party = b.get("responsible_party") or "No owner"
        lines.append(f"{sev_emoji.get(sev,'⚪')} {_esc(b.get('blocker','?'))} <i>({_esc(party)})</i>")
    return "\n".join(lines)


def fmt_actions(data: dict) -> str:
    actions = (data.get("next_actions") or [])[:3]
    if not actions:
        return "No actions identified."
    lines = ["<b>⚡ Top Next Actions</b>"]
    for i, a in enumerate(actions, 1):
        owner = a.get("owner") or ""
        dl = a.get("deadline") or ""
        meta = f" <i>— {_esc(owner)}{', by ' + _esc(dl) if dl else ''}</i>" if owner or dl else ""
        lines.append(f"{i}. {_esc(a.get('action','?'))}{meta}")
    return "\n".join(lines)


def fmt_suggest(data: dict) -> str:
    msg = data.get("followup_message") or "No message generated."
    return f"<b>✉️ Ready-to-Send Message</b>\n\n<code>{_esc(msg)}</code>"


def fmt_full(data: dict) -> str:
    sections = [
        f"<b>📊 Project Pulse — {_esc(data.get('project_name',''))}</b>",
        f"Deadline: {_esc(data.get('deadline',''))}  |  Members: {_esc(', '.join(data.get('members') or []))}",
        "",
        fmt_risk(data),
        "",
        fmt_tasks(data),
        "",
        fmt_blockers(data),
        "",
        fmt_actions(data),
        "",
        fmt_suggest(data),
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# /start  /help
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_html(
        "👋 <b>Project Pulse</b> — AI coordination failure detector\n\n"
        "<b>Commands:</b>\n"
        "/pulse — Analyze a group project chat\n"
        "/score — Risk score from last analysis\n"
        "/tasks — Task list from last analysis\n"
        "/blockers — Blockers from last analysis\n"
        "/suggest — Follow-up message from last analysis\n"
        "/cancel — Cancel ongoing flow\n\n"
        "<i>In groups: type</i> <code>pulse score</code>, <code>pulse tasks</code>, "
        "<code>pulse suggest</code> <i>for quick results.</i>"
    )


# ---------------------------------------------------------------------------
# Conversation flow — /pulse
# ---------------------------------------------------------------------------

async def cmd_pulse(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "🔍 Let's analyze your project.\n\nWhat is the <b>project name</b>?",
        parse_mode=ParseMode.HTML,
    )
    return ASK_NAME


async def got_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["project_name"] = update.message.text.strip()
    await update.message.reply_text(
        "📅 What is the <b>deadline</b>? (format: YYYY-MM-DD)",
        parse_mode=ParseMode.HTML,
    )
    return ASK_DEADLINE


async def got_deadline(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = update.message.text.strip()
    # Basic validation
    if not re.match(r"\d{4}-\d{2}-\d{2}", raw):
        await update.message.reply_text("Please use YYYY-MM-DD format, e.g. 2026-04-10")
        return ASK_DEADLINE
    context.user_data["deadline"] = raw
    await update.message.reply_text(
        "👥 Who are the <b>teammates</b>? (comma-separated, e.g. Alex, Ben, Cara)",
        parse_mode=ParseMode.HTML,
    )
    return ASK_MEMBERS


async def got_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    members = [m.strip() for m in update.message.text.split(",") if m.strip()]
    if not members:
        await update.message.reply_text("Please enter at least one teammate name.")
        return ASK_MEMBERS
    context.user_data["members"] = members
    await update.message.reply_text(
        "💬 Now <b>paste the group chat or meeting notes</b>.\n\n"
        "Format each line as: <code>Name: message</code>\n"
        "Also add your preferred tone on the first line if you want:\n"
        "<code>tone: casual</code> (options: casual, diplomatic, direct, formal)\n\n"
        "Paste everything and send:",
        parse_mode=ParseMode.HTML,
    )
    return ASK_CHAT


async def got_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = update.message.text.strip()

    # Extract optional tone from first line
    tone = "diplomatic"
    lines = raw.split("\n")
    if lines and re.match(r"^tone\s*:", lines[0], re.IGNORECASE):
        tone_val = lines[0].split(":", 1)[1].strip().lower()
        if tone_val in ("casual", "diplomatic", "direct", "formal"):
            tone = tone_val
        raw = "\n".join(lines[1:]).strip()

    context.user_data["raw_chat"] = raw
    context.user_data["tone"] = tone

    await update.message.reply_text("⏳ Analyzing… this may take a few seconds.")

    try:
        data = await analyze_project(
            project_name=context.user_data["project_name"],
            deadline=context.user_data["deadline"],
            members=context.user_data["members"],
            raw_chat=raw,
            tone=tone,
        )
        _cache[update.effective_chat.id] = data
        await _send_long(update, fmt_full(data))
    except Exception as e:
        await update.message.reply_text(f"❌ Error during analysis: {e}")

    return ConversationHandler.END


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


# ---------------------------------------------------------------------------
# Quick result commands — use cached analysis
# ---------------------------------------------------------------------------

async def _require_cache(update: Update) -> dict | None:
    data = _cache.get(update.effective_chat.id)
    if not data:
        await update.message.reply_text(
            "No analysis found for this chat. Run /pulse first."
        )
    return data


async def cmd_score(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await _require_cache(update)
    if data:
        await update.message.reply_html(fmt_risk(data))


async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await _require_cache(update)
    if data:
        await update.message.reply_html(fmt_tasks(data))


async def cmd_blockers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await _require_cache(update)
    if data:
        await update.message.reply_html(fmt_blockers(data))


async def cmd_suggest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = await _require_cache(update)
    if data:
        await update.message.reply_html(fmt_suggest(data))


# ---------------------------------------------------------------------------
# Group keyword handler — "pulse score", "pulse tasks", etc.
# ---------------------------------------------------------------------------

KEYWORD_MAP = {
    "score":    cmd_score,
    "tasks":    cmd_tasks,
    "blockers": cmd_blockers,
    "suggest":  cmd_suggest,
    "overall":  None,  # special: full analysis
}


async def handle_pulse_keyword(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip().lower()
    # Match "pulse <keyword>"
    m = re.match(r"pulse\s+(\w+)", text)
    keyword = m.group(1) if m else "overall"

    if keyword in KEYWORD_MAP and KEYWORD_MAP[keyword] is not None:
        await KEYWORD_MAP[keyword](update, context)
    else:
        # "pulse" or "pulse overall" → show full cached result or prompt
        data = _cache.get(update.effective_chat.id)
        if data:
            await _send_long(update, fmt_full(data))
        else:
            await update.message.reply_text(
                "No analysis yet. Use /pulse to start a full analysis."
            )


# ---------------------------------------------------------------------------
# Utility: send long messages in chunks
# ---------------------------------------------------------------------------

async def _send_long(update: Update, text: str, chunk: int = 4000):
    for i in range(0, len(text), chunk):
        await update.message.reply_html(text[i:i + chunk])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not TELEGRAM_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("pulse", cmd_pulse)],
        states={
            ASK_NAME:     [MessageHandler(filters.TEXT & ~filters.COMMAND, got_name)],
            ASK_DEADLINE: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_deadline)],
            ASK_MEMBERS:  [MessageHandler(filters.TEXT & ~filters.COMMAND, got_members)],
            ASK_CHAT:     [MessageHandler(filters.TEXT & ~filters.COMMAND, got_chat)],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        per_user=True,
        per_chat=True,
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(conv)
    app.add_handler(CommandHandler("score",    cmd_score))
    app.add_handler(CommandHandler("tasks",    cmd_tasks))
    app.add_handler(CommandHandler("blockers", cmd_blockers))
    app.add_handler(CommandHandler("suggest",  cmd_suggest))

    # Group keyword handler (lower priority than conversation)
    app.add_handler(MessageHandler(
        filters.TEXT & filters.Regex(r"(?i)\bpulse\b"),
        handle_pulse_keyword,
    ))

    print("✅ Project Pulse bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
