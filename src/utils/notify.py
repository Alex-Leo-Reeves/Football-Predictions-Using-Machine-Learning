"""Webhook notifications (Discord / Telegram).

Telegram messages are structured per selection with clubs, kickoff time,
country, league and the chosen market — so the user gets a clean, readable
slip. Only matches that have not started are included (the pipeline filters
them before building the ticket).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict

import requests


def send_discord(webhook_url: str, content: str) -> bool:
    try:
        resp = requests.post(webhook_url, json={"content": content}, timeout=10)
        return resp.status_code == 204
    except requests.RequestException:
        return False


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": text,
                                        "parse_mode": "HTML"}, timeout=10)
        ok = resp.status_code == 200
        if not ok:
            print(f"[notify] Telegram send failed: HTTP {resp.status_code} — {resp.text[:200]}")
        return ok
    except requests.RequestException as exc:
        print(f"[notify] Telegram send error: {exc}")
        return False


def _format_time(iso: str) -> str:
    """Format an ISO timestamp to a readable local time (HH:MM)."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return iso or ""


def _selection_block(sel: Dict[str, Any], idx: int) -> str:
    """One structured selection block: clubs, time, country, league, market."""
    match = sel.get("match") or f"{sel.get('home_id', '')} vs {sel.get('away_id', '')}"
    league = sel.get("league") or "—"
    country = sel.get("country") or "—"
    kickoff = _format_time(sel.get("kickoff", ""))
    market = sel.get("market", "")
    odds = sel.get("odds", "")
    confidence = sel.get("confidence", 0.0)
    sport = (sel.get("sport") or "football").capitalize()
    return (
        f"<b>{idx}. {match}</b>\n"
        f"   🕐 {kickoff}  |  🌍 {country}  |  🏆 {league}  |  ⚽ {sport}\n"
        f"   🎯 {market} @ <b>{odds}</b>  (conf {confidence:.1%})"
    )


def _build_message(ticket: Dict[str, Any]) -> str:
    """Build the structured message (Telegram HTML-safe)."""
    date = ticket.get("date", "")
    legs = ticket.get("total_legs", 0)
    combined = ticket.get("combined_odds", 0)
    est_prob = ticket.get("estimated_probability", 0.0)
    target = ticket.get("target_odds", 300.0)

    header = (
        f"🎯 <b>Daily Ticket — {date}</b>\n"
        f"Legs: {legs} | Combined odds: <b>{combined}</b> | "
        f"Est. probability: {est_prob:.4%}\n"
        f"Target: {target} | Confidence floor: "
        f"{os.getenv('MIN_CONFIDENCE', '0.985')}\n"
        "──────────────────────────────"
    )

    if not legs:
        return header + "\n\n⚠️ No qualifying selections today — the engine refused to force low-confidence picks. Better no ticket than a cut ticket."

    blocks = [_selection_block(sel, i + 1) for i, sel in enumerate(ticket.get("selections", []))]
    footer = (
        "──────────────────────────────\n"
        "✅ Only matches that have NOT started are included.\n"
        "🔒 ALL-OR-NOTHING accumulator — every leg must land."
    )
    return "\n\n".join([header, *blocks, footer])


def notify_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    """Send ticket summary to any configured channels.

    Returns a result dict so the workflow can log exactly what happened
    (which channels were configured, whether each send succeeded).
    """
    content = _build_message(ticket)
    results: Dict[str, Any] = {"telegram": None, "discord": None}

    discord_url = os.getenv("DISCORD_WEBHOOK_URL")
    if discord_url:
        results["discord"] = send_discord(discord_url, content)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        results["telegram"] = send_telegram(token, chat_id, content)
    else:
        print("[notify] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skipping Telegram")
    print(f"[notify] notify_ticket result: {results}")
    return results

