from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

logger = logging.getLogger("Notifications")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

_TIMEOUT_S       = 12
_DEDUP_COOLDOWN  = int(os.getenv("ALERT_DEDUP_COOLDOWN_S", "300"))
_sent_cache: Dict[str, float] = {}


def send_telegram_alert(
    message: str,
    entity_key: Optional[str] = None,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: Optional[str] = None,
    disable_notification: bool = False,
    dry_run: bool = False,
    skip_dedup: bool = False,
) -> dict:
    if not skip_dedup:
        fp  = _fingerprint(entity_key or message[:120])
        now = time.monotonic()
        if now - _sent_cache.get(fp, 0.0) < _DEDUP_COOLDOWN:
            remaining = int(_DEDUP_COOLDOWN - (now - _sent_cache[fp]))
            logger.info("Alert suppressed for '%s' (%ds cooldown).", entity_key, remaining)
            return {"success": True, "error": f"dedup_suppressed:{remaining}s", "status_code": 0}

    if dry_run:
        logger.info("Dry-run: message not sent (len=%d).", len(message))
        return {"success": True, "error": "dry_run", "status_code": 0}

    token  = bot_token or TELEGRAM_BOT_TOKEN
    target = chat_id   or TELEGRAM_CHAT_ID

    if not _valid_token(token):
        logger.warning("Telegram token missing or invalid.")
        return {"success": False, "error": "bad_token", "status_code": 0}

    if not target or "YOUR_" in str(target).upper():
        logger.warning("Telegram chat_id not configured.")
        return {"success": False, "error": "bad_chat_id", "status_code": 0}

    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": str(target),
        "text": message,
        "disable_web_page_preview": True,
        "disable_notification": disable_notification,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        resp   = requests.post(url, json=payload, timeout=_TIMEOUT_S)
        result = {"status_code": resp.status_code}

        if resp.status_code == 200 and resp.json().get("ok"):
            result["success"]    = True
            result["message_id"] = resp.json().get("result", {}).get("message_id")
            if not skip_dedup:
                _sent_cache[fp] = time.monotonic()
            logger.info("Telegram alert sent.")
        else:
            error = resp.json().get("description") if resp.status_code == 200 else resp.text[:200]
            logger.error("Telegram API error: %s", error)
            result["success"] = False
            result["error"]   = error

        return result

    except requests.exceptions.Timeout:
        logger.error("Telegram POST timed out after %ds.", _TIMEOUT_S)
        return {"success": False, "error": "timeout", "status_code": 0}
    except Exception as exc:
        logger.error("Telegram send failed: %s", exc)
        return {"success": False, "error": str(exc), "status_code": 0}


def _fingerprint(key: str) -> str:
    return hashlib.md5(key.encode("utf-8", errors="replace")).hexdigest()


def _valid_token(token: str) -> bool:
    return bool(token) and "YOUR_" not in token.upper() and len(token) >= 10
