import os
import requests
import logging

try:
    from dotenv import load_dotenv
    root_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
    load_dotenv(dotenv_path=root_env_path)
except ImportError:
    pass

logger = logging.getLogger("Notifications")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

def send_telegram_alert(
    message: str,
    bot_token: str = None,
    chat_id: str = None,
    parse_mode: str = None,
    disable_notification: bool = False,
    dry_run: bool = False,
) -> dict:
    
    if dry_run:
        logger.info(f"Telegram DRY-RUN. Message not sent. length={len(message)}")
        return {"success": True, "error": "dry_run - message not sent", "status_code": 0}

    # Use provided tokens, fallback to environment
    token = bot_token or TELEGRAM_BOT_TOKEN
    target_chat = chat_id or TELEGRAM_CHAT_ID

    if not token or "YOUR_" in token or len(token) < 10:
        logger.warning("Telegram bot token not configured. Mock Alert.")
        return {"success": False, "error": "mock_alert - bad token", "status_code": 0}

    if not target_chat or "YOUR_" in str(target_chat).upper():
        logger.warning("Telegram chat ID not configured. Mock Alert.")
        return {"success": False, "error": "mock_alert - bad chat_id", "status_code": 0}

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": str(target_chat), 
        "text": message,
        "disable_web_page_preview": True,
        "disable_notification": disable_notification,
    }
    
    if parse_mode:
        payload["parse_mode"] = parse_mode
        
    try:
        resp = requests.post(url, json=payload, timeout=12)
        result = {"status_code": resp.status_code}
        
        if resp.status_code == 200 and resp.json().get("ok"):
            logger.info("Successfully sent alert to Telegram.")
            result["success"] = True
            result["message_id"] = resp.json().get("result", {}).get("message_id")
        else:
            error_details = resp.json().get("description") if resp.status_code == 200 else resp.text[:200]
            logger.error(f"Telegram API error: {error_details}")
            result["success"] = False
            result["error"] = error_details
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to send Telegram alert. Error: {e}")
        return {"success": False, "error": str(e), "status_code": 0}

