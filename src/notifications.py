import os
import requests
import logging

try:
    from dotenv import load_dotenv
    root_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(dotenv_path=root_env_path)
except ImportError:
    pass

logger = logging.getLogger("Notifications")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

def send_telegram_alert(message):
    """Send a notification to a specific Telegram Chat."""
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or TELEGRAM_CHAT_ID == "YOUR_TELEGRAM_CHAT_ID":
        logger.warning(f"Telegram credentials not configured. Mock Alert: {message}")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=5)
        logger.info("Successfully sent alert to Telegram.")
    except Exception as e:
        logger.error(f"Failed to send Telegram alert. Error: {e}")

def analyze_with_gpt(error_text):
    """Pass error details to GPT for root cause adjudication."""
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        logger.warning("OpenAI strictly not configured. Using Mock GPT Analysis.")
        return f"Mock GPT Analysis: The error '{error_text}' could be due to missing container metrics, tensor shape mismatches, or internal model state failures."
        
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4", # or gpt-3.5-turbo
        "messages": [
            {"role": "system", "content": "You are an intelligent Kubernetes SRE adjudication module."},
            {"role": "user", "content": f"The BiLSTM Anomaly inference loop just generated an error. Provide a very brief root cause analysis for this Python/PyTorch exception: {error_text}"}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"GPT Call Failed (HTTP {response.status_code})"
    except Exception as e:
        return f"OpenAI request crashed: {e}"
