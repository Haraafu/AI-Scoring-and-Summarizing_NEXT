import re

EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b')
PHONE = re.compile(r'\b(\+?\d{1,3}[-.\s]?)?\d{8,14}\b')
IDNUM = re.compile(r'\b\d{8,16}\b')
PASS_HINT = re.compile(r'(?i)(password|passwd|pwd|kata\s*sandi|creds?)')

def detect_signals(text: str):
    signals = []
    if EMAIL.search(text): signals.append("email")
    if PASS_HINT.search(text): signals.append("password_hint")
    if IDNUM.search(text): signals.append("id_candidate")
    if PHONE.search(text): signals.append("phone_candidate")
    return signals

def mask_text(text: str):
    text = EMAIL.sub("<EMAIL>", text)
    text = PHONE.sub("<PHONE>", text)
    text = IDNUM.sub("<ID>", text)
    return text