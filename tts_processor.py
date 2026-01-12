import re
from dateutil.parser import parse
from num2words import num2words
import inflect
from ftfy import fix_text

# Initialize the inflect engine
inflect_engine = inflect.engine()

# Define alphabet pronunciation mapping
alphabet_map = {
    "A": " Eh ", "B": " Bee ", "C": " See ", "D": " Dee ", "E": " Eee ",
    "F": " Eff ", "G": " Jee ", "H": " Aitch ", "I": " Eye ", "J": " Jay ",
    "K": " Kay ", "L": " El ", "M": " Emm ", "N": " Enn ", "O": " Ohh ",
    "P": " Pee ", "Q": " Queue ", "R": " Are ", "S": " Ess ", "T": " Tee ",
    "U": " You ", "V": " Vee ", "W": " Double You ", "X": " Ex ", "Y": " Why ", "Z": " Zed "
}

# Function to add ordinal suffix to a number
def add_ordinal_suffix(day):
    """Adds ordinal suffix to a day (e.g., 13 -> 13th)."""
    if 11 <= day <= 13:  # Special case for 11th, 12th, 13th
        return f"{day}th"
    elif day % 10 == 1:
        return f"{day}st"
    elif day % 10 == 2:
        return f"{day}nd"
    elif day % 10 == 3:
        return f"{day}rd"
    else:
        return f"{day}th"

# Function to format dates in a human-readable form
def format_date(parsed_date, include_time=True):
    """Formats a parsed date into a human-readable string."""
    if not parsed_date:
        return None

    # Convert the day into an ordinal (e.g., 13 -> 13th)
    day = add_ordinal_suffix(parsed_date.day)

    # Format the date in a TTS-friendly way
    if include_time and parsed_date.hour != 0 and parsed_date.minute != 0:
        return parsed_date.strftime(f"%B {day}, %Y at %-I:%M %p")  # Unix
    return parsed_date.strftime(f"%B {day}, %Y")  # Only date

# Normalize dates in the text
def normalize_dates(text):
    """
    Finds and replaces date strings with a nicely formatted, TTS-friendly version.
    """
    def replace_date(match):
        raw_date = match.group(0)
        try:
            parsed_date = parse(raw_date)
            if parsed_date:
                include_time = "T" in raw_date or " " in raw_date  # Include time only if explicitly provided
                return format_date(parsed_date, include_time)
        except ValueError:
            pass
        return raw_date

    # Match common date formats
    date_pattern = r"\b(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?|\d{2}/\d{2}/\d{4}|\d{1,2} \w+ \d{4})\b"
    return re.sub(date_pattern, replace_date, text)

# Replace invalid characters and clean text
def replace_invalid_chars(string):
    string = fix_text(string)
    replacements = {
        "**": "",
        '&#x27;': "'",
        'AI;': 'Artificial Intelligence!',
        'iddqd;': 'Immortality cheat code',
        '😉;': 'wink wink!',
        ':D': '*laughs* Ahahaha!',
        ';D': '*laughs* Ahahaha!'
    }
    for old, new in replacements.items():
        string = string.replace(old, new)
    return string

# Replace numbers with their word equivalents
def replace_numbers(string):
    ipv4_pattern = r'(\b\d{1,3}(\.\d{1,3}){3}\b)'
    ipv6_pattern = r'([0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}'
    range_pattern = r'\b\d+-\d+\b'  # Detect ranges like 1-4
    date_pattern = r'\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?\b'
    alphanumeric_pattern = r'\b[A-Za-z]+\d+|\d+[A-Za-z]+\b'

    # Do not process IP addresses, date patterns, or alphanumerics
    if re.search(ipv4_pattern, string) or re.search(ipv6_pattern, string) or re.search(range_pattern, string) or re.search(date_pattern, string) or re.search(alphanumeric_pattern, string):
        return string

    # Convert standalone numbers and port numbers
    def convert_number(match):
        number = match.group()
        return num2words(int(number)) if number.isdigit() else number

    pattern = re.compile(r'\b\d+\b')
    return re.sub(pattern, convert_number, string)

# Replace abbreviations with expanded form
def replace_abbreviations(string):
    words = string.split()
    for i, word in enumerate(words):
        if word.isupper() and len(word) <= 4 and not any(char.isdigit() for char in word) and word not in ["ID", "AM", "PM"]:
            words[i] = ''.join([alphabet_map.get(char, char) for char in word])
    return ' '.join(words)

def clean_whitespace(string):
    # Remove spaces before punctuation
    string = re.sub(r'\s+([.,?!])', r'\1', string)
    # Collapse multiple spaces into one, but don’t touch inside tokens like "test.com"
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip()

def make_dots_tts_friendly(text):
    # Handle IP addresses (force "dot")
    ipv4_pattern = r'\b\d{1,3}(\.\d{1,3}){3}\b'
    text = re.sub(ipv4_pattern, lambda m: m.group(0).replace('.', ' dot '), text)

    # Handle domain-like endings (force "dot")
    domain_pattern = r'\b([\w-]+)\.(com|net|org|io|gov|edu|exe|dll|local)\b'
    text = re.sub(domain_pattern, lambda m: m.group(0).replace('.', ' dot '), text)

    # Handle decimals (use "point")
    decimal_pattern = r'\b\d+\.\d+\b'
    text = re.sub(decimal_pattern, lambda m: m.group(0).replace('.', ' point '), text)

    # Handle leading dot words (.Net → dot Net)
    text = re.sub(r'\.(?=\w)', 'dot ', text)

    return text

def tech_humanize(text):
    """
    Humanize technical tokens (URLs, emails, UUIDs, MACs, paths) for TTS.
    Keep outputs ASCII and TTS-friendly.
    """
    def spell_chars(token):
        return " ".join(list(token))

    def normalize_url(match):
        url = match.group(0)
        url = url.replace("https://", "HTTPS://").replace("http://", "HTTP://")
        url = url.replace("://", " colon slash slash ")
        url = url.replace("/", " slash ")
        url = url.replace("?", " question mark ")
        url = url.replace("&", " and ")
        url = url.replace("=", " equals ")
        url = url.replace("#", " hash ")
        url = url.replace("_", " underscore ")
        url = url.replace("-", " dash ")
        url = url.replace(".", " dot ")
        return url

    def normalize_email(match):
        email = match.group(0)
        email = email.replace("@", " at ")
        email = email.replace(".", " dot ")
        email = email.replace("_", " underscore ")
        email = email.replace("-", " dash ")
        return email

    def normalize_uuid(match):
        uuid_text = match.group(0)
        groups = uuid_text.split("-")
        spelled = [" ".join(list(group)) for group in groups]
        return " dash ".join(spelled)

    def normalize_mac(match):
        mac_text = match.group(0)
        groups = mac_text.split(":")
        spelled = [" ".join(list(group)) for group in groups]
        return " colon ".join(spelled)

    def normalize_ipv6(match):
        ipv6_text = match.group(0)
        groups = ipv6_text.split(":")
        spelled = [" ".join(list(group)) for group in groups if group]
        return " colon ".join(spelled)

    def normalize_ipv6_compact(match):
        ipv6_text = match.group(0)
        left, _, right = ipv6_text.partition("::")
        left_groups = [g for g in left.split(":") if g]
        right_groups = [g for g in right.split(":") if g]
        left_spelled = [" ".join(list(group)) for group in left_groups]
        right_spelled = [" ".join(list(group)) for group in right_groups]
        middle = " double colon "
        left_part = " colon ".join(left_spelled)
        right_part = " colon ".join(right_spelled)
        if left_part and right_part:
            return f"{left_part}{middle}{right_part}"
        if left_part:
            return f"{left_part}{middle}"
        return f"{middle}{right_part}"

    def normalize_mac_dash(match):
        mac_text = match.group(0)
        groups = mac_text.split("-")
        spelled = [" ".join(list(group)) for group in groups]
        return " dash ".join(spelled)

    def normalize_hex(match):
        hex_text = match.group(1)
        return "hex " + " ".join(list(hex_text))

    def normalize_cve(match):
        year = match.group(1)
        ident = match.group(2)
        return f"C V E {year} dash {ident}"

    # Common protocol tokens (force letter-by-letter)
    text = re.sub(r"\bhttps\b", "H T T P S", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhttp\b", "H T T P", text, flags=re.IGNORECASE)
    text = re.sub(r"\bssh\b", "S S H", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdns\b", "D N S", text, flags=re.IGNORECASE)
    text = re.sub(r"\bntp\b", "N T P", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsnmp\b", "S N M P", text, flags=re.IGNORECASE)
    text = re.sub(r"\btcp\b", "T C P", text, flags=re.IGNORECASE)
    text = re.sub(r"\budp\b", "U D P", text, flags=re.IGNORECASE)
    text = re.sub(r"\bicmp\b", "I C M P", text, flags=re.IGNORECASE)
    text = re.sub(r"\bip\b", "I P", text, flags=re.IGNORECASE)
    text = re.sub(r"\bipv4\b", "I P v four", text, flags=re.IGNORECASE)
    text = re.sub(r"\bipv6\b", "I P v six", text, flags=re.IGNORECASE)
    text = re.sub(r"\btls\b", "T L S", text, flags=re.IGNORECASE)
    text = re.sub(r"\bssl\b", "S S L", text, flags=re.IGNORECASE)
    text = re.sub(r"\brdp\b", "R D P", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsql\b", "S Q L", text, flags=re.IGNORECASE)
    text = re.sub(r"\bapi\b", "A P I", text, flags=re.IGNORECASE)
    text = re.sub(r"\buid\b", "U I D", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgpu\b", "G P U", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcpu\b", "C P U", text, flags=re.IGNORECASE)
    text = re.sub(r"\bram\b", "R A M", text, flags=re.IGNORECASE)
    text = re.sub(r"\bttl\b", "T T L", text, flags=re.IGNORECASE)
    text = re.sub(r"\brtt\b", "R T T", text, flags=re.IGNORECASE)

    # Version tokens like TLS1.3 or HTTP/2
    text = re.sub(r"\b(tls|ssl)\s*(\d+(?:\.\d+)?)\b", lambda m: f"{m.group(1).upper()} {m.group(2).replace('.', ' point ')}", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhttps?/(\d+(?:\.\d+)?)\b", lambda m: f"H T T P slash {m.group(1).replace('.', ' point ')}", text, flags=re.IGNORECASE)

    # Hex values and CVEs
    text = re.sub(r"\b0x([0-9A-Fa-f]+)\b", normalize_hex, text)
    text = re.sub(r"\bCVE-(\d{4})-(\d{4,7})\b", normalize_cve, text)

    # Interfaces like eth0, wlan0, en0, lo0
    text = re.sub(r"\b(eth|wlan|en|lo)(\d+)\b", lambda m: f"{m.group(1)} {m.group(2)}", text, flags=re.IGNORECASE)

    # URLs and emails
    text = re.sub(r"\bhttps?://[^\s]+", normalize_url, text, flags=re.IGNORECASE)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", normalize_email, text)

    # UUIDs, MACs, IPv6
    text = re.sub(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", normalize_uuid, text)
    text = re.sub(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b", normalize_mac, text)
    text = re.sub(r"\b(?:[0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}\b", normalize_mac_dash, text)
    text = re.sub(r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b", normalize_ipv6, text)
    text = re.sub(r"\b[0-9A-Fa-f:]*::[0-9A-Fa-f:]*\b", normalize_ipv6_compact, text)

    # Common separators in paths/flags
    text = re.sub(r"(?<=\w)/(?!\s)", " slash ", text)
    text = re.sub(r"\\", " backslash ", text)
    text = re.sub(r"(?<=\w)-(?=\w)", " dash ", text)
    text = re.sub(r"(?<=\w)_(?=\w)", " underscore ", text)
    text = re.sub(r"(?<=\w):(?=\w)", " colon ", text)
    text = re.sub(r"--", " double dash ", text)
    text = re.sub(r"->", " arrow ", text)
    text = re.sub(r"=>", " arrow ", text)
    text = re.sub(r"\b(\d+)%\b", r"\1 percent", text)

    # Versions like v1.2.3 -> v 1 point 2 point 3
    text = re.sub(r"\bv(\d+(?:\.\d+)+)\b", lambda m: "v " + m.group(1).replace(".", " point "), text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+\.\d+\.\d+)\b", lambda m: m.group(1).replace(".", " point "), text)

    # Units and rates
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*kbps\b", r"\1 kilobits per second", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*mbps\b", r"\1 megabits per second", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*gbps\b", r"\1 gigabits per second", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*tbps\b", r"\1 terabits per second", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*kb\b", r"\1 kilobytes", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*mb\b", r"\1 megabytes", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*gb\b", r"\1 gigabytes", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*tb\b", r"\1 terabytes", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*mhz\b", r"\1 mega hertz", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*ghz\b", r"\1 giga hertz", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*ms\b", r"\1 milliseconds", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*us\b", r"\1 microseconds", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*ns\b", r"\1 nanoseconds", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*s\b", r"\1 seconds", text)
    text = re.sub(r"\b(\d+(?:\.\d+)?)\s*min\b", r"\1 minutes", text, flags=re.IGNORECASE)

    return text

# Main preprocessing pipeline
def preprocess_all(string):
    string = normalize_dates(string)
    string = replace_invalid_chars(string)
    string = replace_numbers(string)
    string = tech_humanize(string)
    string = replace_abbreviations(string)
    string = make_dots_tts_friendly(string)
    string = clean_whitespace(string)
    return string

# Expose a testing function for external use
def test_preprocessing(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        original = line.strip()
        processed = preprocess_all(original)
        print(f"Original: {original}")
        print(f"Processed: {processed}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        test_preprocessing(test_file)
    else:
        print("Please provide a file path as an argument.")
