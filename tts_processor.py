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

# Clean up whitespace in the text
def clean_whitespace(string):
    string = re.sub(r'\s+([.,?!])', r'\1', string)
    return ' '.join(string.split())

# Main preprocessing pipeline
def preprocess_all(string):
    string = normalize_dates(string)
    string = replace_invalid_chars(string)
    string = replace_numbers(string)
    string = replace_abbreviations(string)
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

