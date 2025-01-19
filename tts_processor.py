# Credit to Diffusion_TSS and https://github.com/netease-youdao/EmotiVoice

import re
import dateparser
from num2words import num2words

punctuation = r'[\s,.?!/)\'\]>]'
alphabet_map = {
    "A": " Eh ",
    "B": " Bee ",
    "C": " See ",
    "D": " Dee ",
    "E": " Eee ",
    "F": " Eff ",
    "G": " Jee ",
    "H": " Aitch ",
    "I": " Eye ",
    "J": " Jay ",
    "K": " Kay ",
    "L": " El ",
    "M": " Emm ",
    "N": " Enn ",
    "O": " Ohh ",
    "P": " Pee ",
    "Q": " Queue ",
    "R": " Are ",
    "S": " Ess ",
    "T": " Tee ",
    "U": " You ",
    "V": " Vee ",
    "W": " Double You ",
    "X": " Ex ",
    "Y": " Why ",
    "Z": " Zed "
}


def preprocess_all(string):
    # the order for some of these matter
    # For example, you need to remove the commas in numbers before expanding them
    string = normalize_dates(string)
    string = replace_invalid_chars(string)
    string = replace_numbers(string)

    # TODO Try to use a ML predictor to expand abbreviations. It's hard, dependent on context, and whether to actually
    # try to say the abbreviation or spell it out as I've done below is not agreed upon

    # For now, expand abbreviations to pronunciations
    # replace_abbreviations adds a lot of unnecessary whitespace to ensure separation
    string = replace_abbreviations(string)

    # cleanup whitespaces
    string = clean_whitespace(string)

    return string

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

def format_date(parsed_date):
    """Formats a parsed date into a human-readable string."""
    if not parsed_date:
        return None

    # Convert the day into an ordinal (e.g., 13 -> 13th)
    day = add_ordinal_suffix(parsed_date.day)

    # Format the date in a TTS-friendly way
    return parsed_date.strftime(f"%B {day}, %Y at %-I:%M %p")  # Unix
    # On Windows, use "%B {day}, %Y at %#I:%M %p"

def normalize_dates(text):
    """
    Finds and replaces date strings with a nicely formatted, TTS-friendly version.
    """
    def replace_date(match):
        raw_date = match.group(0)
        # Attempt to parse the date
        parsed_date = dateparser.parse(raw_date)

        if parsed_date:
            # Format the parsed date
            return format_date(parsed_date)

        # Return the original text if parsing fails
        return raw_date

    # Match common date formats
    date_pattern = r"\b(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?|\d{2}/\d{2}/\d{4}|\d{1,2} \w+ \d{4})\b"
    return re.sub(date_pattern, replace_date, text)

def replace_invalid_chars(string):
    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('`', '')
    string = string.replace("'","")
    string = string.replace('\u201D', '').replace('\u201C', '')  # right and left quote
    string = string.replace('\u201F', '')  # italic looking quote
    string = string.replace('\n', ' ')
    string = string.replace('&#x27;', '')
    string = string.replace('AI;', 'Artificial Intelligence!')
    string = string.replace('iddqd;', 'Immortality cheat code')
    string = string.replace('😉;', 'wink wink!')
    string = string.replace(';);', 'wink wink!')
    string = string.replace(';-);', 'wink wink!')
    string = string.replace(':D', '*laughs* Ahahaha!')
    string = string.replace(';D', '*laughs* Ahahaha!')
    string = string.replace(':-D', '*laughs* Ahahaha!')
    return string


def replace_numbers(string):
    string = convert_num_locale(string)
    string = replace_negative(string)
    string = replace_roman(string)
    string = hyphen_range_to(string)
    string = num_to_words(string)
    return string


def remove_surrounded_chars(string):
    # first this expression will check if there is a string nested exclusively between a alt=
    # and a style= string. This would correspond to only a the alt text of an embedded image
    # If it matches it will only keep that part as the string, and rend it for further processing
    # Afterwards this expression matches to 'as few symbols as possible (0 upwards) between any
    # asterisks' OR' as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    if re.search(r'(?<=alt=)(.*)(?=style=)', string, re.DOTALL):
        m = re.search(r'(?<=alt=)(.*)(?=style=)', string, re.DOTALL)
        string = m.group(0)
    return re.sub(r'\*[^*]*?(\*|$)', '', string)


def convert_num_locale(text):
    # This detects locale and converts it to American without comma separators
    pattern = re.compile(r'(?:\s|^)\d{1,3}(?:\.\d{3})+(,\d+)(?:\s|$)')
    result = text
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start] + result[start:end].replace('.', '').replace(',', '.') + result[end:len(result)]

    # removes comma separators from existing American numbers
    pattern = re.compile(r'(\d),(\d)')
    result = pattern.sub(r'\1\2', result)

    return result


def replace_negative(string):
    # handles situations like -5. -5 would become negative 5, which would then be expanded to negative five
    return re.sub(rf'(\s)(-)(\d+)({punctuation})', r'\1negative \3\4', string)


def replace_roman(string):
    # find a string of roman numerals.
    # Only 2 or more, to avoid capturing I and single character abbreviations, like names
    pattern = re.compile(rf'\s[IVXLCDM]{{2,}}{punctuation}')
    result = string
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start + 1] + str(roman_to_int(result[start + 1:end - 1])) + result[end - 1:len(result)]

    return result


def roman_to_int(s):
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val


def hyphen_range_to(text):
    pattern = re.compile(r'(\d+)[-–](\d+)')
    result = pattern.sub(lambda x: x.group(1) + ' to ' + x.group(2), text)
    return result


def num_to_words(text):
    # 1000 or 10.23
    pattern = re.compile(r'\d+\.\d+|\d+')
    result = pattern.sub(lambda x: num2words(float(x.group())), text)
    return result


def replace_abbreviations(string):
    string = replace_uppercase_abbreviations(string)
    string = replace_lowercase_abbreviations(string)
    return string


def replace_uppercase_abbreviations(string):
    # abbreviations 1 to 4 characters long. It will get things like A and I, but those are pronounced with their letter
    pattern = re.compile(rf'(^|[\s(.\'\[<])([A-Z]{{1,4}})({punctuation}|$)')
    result = string
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start] + replace_abbreviation(result[start:end]) + result[end:len(result)]

    return result


def replace_lowercase_abbreviations(string):
    # abbreviations 1 to 4 characters long, separated by dots i.e. e.g.
    pattern = re.compile(rf'(^|[\s(.\'\[<])(([a-z]\.){{1,4}})({punctuation}|$)')
    result = string
    while True:
        match = pattern.search(result)
        if match is None:
            break

        start = match.start()
        end = match.end()
        result = result[0:start] + replace_abbreviation(result[start:end].upper()) + result[end:len(result)]

    return result


def replace_abbreviation(string):
    result = ""
    for char in string:
        result += match_mapping(char)

    return result


def match_mapping(char):
    for mapping in alphabet_map.keys():
        if char == mapping:
            return alphabet_map[char]

    return char


def clean_whitespace(string):
    # remove whitespace before punctuation
    string = re.sub(rf'\s+({punctuation})', r'\1', string)
    string = string.strip()
    # compact whitespace
    string = ' '.join(string.split())
    return string


def __main__(args):
    print(preprocess_all(args[1]))

if __name__ == "__main__":
    import sys
    __main__(sys.argv)

