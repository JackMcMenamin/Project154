import re

def extract_number(filename):
    parts = re.findall(r'(\d+)', filename)
    return tuple(map(int, parts)) if parts else (0,)
