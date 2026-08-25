import requests
import pandas as pd
from io import StringIO

# Test HoopsHype
url = "https://hoopshype.com/salaries/players/2024-2025/"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
r = requests.get(url, headers=headers)
print(f"Status: {r.status_code}")
print(f"Has <table: {'<table' in r.text}")
print(f"Content length: {len(r.text)}")

if "<table" in r.text:
    dfs = pd.read_html(StringIO(r.text))
    print(f"Tables found: {len(dfs)}")
    if dfs:
        print(dfs[0].head(3))
else:
    # Check for JSON data or script tags
    print("No tables found in HTML. Checking for salary data patterns...")
    idx = r.text.find("salary")
    if idx > 0:
        print(f"Found 'salary' at position {idx}")
        print(r.text[max(0,idx-50):idx+200])
