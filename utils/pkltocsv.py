import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import joblib

rules = joblib.load("recommendation_rules.pkl")
rules.to_csv("data/recommendation_rules_readable.csv", index=False, encoding="utf-8-sig")
print("Da xuat file data/recommendation_rules_readable.csv")