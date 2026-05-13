import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
import joblib

MIN_CONF = 0.2  # Ngưỡng confidence tối thiểu cho stable rate

print("Đang tải dữ liệu train, test và rules...")
train_df = pd.read_csv('data/train_transactions.csv')
test_df = pd.read_csv('data/test_transactions.csv')
rules_strong_r = joblib.load('recommendation_rules.pkl')

train_r = train_df.copy()
train_r['Items'] = train_r['Items_str'].apply(lambda x: str(x).split('|') if pd.notna(x) else [])

test_r = test_df.copy()
test_r['Items'] = test_r['Items_str'].apply(lambda x: str(x).split('|') if pd.notna(x) else [])

rules_strong_r['antecedents'] = rules_strong_r['antecedents'].apply(frozenset)
rules_strong_r['consequents'] = rules_strong_r['consequents'].apply(frozenset)

test_sets = test_r['Items'].apply(set).tolist()
N_test = len(test_sets)


def support_test(X):
    X = set(X)
    return sum(1 for s in test_sets if X.issubset(s)) / N_test


rows = []
for _, r in rules_strong_r.iterrows():
    A = set(r['antecedents'])
    B = set(r['consequents'])
    sA = support_test(A)
    sAB = support_test(A | B)
    conf_t = (sAB / sA) if sA > 0 else 0.0

    rows.append({
        "A": tuple(sorted(A)),
        "B": tuple(sorted(B)),
        "conf_train": r['confidence'],
        "conf_test": conf_t,
        "lift_train": r['lift'],
        "conf_drop": r['confidence'] - conf_t,
        "support_train": r['support'],
        "support_test(A)": sA,
    })

eval_tbl = pd.DataFrame(rows)

stable_rate = (eval_tbl['conf_test'] >= MIN_CONF).mean() if len(eval_tbl) else 0.0
avg_drop = eval_tbl['conf_drop'].mean() if len(eval_tbl) else 0.0

print(f"StableRate(conf_test >= {MIN_CONF}): {stable_rate:.3f}")
print(f"AvgConfDrop: {avg_drop:.3f}")

A_list = [set(a) for a in rules_strong_r['antecedents']]


def rule_coverage_on(df):
    sets = df['Items'].apply(set).tolist()
    covered = 0
    for s in sets:
        if any(A.issubset(s) for A in A_list):
            covered += 1
    return covered / len(sets)


print("RuleCoverage train_random:", round(rule_coverage_on(train_r), 3))
print("RuleCoverage test_random :", round(rule_coverage_on(test_r), 3))

print("\nTop 20 luật theo conf_drop:")
print(eval_tbl.sort_values('conf_drop', ascending=False).head(20).to_string(index=False))
