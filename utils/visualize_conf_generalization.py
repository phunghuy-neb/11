from pathlib import Path
from typing import List, Set

import joblib
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "recommendation_rules.pkl"
TEST_FILE = BASE_DIR / "data" / "test_transactions.csv"


def to_tx_sets(items_str_series: pd.Series) -> List[Set[str]]:
    tx = []
    for x in items_str_series.dropna().astype(str):
        tx.append(set(i.strip().upper() for i in x.split("|") if i.strip()))
    return tx


def rule_conf_on_test(antecedent: Set[str], consequent: Set[str], tx_sets: List[Set[str]]) -> float:
    ant_count = 0
    both_count = 0
    target = antecedent | consequent
    for t in tx_sets:
        if antecedent.issubset(t):
            ant_count += 1
            if target.issubset(t):
                both_count += 1
    if ant_count == 0:
        return float("nan")
    return both_count / ant_count


def main() -> None:
    if not MODEL_FILE.exists():
        print(f"Khong tim thay model: {MODEL_FILE}")
        return
    if not TEST_FILE.exists():
        print(f"Khong tim thay file test: {TEST_FILE}")
        return

    rules = joblib.load(MODEL_FILE)
    if not isinstance(rules, pd.DataFrame) or rules.empty:
        print("Model khong hop le hoac khong co luat.")
        return
    if "confidence" not in rules.columns:
        print("Khong co cot confidence trong model.")
        return

    test_df = pd.read_csv(TEST_FILE)
    if "Items_str" not in test_df.columns:
        print("Khong co cot Items_str trong data/test_transactions.csv")
        return

    tx_sets = to_tx_sets(test_df["Items_str"])
    if not tx_sets:
        print("Tap test rong.")
        return

    eval_df = rules.copy()
    eval_df["antecedent_set"] = eval_df["antecedents"].apply(lambda x: set(str(i).strip().upper() for i in x))
    eval_df["consequent_set"] = eval_df["consequents"].apply(lambda x: set(str(i).strip().upper() for i in x))
    eval_df["conf_train"] = eval_df["confidence"]
    eval_df["conf_test"] = eval_df.apply(
        lambda r: rule_conf_on_test(r["antecedent_set"], r["consequent_set"], tx_sets), axis=1
    )
    eval_df = eval_df.dropna(subset=["conf_train", "conf_test"]).copy()

    if eval_df.empty:
        print("Khong co luat nao tinh duoc conf_test tren tap test.")
        return

    # Scatter: conf_train vs conf_test + y=x
    plt.figure(figsize=(8, 7))
    plt.scatter(eval_df["conf_train"], eval_df["conf_test"], alpha=0.7, color="#1f77b4")
    plt.plot([0, 1], [0, 1], "--", color="#d62728", linewidth=1.5, label="y=x")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xlabel("conf_train")
    plt.ylabel("conf_test")
    plt.title("Strong rules generalization: conf_train vs conf_test")
    plt.legend()
    plt.tight_layout()
    out_plot = BASE_DIR / "conf_train_vs_conf_test.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()

    # Summary
    mae = (eval_df["conf_train"] - eval_df["conf_test"]).abs().mean()
    corr = eval_df["conf_train"].corr(eval_df["conf_test"])
    summary = [
        "=== RULE GENERALIZATION SUMMARY ===",
        f"So luat duoc danh gia: {len(eval_df):,}",
        f"MAE |conf_train - conf_test|: {mae:.4f}",
        f"Tuong quan Pearson(conf_train, conf_test): {corr:.4f}",
        f"Da luu bieu do: {out_plot.name}",
    ]
    out_txt = BASE_DIR / "conf_generalization_summary.txt"
    out_txt.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
