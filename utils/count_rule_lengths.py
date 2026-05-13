from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "recommendation_rules.pkl"


def main() -> None:
    if not MODEL_FILE.exists():
        print(f"Khong tim thay file model: {MODEL_FILE}")
        return

    rules = joblib.load(MODEL_FILE)
    if not isinstance(rules, pd.DataFrame) or rules.empty:
        print("Model khong hop le hoac khong co luat.")
        return

    if "antecedents" not in rules.columns:
        print("Khong tim thay cot 'antecedents' trong model.")
        return

    ant_len = rules["antecedents"].apply(len)

    count_len_1 = int((ant_len == 1).sum())  # A => B
    count_len_2 = int((ant_len == 2).sum())  # A,B => C
    count_len_3 = int((ant_len == 3).sum())  # A,B,C => D

    print("=== THONG KE DO DAI VE TRAI CUA LUAT ===")
    print(f"Tong so luat: {len(rules)}")
    print(f"Len = 1 (A => B): {count_len_1}")
    print(f"Len = 2 (A,B => C): {count_len_2}")
    print(f"Len = 3 (A,B,C => D): {count_len_3}")
    
    if not {"support", "consequents"}.issubset(rules.columns):
        print("\nKhong tim thay day du cot support/consequents de in top luat.")
        return

    top10 = rules.sort_values(by="support", ascending=False).head(10)
    print("\n=== TOP 10 LUAT PHO BIEN NHAT (THEO SUPPORT) ===")
    for _, row in top10.iterrows():
        left = set(row["antecedents"])
        right = set(row["consequents"])
        print(
            f"{left} => {right} | "
            f"Support: {row['support']:.4f}, "
            f"Confidence: {row.get('confidence', 0):.2f}, "
            f"Lift: {row.get('lift', 0):.2f}"
        )


if __name__ == "__main__":
    main()
