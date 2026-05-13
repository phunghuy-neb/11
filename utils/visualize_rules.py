from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "recommendation_rules.pkl"


def main() -> None:
    if not MODEL_FILE.exists():
        print(f"Khong tim thay model: {MODEL_FILE}")
        return

    rules = joblib.load(MODEL_FILE)
    if not isinstance(rules, pd.DataFrame) or rules.empty:
        print("Model khong hop le hoac khong co luat de ve.")
        return

    need_cols = {"antecedents", "consequents", "support", "confidence", "lift"}
    if not need_cols.issubset(rules.columns):
        print("Model thieu cot can thiet de truc quan hoa.")
        return

    # 1) Top 10 luat theo lift
    top10 = rules.sort_values(by="lift", ascending=False).head(10).copy()
    top10["rule"] = top10.apply(
        lambda r: f"{set(r['antecedents'])} => {set(r['consequents'])}", axis=1
    )

    plt.figure(figsize=(13, 7))
    plt.barh(top10["rule"][::-1], top10["lift"][::-1], color="steelblue")
    plt.title("Top 10 Luat Ket Hop Theo Lift", fontsize=14)
    plt.xlabel("Lift")
    plt.ylabel("Rule")
    plt.tight_layout()
    out1 = BASE_DIR / "top10_rules_by_lift.png"
    plt.savefig(out1, dpi=150)
    plt.close()

    # 2) Phan phoi cac do do Support/Confidence/Lift
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(rules["support"], bins=25, color="#1f77b4", alpha=0.85)
    axes[0].set_title("Phan phoi Support")
    axes[0].set_xlabel("Support")
    axes[0].set_ylabel("So luat")

    axes[1].hist(rules["confidence"], bins=25, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("Phan phoi Confidence")
    axes[1].set_xlabel("Confidence")

    axes[2].hist(rules["lift"], bins=25, color="#2ca02c", alpha=0.85)
    axes[2].set_title("Phan phoi Lift")
    axes[2].set_xlabel("Lift")

    plt.tight_layout()
    out2 = BASE_DIR / "rules_metrics_distribution.png"
    plt.savefig(out2, dpi=150)
    plt.close()

    # 3) Phan bo do dai ve trai cua luat (len antecedents)
    ant_len = rules["antecedents"].apply(len)
    len_counts = ant_len.value_counts().sort_index()
    plt.figure(figsize=(7, 4.5))
    plt.bar(len_counts.index.astype(str), len_counts.values, color="#9467bd")
    plt.title("Phan bo do dai Antecedent (len)", fontsize=13)
    plt.xlabel("Len antecedents")
    plt.ylabel("So luat")
    for x, y in zip(len_counts.index.astype(str), len_counts.values):
        plt.text(x, y, str(int(y)), ha="center", va="bottom")
    plt.tight_layout()
    out3 = BASE_DIR / "rule_length_distribution.png"
    plt.savefig(out3, dpi=150)
    plt.close()

    # 4) Danh gia tong quan model rules
    summary_lines = [
        "=== DANH GIA TONG QUAN TAP LUAT ===",
        f"Tong so luat: {len(rules):,}",
        f"Support trung binh: {rules['support'].mean():.4f}",
        f"Confidence trung binh: {rules['confidence'].mean():.4f}",
        f"Lift trung binh: {rules['lift'].mean():.4f}",
        f"So luat Lift > 2: {int((rules['lift'] > 2).sum()):,}",
        f"So luat Lift > 3: {int((rules['lift'] > 3).sum()):,}",
        "",
        "Phan bo do dai antecedent:",
    ]
    for k, v in len_counts.items():
        summary_lines.append(f"- Len = {int(k)}: {int(v):,} luat")

    summary_file = BASE_DIR / "rules_evaluation_summary.txt"
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Da ve xong bieu do:")
    print(f"- {out1.name}")
    print(f"- {out2.name}")
    print(f"- {out3.name}")
    print(f"\nDa luu danh gia tong quan: {summary_file.name}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
