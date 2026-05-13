import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import joblib
import time

from rule_policy import MAX_ANTECEDENT_ITEMS, filter_rules_recommend_format

# 1. Đọc dữ liệu
print("1. Đang tải dữ liệu...")
df = pd.read_csv('data/train_transactions.csv')

# Chuyển đổi cột Items_str (A|B|C) thành list thực sự ['A', 'B', 'C']
# Lưu ý: Cập nhật tên cột nếu ở bước trước bạn đặt tên khác
transactions = df['Items_str'].dropna().apply(lambda x: str(x).split('|')).tolist()

print(f"-> Đã tải {len(transactions)} hóa đơn.")

# 2. One-Hot Encoding
print("\n2. Đang mã hóa dữ liệu (Transaction Encoding)...")
start_time = time.time()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
print(f"-> Hoàn thành mã hóa trong {time.time() - start_time:.2f} giây.")

# 3. Chạy thuật toán FP-Growth
# Bắt đầu với min_support = 0.02 (sản phẩm phải xuất hiện trong ít nhất 2% số hóa đơn)
MIN_SUPPORT = 0.01
print(f"\n3. Đang chạy FP-Growth với min_support = {MIN_SUPPORT}...")
start_time = time.time()
frequent_itemsets = fpgrowth(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
print(f"-> Tìm thấy {len(frequent_itemsets)} tập phổ biến. (Thời gian: {time.time() - start_time:.2f} giây)")

if frequent_itemsets.empty:
    print(" Không tìm thấy tập phổ biến nào. Bạn hãy giảm MIN_SUPPORT xuống (VD: 0.015 hoặc 0.01)!")
else:
    # 4. Sinh luật kết hợp
    # Dùng độ đo Lift > 1.2 để lấy các luật có tương quan dương mạnh
    print("\n4. Đang sinh luật kết hợp và đánh giá...")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

    if rules.empty:
        print(" Có tập phổ biến nhưng không sinh được luật nào đạt chuẩn Lift > 1.2.")
    else:
        # So sánh tác động của các ngưỡng min_confidence để chọn mức phù hợp
        print("\n   --- So sánh ngưỡng min_confidence ---")
        print(f"   {'min_conf':>9} | {'#luật':>8} | {'conf_TB':>8} | {'lift_TB':>8} | {'sup_TB':>8}")
        for c in [0.1, 0.2, 0.3, 0.5]:
            r_c = rules[rules['confidence'] >= c]
            if len(r_c) > 0:
                print(
                    f"   {c:>9.2f} | {len(r_c):>8,} | "
                    f"{r_c['confidence'].mean():>8.3f} | "
                    f"{r_c['lift'].mean():>8.3f} | "
                    f"{r_c['support'].mean():>8.4f}"
                )
            else:
                print(f"   {c:>9.2f} | {0:>8} | {'-':>8} | {'-':>8} | {'-':>8}")
        print("   ------------------------------------------")

        # Áp ngưỡng confidence chính thức (đồng bộ với pipeline đánh giá StableRate)
        MIN_CONFIDENCE = 0.2
        rules = rules[rules['confidence'] >= MIN_CONFIDENCE].copy()
        print(f"-> Chọn min_confidence = {MIN_CONFIDENCE}: còn {len(rules):,} luật.")

        before = len(rules)
        rules = filter_rules_recommend_format(rules)
        print(
            f"   (Lọc dạng recommend: vế trái 1-{MAX_ANTECEDENT_ITEMS} món, vế phải 1 món: "
            f"{before} -> {len(rules)} luật)"
        )
        if rules.empty:
            print(" Sau khi lọc dạng luật, không còn luật nào. Thử giảm min_support hoặc min_lift.")

    if not rules.empty:
        # Sắp xếp luật theo Lift giảm dần (luật xịn nhất lên đầu)
        rules = rules.sort_values(by='lift', ascending=False)
        print(f" Thành công! Đã sinh ra {len(rules)} luật kết hợp chất lượng cao.")

        # In Top 10 luật theo Lift để kiểm tra nhanh chất lượng
        print("\n--- TOP 10 LUẬT THEO LIFT ---")
        for idx, row in rules.head(10).iterrows():
            A = set(row['antecedents'])
            B = set(row['consequents'])
            print(
                f"{A}  =>  {B}  "
                f"(Support: {row['support']:.4f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})"
            )

        # 5. Đóng gói mô hình cho Hệ thống
        # Lưu frozenset dưới dạng list để dễ xài trên API
        rules['antecedents'] = rules['antecedents'].apply(list)
        rules['consequents'] = rules['consequents'].apply(list)

        model_filename = 'recommendation_rules.pkl'
        joblib.dump(rules, model_filename)
        print(f"\n5. Đã xuất mô hình ra file: {model_filename}")
        print("=== BẠN ĐÃ SẴN SÀNG LẮP VÀO HỆ THỐNG MINING! ===")
