import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt
import time

from rule_policy import filter_rules_recommend_format

print("1. Đang tải và mã hóa dữ liệu...")
df = pd.read_csv('data/train_transactions.csv')
transactions = df['Items_str'].dropna().apply(lambda x: str(x).split('|')).tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 2. Cấu hình các tham số cần thử nghiệm
# Dò từ 5% xuống 0.8%
support_thresholds = [0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.008] 
fixed_confidence = 0.2
fixed_lift = 1.2

results = []

print(f"\n2. Bắt đầu dò tìm với Confidence >= {fixed_confidence} và Lift >= {fixed_lift}...\n")
print(f"{'Min Support':<15} | {'Số tập phổ biến':<20} | {'Số luật sinh ra':<15} | {'Thời gian (s)'}")
print("-" * 75)

# 3. Chạy vòng lặp thử nghiệm
for min_sup in support_thresholds:
    start_time = time.time()
    
    # Tìm tập phổ biến
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_sup, use_colnames=True)
    
    if frequent_itemsets.empty:
        num_itemsets = 0
        num_rules = 0
    else:
        num_itemsets = len(frequent_itemsets)
        # Sinh luật
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=fixed_lift)
        # Lọc thêm bằng confidence
        rules = rules[rules['confidence'] >= fixed_confidence]
        rules = filter_rules_recommend_format(rules)
        num_rules = len(rules)
        
    elapsed_time = time.time() - start_time
    results.append({'Support': min_sup, 'Rules': num_rules})
    
    print(f"{min_sup:<15.3f} | {num_itemsets:<20} | {num_rules:<15} | {elapsed_time:.2f}")

# 4. Vẽ biểu đồ trực quan hóa
print("\n3. Đang vẽ biểu đồ...")
res_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
# Đảo ngược trục X vì ta dò từ to xuống nhỏ
plt.plot(res_df['Support'], res_df['Rules'], marker='o', linestyle='-', color='b', linewidth=2)
plt.gca().invert_xaxis() 

plt.title('Biểu đồ: Số lượng luật kết hợp theo Min Support', fontsize=14)
plt.xlabel('Min Support (Ngưỡng hỗ trợ tối thiểu)', fontsize=12)
plt.ylabel('Số lượng luật sinh ra', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Gắn nhãn số liệu lên từng điểm
for i, txt in enumerate(res_df['Rules']):
    plt.annotate(txt, (res_df['Support'].iloc[i], res_df['Rules'].iloc[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('support_vs_rules_chart.png')
print("✅ Đã lưu biểu đồ thành file: 'support_vs_rules_chart.png'")
plt.show()