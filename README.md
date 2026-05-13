# Cấu trúc thư mục

```
code/
├── 1_prepare/       # B1. Chuẩn bị dữ liệu
│   ├── preprocessing.py        # dataset.xlsx → data/transactions_ready_for_fpgrowth.csv
│   └── split_data.py           # chia train/test 80/20
│
├── 2_train/         # B2. Huấn luyện & đánh giá model
│   ├── train_model.py          # FP-Growth → recommendation_rules.pkl
│   ├── find_best_params.py     # dò min_support tối ưu
│   └── evaluate_model.py       # chấm Precision/Recall/F-measure trên test
│
├── 3_app/           # B3. App demo & Dashboard
│   ├── mining_app.py           # Streamlit: storefront + admin (chạy chính)
│   ├── admin_dashboard.py      # script xuất biểu đồ EDA + revenue
│   ├── group_transactions_demo.py   # demo bước gom giao dịch
│   └── binary_encoding_demo.py      # demo bước mã hóa 0-1
│
├── utils/           # Tiện ích / báo cáo
│   ├── pkltocsv.py                  # xuất luật pkl → csv đọc được
│   ├── count_rule_lengths.py        # thống kê độ dài antecedent
│   ├── visualize_rules.py           # biểu đồ phân bố luật
│   └── visualize_conf_generalization.py  # conf_train vs conf_test
│
├── rule_policy.py   # Module dùng chung (giữ ở root)
├── dataset.xlsx     # Dữ liệu gốc
├── recommendation_rules.pkl    # Model luật đã train
├── data/            # File CSV trung gian
└── images/          # PNG xuất ra
```

## Pipeline chạy lần đầu

```bash
python 1_prepare/preprocessing.py        # làm sạch dataset.xlsx
python 1_prepare/split_data.py           # chia train/test
python 2_train/train_model.py            # train model
python 2_train/evaluate_model.py         # đánh giá trên test
streamlit run 3_app/mining_app.py        # chạy web app
```

## Ghi chú

- Mỗi script trong subfolder đều có **bootstrap** tự chuyển CWD về project root, nên có thể chạy từ bất kỳ đâu (`python utils/visualize_rules.py` hay `cd utils && python visualize_rules.py` đều được).
- `rule_policy.py` cố tình giữ ở root vì cả `2_train/` và `3_app/` đều dùng chung.
