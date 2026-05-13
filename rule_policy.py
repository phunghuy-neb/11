"""Quy ước luật cho gợi ý: A=>B, A,B=>C, A,B,C=>D (vế phải luôn 1 món)."""

import pandas as pd

# Số món tối đa ở vế trái (antecedent). Tối thiểu luôn >= 1 với luật chuẩn.
MAX_ANTECEDENT_ITEMS = 3


def filter_rules_recommend_format(rules: pd.DataFrame) -> pd.DataFrame:
    """Giữ luật có 1..MAX_ANTECEDENT_ITEMS món bên trái và đúng 1 món bên phải."""
    if rules.empty:
        return rules
    ant_len = rules["antecedents"].apply(len)
    con_len = rules["consequents"].apply(len)
    return rules[(ant_len >= 1) & (ant_len <= MAX_ANTECEDENT_ITEMS) & (con_len == 1)].copy()
    