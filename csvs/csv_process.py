import pandas as pd
import ast


def list_length(cell):
    """
    セルの中身が list 表記文字列なら、その長さを返す。
    それ以外（NaN・不正形式）は 0。
    """
    if pd.isna(cell):
        return 0
    try:
        value = ast.literal_eval(cell)
        return len(value) if isinstance(value, list) else 0
    except (ValueError, SyntaxError):
        return 0


def add_list_length_column(df, source_col, new_col):
    """
    指定した列(source_col)からリスト長を計算し、
    新しい列(new_col)として DataFrame に追加する。
    """
    df[new_col] = df[source_col].apply(list_length)
    return df


def average_list_length(df, length_col):
    """
    リスト長が入った列の平均を返す。
    """
    return df[length_col].mean()



# CSV読み込み（ここだけ直書き）
df = pd.read_csv("/home/nakagawa/csvs/layer3_16_4_3_results.csv")

# 列名
col_a = "pred_apis"
col_b = "true_apis"

# 各列のリスト長を追加
df = add_list_length_column(df, col_a, "len_pred")
df = add_list_length_column(df, col_b, "len_true")

# 平均を計算
avg_pred = average_list_length(df, "len_pred")
avg_true = average_list_length(df, "len_true")

print("pred_apis 平均長:", avg_pred)
print("true_apis 平均長:", avg_true)
