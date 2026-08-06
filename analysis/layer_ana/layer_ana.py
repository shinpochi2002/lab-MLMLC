import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def compute_E_from_branching(branching_factors):
    """
    branching_factors:
        例: [12, 4, 4, 6]
        - 最後の 6 は API 推薦層への分岐数
        - E の計算では API 推薦層を除外し、カテゴリ層のみを使う

    E = sum_l w_l * (N_l / N_L)
    ただし N_l は branching_factors から得る累積分類数を用いる
    """
    if len(branching_factors) < 2:
        raise ValueError("branching_factors must include at least one category layer and one API layer.")

    category_branch = branching_factors[:-1]  # API 層を除外
    L = len(category_branch)

    cumulative = []
    prod = 1
    for b in category_branch:
        prod *= b
        cumulative.append(prod)

    N_L = cumulative[-1]

    E = 0.0
    for l, N_l in enumerate(cumulative, start=1):
        w_l = (L - l + 1) / L
        E += w_l * (N_l / N_L)

    return E


# ============================================================
# branch: E計算用の分岐数系列（最後はAPI推薦層への分岐数）
# display: 表1に合わせた各段階の機能候補数（最後は909）
# f1: 最終API推薦F1
# ============================================================
data = [
    {"branch": [4, 38, 6],       "display": [4, 152, 909],             "f1": 0.433},
    {"branch": [6, 26, 6],       "display": [6, 152, 909],             "f1": 0.468},
    {"branch": [8, 19, 6],       "display": [8, 152, 909],             "f1": 0.463},
    {"branch": [12, 13, 6],      "display": [12, 152, 909],            "f1": 0.490},
    {"branch": [15, 10, 6],      "display": [15, 152, 909],            "f1": 0.429},
    {"branch": [24, 7, 6],       "display": [24, 152, 909],            "f1": 0.404},
    {"branch": [33, 5, 6],       "display": [33, 152, 909],            "f1": 0.433},
    {"branch": [51, 3, 6],       "display": [51, 152, 909],            "f1": 0.415},
    {"branch": [4, 4, 10, 6],    "display": [4, 16, 152, 909],         "f1": 0.471},
    {"branch": [8, 4, 5, 6],     "display": [8, 32, 152, 909],         "f1": 0.462},
    {"branch": [12, 2, 7, 6],    "display": [12, 24, 152, 909],        "f1": 0.470},
    {"branch": [12, 4, 4, 6],    "display": [12, 48, 152, 909],        "f1": 0.538},
    {"branch": [12, 6, 3, 6],    "display": [12, 72, 152, 909],        "f1": 0.455},
    {"branch": [16, 4, 3, 6],    "display": [16, 64, 152, 909],        "f1": 0.451},
    {"branch": [12, 2, 2, 4, 6], "display": [12, 24, 48, 152, 909],    "f1": 0.454},
    {"branch": [8, 3, 2, 4, 6],  "display": [8, 24, 48, 152, 909],     "f1": 0.413},
]

# ===== DataFrame作成 =====
records = []
for item in data:
    branch = item["branch"]
    display = item["display"]
    f1 = item["f1"]

    E = compute_E_from_branching(branch)

    # 図中ラベルは API候補数 909 を省略し、カテゴリ層のみ表示
    label_plot = "-".join(map(str, display))
    # 表1対応の全文字列（必要ならCSVに残す）
    label_full = ", ".join(map(str, display))

    records.append({
        "stages": len(display),                    # API推薦層込み段階数
        "branch_structure": "-".join(map(str, branch[:-1])),   # 参考用
        "display_structure_full": label_full,     # 表1対応
        "display_structure_plot": label_plot,     # 図1対応
        "E": E,
        "F1": f1
    })

df = pd.DataFrame(records)

# ===== 相関係数 =====
pearson_r = df["E"].corr(df["F1"], method="pearson")
spearman_rho = df["E"].corr(df["F1"], method="spearman")

print("=== Correlation ===")
print(f"Pearson r  = {pearson_r:.4f}")
print(f"Spearman ρ = {spearman_rho:.4f}")

# ===== 保存先 =====
out_dir = Path(".")
csv_path = out_dir / "E_F1_scatter_for_paper.csv"
png_path = out_dir / "figure_E_F1_scatter_revised_slide.png"
pdf_path = out_dir / "figure_E_F1_scatter_revised_slide.pdf"

df.to_csv(csv_path, index=False)

# ===== プロット =====
fig, ax = plt.subplots(figsize=(10, 5.625))

marker_map = {
    3: "o",
    4: "s",
    5: "^",
}

for stages in sorted(df["stages"].unique()):
    group = df[df["stages"] == stages].sort_values("E")

    ax.scatter(
        group["E"],
        group["F1"],
        label=f"{stages}-stage",
        marker=marker_map.get(stages, "o"),
        s=60
    )

    for _, row in group.iterrows():
        ax.annotate(
            f'({row["display_structure_plot"]})',
            xy=(row["E"], row["F1"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="bottom"
        )

ax.set_xlabel(r"Design indicator $E$")
ax.set_ylabel("Final API recommendation F1")
ax.legend(title="Number of stages")
ax.grid(True, alpha=0.25)
fig.tight_layout()

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.show()

print("=== Saved files ===")
print(f"CSV : {csv_path}")
print(f"PNG : {png_path}")
print(f"PDF : {pdf_path}")