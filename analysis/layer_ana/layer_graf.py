import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========= 設定 =========
base_dir = Path("ver2")   # CSVが入っているディレクトリ
out_dir = Path(".")       # 画像保存先

groups = {
    "3-layers": [
        "layer2_2_4_results.csv",
        "layer2_2_6_results.csv",
        "layer2_2_8_results.csv",
        "layer2_2_12_results.csv",
        "layer2_2_15_results.csv",
        "layer2_2_24_results.csv",
        "layer2_2_33_results.csv",
        "layer2_2_51_results.csv",
    ],
    "4-layers": [
        "layer3_4_4_8_results.csv",
        "layer3_8_4_4_results.csv",
        "layer3_12_2_6_results.csv",
        "layer3_12_4_3_results.csv",
        "layer3_12_6_2_results.csv",
        "layer3_16_4_3_results.csv",
    ],
    "5-layers": [
        "layer4_8_3_2_3_results.csv",
        "layer4_12_2_2_3_results.csv",
    ],
}

# 表1に合わせた表示名
label_map = {
    "layer2_2_4_results.csv": "(4,152,909)",
    "layer2_2_6_results.csv": "(6,152,909)",
    "layer2_2_8_results.csv": "(8,152,909)",
    "layer2_2_12_results.csv": "(12,152,909)",
    "layer2_2_15_results.csv": "(15,152,909)",
    "layer2_2_24_results.csv": "(24,152,909)",
    "layer2_2_33_results.csv": "(33,152,909)",
    "layer2_2_51_results.csv": "(51,152,909)",

    "layer3_4_4_8_results.csv": "(4,16,152,909)",
    "layer3_8_4_4_results.csv": "(8,32,152,909)",
    "layer3_12_2_6_results.csv": "(12,24,152,909)",
    "layer3_12_4_3_results.csv": "(12,48,152,909)",
    "layer3_12_6_2_results.csv": "(12,72,152,909)",
    "layer3_16_4_3_results.csv": "(16,64,152,909)",

    "layer4_8_3_2_3_results.csv": "(8,24,48,152,909)",
    "layer4_12_2_2_3_results.csv": "(12,24,48,152,909)",
}

# 各段階のbest構成
highlight = {
    "3-layers": {"layer2_2_12_results.csv"},
    "4-layers": {"layer3_12_4_3_results.csv"},
    "5-layers": {"layer4_12_2_2_3_results.csv"},
}

def get_stagewise_f1(df: pd.DataFrame):
    """
    layer1_f1, layer2_f1, ... に加えて
    最後に最終API推薦の f1 を追加して返す
    """
    layer_f1_cols = [c for c in df.columns if c.startswith("layer") and c.endswith("_f1")]
    layer_f1_cols = sorted(
        layer_f1_cols,
        key=lambda x: int(x.replace("layer", "").replace("_f1", ""))
    )

    y = [df[col].mean() for col in layer_f1_cols]

    if "f1" not in df.columns:
        raise ValueError(f"'f1' column not found in CSV: {df.columns.tolist()}")

    y.append(df["f1"].mean())  # 最後に API 推薦 F1
    return y

def get_colors(n):
    """
    構成数に応じて見分けやすい色を返す
    """
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n)]

# ========= 各段階ごとに個別出力 =========
for stage_name, file_list in groups.items():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    colors = get_colors(len(file_list))
    max_len = 0

    for i, filename in enumerate(file_list):
        csv_path = base_dir / filename
        df = pd.read_csv(csv_path)

        y = get_stagewise_f1(df)
        x = list(range(1, len(y) + 1))
        max_len = max(max_len, len(y))

        color = colors[i]

        if filename in highlight[stage_name]:
            ax.plot(
                x, y,
                marker="o",
                linewidth=3.0,
                markersize=7,
                color=color,
                alpha=1.0,
                label=label_map[filename],
                zorder=3
            )
        else:
            ax.plot(
                x, y,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=color,
                alpha=0.85,
                label=label_map[filename],
                zorder=2
            )

    tick_labels = [f"layer{i}" for i in range(1, max_len)] + ["API"]
    ax.set_xticks(list(range(1, max_len + 1)))
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Average F1")
    ax.set_ylim(0.4, 0.75) 
    ax.grid(True, alpha=0.3)

    # タイトルは必要なら使う
    # ax.set_title(stage_name)

    ax.legend(
        fontsize=8,
        loc="best",
        frameon=True
    )

    fig.tight_layout()

    safe_name = stage_name.replace("-", "_")
    png_path = out_dir / f"figure_stagewise_f1_{safe_name}_colored_4.png"
    pdf_path = out_dir / f"figure_stagewise_f1_{safe_name}_colored_4.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")