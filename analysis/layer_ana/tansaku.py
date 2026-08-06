import csv

def compute_E_from_cumulative(cumulative):
    L = len(cumulative)
    N_L = cumulative[-1]
    E = 0.0
    for l, N_l in enumerate(cumulative, start=1):
        w_l = (L - l + 1) / L
        E += w_l * (N_l / N_L)
    return E

def search_by_E_stop_on_exceed(
    E_min,
    E_max,
    L_min=2,
    L_max=7,
    b_min=2,
    b_max=72,
    threshold=151,
    max_over=50,
    top_k=500
):
    results = []

    def push(branches, cumulative):
        E = compute_E_from_cumulative(cumulative)
        if E_min <= E <= E_max:
            results.append({
                "branches": tuple(branches),
                "L": len(branches),
                "cumulative": list(cumulative),
                "final_categories": cumulative[-1],
                "E": E,
                "abs_diff": min(abs(E - E_min), abs(E - E_max))
            })

    def dfs(branches, cumulative, prod_now):
        if prod_now > threshold:
            if L_min <= len(branches) <= L_max and prod_now <= threshold + max_over:
                push(branches, cumulative)
            return

        if len(branches) == L_max:
            return

        for b in range(b_min, b_max + 1):
            new_prod = prod_now * b
            if new_prod > threshold + max_over:
                continue

            branches.append(b)
            cumulative.append(new_prod)
            dfs(branches, cumulative, new_prod)
            cumulative.pop()
            branches.pop()

    dfs([], [], 1)

    results.sort(key=lambda x: (x["abs_diff"], x["final_categories"], x["L"]))
    return results[:top_k]

def save_candidates_to_csv(cands, filepath):
    fieldnames = [
        "branches",
        "cumulative",
        "L",
        "final_categories",
        "E",
        "abs_diff"
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for c in cands:
            writer.writerow({
                "branches": ",".join(map(str, c["branches"])),
                "cumulative": ",".join(map(str, c["cumulative"])),
                "L": c["L"],
                "final_categories": c["final_categories"],
                "E": c["E"],
                "abs_diff": c["abs_diff"],
            })

cands = search_by_E_stop_on_exceed(
    E_min=0.55,
    E_max=0.65,
    max_over=20,
    top_k=500
)

for c in cands:
    print(
        c["branches"],
        c["cumulative"],
        c["final_categories"],
        round(c["E"], 4)
    )

save_candidates_to_csv(
    cands,
    "layer_candidates_E_0.55_0.65_ver2.csv"
)
