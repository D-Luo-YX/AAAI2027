
def save_latex_table(summary_df, out_path):
    grouped = (
        summary_df.groupby(["dataset", "model", "method"])["test_acc"]
        .agg(["mean", "std"])
        .reset_index()
    )

    grouped["score"] = (
        (grouped["mean"] * 100).round(2).astype(str)
        + " $\\pm$ "
        + (grouped["std"].fillna(0) * 100).round(2).astype(str)
    )

    table = grouped.pivot_table(
        index=["dataset", "model"],
        columns="method",
        values="score",
        aggfunc="first",
    )

    with open(out_path, "w", encoding="utf-8") as file:
        file.write(table.to_latex(escape=False))
