import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# GitHub Data
data = {
    "hot_words": [
        "error", "bug", "version", "image", "file", "assets", "data", "code", "issue", "api"
    ],
    "2024-Q1": [244, 203, 187, 197, 187, 179, 177, 174, 174, 134],
    "2024-Q2": [261, 192, 189, 179, 191, 200, 162, 139, 144, 157],
    "2024-Q3": [232, 215, 206, 149, 166, 151, 151, 110, 97, 91],
}

data_2023 = {
    "hot_words": [
        "error", "bug", "version", "image", "file", "assets", "data", "code", "issue", "api"
    ],
    "2023-Q1": [235, 161, 196, 124, 166, 170, 178, 134, 118, 90],
    "2023-Q2": [223, 194, 168, 157, 164, 141, 123, 139, 151, 86],
    "2023-Q3": [237, 204, 212, 159, 193, 148, 160, 158, 152, 79],
    "2023-Q4": [215, 159, 172, 163, 143, 166, 122, 116, 132, 117],
}

df_2024 = pd.DataFrame(data)
df_2023 = pd.DataFrame(data_2023)
df = pd.merge(df_2023, df_2024, on="hot_words", how="outer").fillna(0)
df.fillna(0, inplace=True)
heatmap_data = df.set_index("hot_words").T

plt.figure(figsize=(12, 9))
plt.title("GitHub Hot Words Trend 2023-2024", fontsize=20)
sns.heatmap(
    heatmap_data.T,
    annot=True,
    fmt="g",
    cmap="YlGnBu",
    cbar=True,
    linewidths=0.5,
    linecolor="gray",
    square=False
)

plt.xlabel("Timeline", fontsize=18)
plt.ylabel("Hot words", fontsize=18)
plt.xticks(fontsize=16, rotation=45)
plt.yticks(fontsize=16, rotation=0)
plt.tight_layout()
plt.savefig(f"result/github/github-trend.png")
plt.show()


# StackOverflow Data
data_so = {
    "hot_words": [
        "error", "code", "data", "file", "import", "function", "value", "class", "type", "public"
    ],
    "2023-Q1": [1414, 1280, 1242, 1122, 1056, 1002, 845, 828, 710, 665],
    "2023-Q2": [1477, 1439, 1268, 1357, 1266, 1027, 872, 897, 755, 661],
    "2023-Q3": [1576, 1349, 1352, 1283, 1237, 1058, 846, 867, 701, 883],
    "2023-Q4": [1797, 1413, 1220, 1417, 1421, 1044, 794, 831, 858, 781],
    "2024-Q1": [1968, 1566, 1534, 1601, 1603, 1313, 959, 1044, 938, 948],
}

df_so = pd.DataFrame(data_so)
filtered_heatmap_data = df_so.set_index("hot_words").T

plt.figure(figsize=(12, 9))
plt.title("StackOverflow Hot Words Trend 2023-2024", fontsize=20)
sns.heatmap(
    filtered_heatmap_data.T,
    annot=True,
    fmt="g",
    cmap="YlGnBu",
    cbar=True,
    linewidths=0.5,
    linecolor="gray",
    square=False
)

plt.xlabel("Timeline", fontsize=18)
plt.ylabel("Hot words", fontsize=18)
plt.xticks(fontsize=16, rotation=45)
plt.yticks(fontsize=16, rotation=0)
plt.tight_layout()
plt.savefig(f"result/stackoverflow/stackoverflow-trend.png")
plt.show()


