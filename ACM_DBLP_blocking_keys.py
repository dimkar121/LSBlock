import pandas as pd
import numpy as np
import random
import discriminative_power as dp


def jaccard_distance(minhash1, minhash2):
    intersection = np.sum(minhash1 == minhash2)
    union = len(minhash1)
    return 1 - (intersection / union)


df2 = pd.read_parquet(f"./data/ACM_embedded_minhash_all.pqt")
df1 = pd.read_parquet(f"./data/DBLP_embedded_minhash_all.pqt")
gold_standard = pd.read_csv("./data/truth_ACM_DBLP.csv", sep=",", encoding="utf-8", keep_default_na=False)

X = []
y = []
sample_size = len(gold_standard) // 2
matched_pairs = gold_standard.sample(n=sample_size, random_state=42)
matched_pairs = matched_pairs.rename(columns={
matched_pairs.columns[0]: 'id1',  # Rename first column
matched_pairs.columns[1]: 'id2',  # Rename second column
})
df1_ids = df1['id'].tolist()
df2_ids = df2['id'].tolist()
existing_pairs = set(zip(gold_standard.iloc[:, 0], gold_standard.iloc[:, 1]))
non_matched = []
attempts = 0
max_attempts = sample_size * 100  # Prevent infinite loop
while len(non_matched) < sample_size and attempts < max_attempts:
    id1 = random.choice(df1_ids)
    id2 = random.choice(df2_ids)
    if (id1, id2) not in existing_pairs and (id1, id2) not in non_matched:
        non_matched.append((id1, id2))
    attempts += 1
non_matched_pairs = pd.DataFrame(non_matched,  columns=["id1", "id2"])  # ,"title1","title2","authors1","authors2","venue1","venue2"])
non_matched_pairs["match"] = 0
matched_pairs["match"] = 1
data = pd.concat([matched_pairs, non_matched_pairs])
data=data.reset_index()
data=data.drop("index", axis=1)

titles_diff = []
venues_diff =[]
authorss_diff =[]
for _, row in data.iterrows():
    id1 = row.iloc[0]
    id2 = row.iloc[1]
    label = row.iloc[2]
    title1 =  df1[df1["id"] == id1]["title"].values[0]
    title2 = df2[df2["id"] == id2]["title"].values[0]
    title_diff = jaccard_distance(title1, title2)
    titles_diff.append(title_diff)
    venue1 =  df1[df1["id"] == id1]["venue"].values[0]
    venue2 = df2[df2["id"] == id2]["venue"].values[0]
    venue_diff = jaccard_distance(venue1, venue2)
    venues_diff.append(venue_diff)
    authors1 =  df1[df1["id"] == id1]["authors"].values[0]
    authors2 = df2[df2["id"] == id2]["authors"].values[0]
    authors_diff = jaccard_distance(authors1, authors2)
    authorss_diff.append(authors_diff)
    y.append(label)
data["title"] = titles_diff
data["venue"] = venues_diff
data["authors"] = authorss_diff
data=data.reset_index()
data=data.drop("index", axis=1)
y = data["match"]
data = data.drop(columns=["match","id1","id2"])
X = data

keys, importance = dp.discriminative(X, y, ["title", "venue","authors"])
print("shap_importance", importance, keys)


