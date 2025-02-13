import pandas as pd
import numpy as np
import random
import discriminative_power as dp


def jaccard_distance(minhash1, minhash2):
    intersection = np.sum(minhash1 == minhash2)
    union = len(minhash1)
    return 1 - (intersection / union)


df1 = pd.read_parquet(f"./data/Amazon_embedded_minhash_all.pqt")
df2 = pd.read_parquet(f"./data/Google_embedded_minhash_all.pqt")
gold_standard = pd.read_csv("./data/truth_Amazon_googleProducts.csv", sep=",", encoding="utf-8", keep_default_na=False)

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

names_diff = []
descriptions_diff =[]
for _, row in data.iterrows():
    id1 = row.iloc[0]
    id2 = row.iloc[1]
    label = row.iloc[2]
    name1 =  df1[df1["id"] == id1]["name"].values[0]
    name2 = df2[df2["id"] == id2]["name"].values[0]
    name_diff = jaccard_distance(name1, name2)
    names_diff.append(name_diff)
    description1 =  df1[df1["id"] == id1]["description"].values[0]
    description2 = df2[df2["id"] == id2]["description"].values[0]
    description_diff = jaccard_distance(description1, description2)
    descriptions_diff.append(description_diff)

    y.append(label)
data["name"] = names_diff
data["description"] = descriptions_diff
data=data.reset_index()
data=data.drop("index", axis=1)

y = data["match"]
data = data.drop(columns=["match","id1","id2"])
X = data

keys, importance = dp.discriminative(X, y, ["name","description"])
print("shap_importance", importance, keys)


