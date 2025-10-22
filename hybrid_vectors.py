FASTTEXT_MODEL_PATH = "cc.en.300.bin"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from gensim.models.fasttext import load_facebook_vectors
from torch.nn.functional import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
import faiss
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------------------------
# 1. Load Abt, Buy, and Gold Standard CSV files
# ---------------------------------------------
abt = pd.read_csv("./data/Abt.csv",sep=",", encoding="unicode_escape")
buy = pd.read_csv("./data/Buy.csv", sep=",", encoding="unicode_escape")
gs = pd.read_csv("./data/truth_abt_buy.csv",sep=",", encoding="utf-8",keep_default_na=False)

# Expected columns:
# abt: idAbt, name, description, price
# buy: idBuy, name, description, price
# gs: idAbt, idBuy

# Clean text
def clean_text(t):
    return str(t).lower().replace("-", " ").replace("_", " ")

abt["text"] = (abt["name"].fillna("") + " " + abt["description"].fillna("")).apply(clean_text)
buy["text"] = (buy["name"].fillna("") + " " + buy["description"].fillna("")).apply(clean_text)

# Index by ID for fast lookup
abt = abt.set_index("id")
buy = buy.set_index("id")

# ---------------------------------------------
# 2. Load models
# ---------------------------------------------
print("Loading FastText and MiniLM models...")
fasttext = load_facebook_vectors(FASTTEXT_MODEL_PATH)   # https://fasttext.cc/docs/en/crawl-vectors.html
sem_model = SentenceTransformer("all-MiniLM-L6-v2")
#sem_model = SentenceTransformer("google/gemma-3-270m-it")

# ---------------------------------------------
# 3. Define embedding functions
# ---------------------------------------------
def get_fasttext_vector(text):
    words = text.split()
    vectors = [fasttext[w] for w in words if w in fasttext]
    if len(vectors) == 0:
        return np.zeros(fasttext.vector_size)
    return np.mean(vectors, axis=0)

def get_semantic_vector(text):
    return sem_model.encode(text, normalize_embeddings=True)

def embed_record(text):
    return get_fasttext_vector(text), get_semantic_vector(text)

# ---------------------------------------------
# 4. Define Gated Fusion Module
# ---------------------------------------------
class GatedFusion(nn.Module):
    def __init__(self, dim_lex=300, dim_sem=384, dim_out=384):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim_lex + dim_sem, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
            nn.Sigmoid()
        )
        self.proj_lex = nn.Linear(dim_lex, dim_out)
        self.proj_sem = nn.Linear(dim_sem, dim_out)

    def forward(self, lex, sem):
        gate = self.gate(torch.cat([lex, sem], dim=1))
        fused = gate * self.proj_lex(lex) + (1 - gate) * self.proj_sem(sem)
        return fused

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion = GatedFusion().to(device)
fusion.eval()

# ---------------------------------------------
# 5. Embed all records (store to dict)
# ---------------------------------------------
print("Embedding Abt records...")
abt_lex = {}
abt_sem = {}
for idx, row in tqdm(abt.iterrows(), total=len(abt)):
    lex, sem = embed_record(row["text"])
    abt_lex[idx] = lex
    abt_sem[idx] = sem

print("Embedding Buy records...")
buy_lex = {}
buy_sem = {}
for idx, row in tqdm(buy.iterrows(), total=len(buy)):
    lex, sem = embed_record(row["text"])
    buy_lex[idx] = lex
    buy_sem[idx] = sem

# ---------------------------------------------
# 6. Compute hybrid embeddings
# ---------------------------------------------
def fuse(lex_np, sem_np):
    lex_t = torch.tensor(lex_np, dtype=torch.float32).unsqueeze(0).to(device)
    sem_t = torch.tensor(sem_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return fusion(lex_t, sem_t).cpu().numpy().flatten()

print("Creating hybrid embeddings...")
abt_hybrid = {k: fuse(abt_lex[k], abt_sem[k]) for k in tqdm(abt.index)}
buy_hybrid = {k: fuse(buy_lex[k], buy_sem[k]) for k in tqdm(buy.index)}

# ---------------------------------------------
# 7. Evaluate similarity on Gold Standard pairs
# ---------------------------------------------
def cosine_sim(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

sims = []
for _, row in gs.iterrows():
    a_id, b_id = row["idAbt"], row["idBuy"]
    if a_id in abt_hybrid and b_id in buy_hybrid:
        sim = cosine_sim(abt_hybrid[a_id], buy_hybrid[b_id])
        sims.append(sim)

print(f"Mean similarity for true pairs: {np.mean(sims):.4f}")

# Optionally generate negatives to check separation
import random
neg_sims = []
for _ in range(len(gs)):
    a_id = random.choice(list(abt.index))
    b_id = random.choice(list(buy.index))
    neg_sims.append(cosine_sim(abt_hybrid[a_id], buy_hybrid[b_id]))



# Ensure consistent ordering
abt_ids = list(abt_hybrid.keys())
buy_ids = list(buy_hybrid.keys())

abt_matrix = np.vstack([abt_hybrid[i] for i in abt_ids]).astype('float32')
buy_matrix = np.vstack([buy_hybrid[i] for i in buy_ids]).astype('float32')

# -------------------------------------------------
# 2. Build FAISS HNSW index for Abt vectors
# -------------------------------------------------
d = abt_matrix.shape[1]
print(f"The dimensionality is {d}.")
m = 32            # number of bi-directional links per node (tradeoff: memory vs recall)
ef_construction = 100

index = faiss.IndexHNSWFlat(d, m)
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = 128  # can increase to 128–256 for better recall

print("Building HNSW index...")
index.add(abt_matrix)

# -------------------------------------------------
# 3. Query Buy vectors to find top-k candidates
# -------------------------------------------------
k = 5
print(f"Querying {len(buy_matrix)} Buy items...")
D, I = index.search(buy_matrix, k)  # D: distances, I: indices of nearest neighbors

# Map indices back to IDs
predictions = []
print(I)
for q_idx, abt_idx in enumerate(I[:, ]):
    buy_id = buy_ids[q_idx]
    for abt_idx in abt_idx:
        abt_id = abt_ids[abt_idx]
        predictions.append((abt_id, buy_id))

pred_df = pd.DataFrame(predictions, columns=["idAbt_pred", "idBuy"])

# -------------------------------------------------
# 4. Evaluate vs. gold standard
# -------------------------------------------------
# Convert to sets for fast lookup
gold_pairs = set([tuple(x) for x in gs[["idAbt", "idBuy"]].values])
pred_pairs = set([tuple(x) for x in pred_df.values])

tp = len(gold_pairs & pred_pairs)
fp = len(pred_pairs - gold_pairs)
fn = len(gold_pairs - pred_pairs)

precision = tp / (tp + fp + 1e-10)
recall = tp / (tp + fn + 1e-10)
f1 = 2 * precision * recall / (precision + recall + 1e-10)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")



