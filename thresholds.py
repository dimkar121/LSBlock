import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
import random



def jaccard_distance(minhash1, minhash2):
    intersection = np.sum(minhash1 == minhash2)
    union = len(minhash1)
    return 1 - (intersection / union)



def opt_jaccard_threshold(df1, df2, gold_standard):
  X = []
  y = []
  sample_size = len(gold_standard) // 2
  # Sample matching pairs
  matched_pairs = gold_standard.sample(n=sample_size, random_state=42)
  matched_pairs = matched_pairs.rename(columns={
    matched_pairs.columns[0]: 'id1',  # Rename first column
    matched_pairs.columns[1]: 'id2'   # Rename second column
  } )
  # Generate non-matching pairs
  df1_ids = df1['id'].tolist()
  df2_ids = df2['id'].tolist()
  existing_pairs = set(zip(gold_standard.iloc[:,0], gold_standard.iloc[:,1]))
    
  non_matched = []
  attempts = 0
  max_attempts = sample_size * 100  # Prevent infinite loop
  while len(non_matched) < sample_size and attempts < max_attempts:
        id1 = random.choice(df1_ids)
        id2 = random.choice(df2_ids)
        if (id1, id2) not in existing_pairs and (id1, id2) not in non_matched:
            non_matched.append((id1, id2))
        attempts += 1

  non_matched_pairs = pd.DataFrame(non_matched, columns=["id1","id2"])
  non_matched_pairs["match"] = 0
  matched_pairs["match"] = 1
  #print(matched_pairs.columns, non_matched_pairs.columns)
  fd = pd.concat([matched_pairs, non_matched_pairs])
  #print(fd)
  

  for _, row in fd.iterrows():
    id1 = row.iloc[0]
    id2 = row.iloc[1]
    label = row.iloc[2]

    #print(id1, id2)
    minhash1 = df1[df1["id"] == id1]["v"].values[0]
    minhash2 = df2[df2["id"] == id2]["v"].values[0]
    #print(minhash1)
    #print(minhash2)

    distance = jaccard_distance(minhash1, minhash2)    
    X.append(distance)
    y.append(label)

  X  = np.array(X).reshape(-1,1) # .reshape(-1, 1)  # Reshape for sklearn/neural network
  y = np.array(y)
  threshold = choose_threshold(X, y)
  return threshold


def opt_inner_threshold(df1, df2, gold_standard):
   X = []
   y = []
   sample_size = len(gold_standard) // 2
   # Sample matching pairs
   matched_pairs = gold_standard.sample(n=sample_size, random_state=42)
   matched_pairs = matched_pairs.rename(columns={
     matched_pairs.columns[0]: 'id1',  # Rename first column
     matched_pairs.columns[1]: 'id2'   # Rename second column
   } )
   # Generate non-matching pairs
   df1_ids = df1['id'].tolist()
   df2_ids = df2['id'].tolist()
   existing_pairs = set(zip(gold_standard.iloc[:,0], gold_standard.iloc[:,1]))

   non_matched = []
   attempts = 0
   max_attempts = sample_size * 100  # Prevent infinite loop
   while len(non_matched) < sample_size and attempts < max_attempts:
         id1 = random.choice(df1_ids)
         id2 = random.choice(df2_ids)
         if (id1, id2) not in existing_pairs and (id1, id2) not in non_matched:
             non_matched.append((id1, id2))
         attempts += 1

   non_matched_pairs = pd.DataFrame(non_matched, columns=["id1","id2"])
   non_matched_pairs["match"] = 0
   matched_pairs["match"] = 1
   #print(matched_pairs.columns, non_matched_pairs.columns)
   fd = pd.concat([matched_pairs, non_matched_pairs])
   #print(fd)
 
   for _, row in fd.iterrows():
      id1 = row.iloc[0]
      id2 = row.iloc[1]
      label = row.iloc[2]
 
      v1 = df1[df1["id"] == id1]["v"].values[0]
      v2 = df2[df2["id"] == id2]["v"].values[0]
 
      sim = np.inner(v1, v2)
      X.append(sim)
      y.append(label)
 
   X  = np.array(X).reshape(-1,1) 
   y = np.array(y)
   threshold = choose_threshold(X, y)
   return threshold



def choose_threshold(X, y):  
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

  # Convert to PyTorch tensors
  X_train = torch.tensor(X_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
  X_val = torch.tensor(X_val, dtype=torch.float32)
  y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)


  # Define a simple neural network
  class MatchPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear( 1, 16)
        self.fc2 = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

  # Initialize model, loss, and optimizer
  model = MatchPredictor()
  criterion = nn.BCELoss()  # Binary cross-entropy loss
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Training loop
  num_epochs = 10
  for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")


  # Predict probabilities for the validation set
  with torch.no_grad():
    val_probs = model(X_val).numpy()

  # Find the optimal threshold using precision-recall curve
  precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
  f1_scores = 2 * (precision * recall) / (precision + recall)
  optimal_idx = np.argmax(f1_scores)
  optimal_threshold = thresholds[optimal_idx]

  print(f"Optimal Threshold: {optimal_threshold:.4f}")
  #print(f"Max F1-Score: {f1_scores[optimal_idx]:.4f}")
  return optimal_threshold

