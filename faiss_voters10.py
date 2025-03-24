import pandas as pd
import numpy as np
import faiss
import time
from faiss import write_index, read_index
import math

if __name__ == '__main__':
    model = "minhash"
    df1 = pd.read_parquet(f"./data/voters10_{model}_A.pqt")
    df2 = pd.read_parquet(f"./data/voters10_{model}_B.pqt")




    #dfv1 =  pd.read_parquet(f"./data/voters10_mini.pqt")
    #print("len dfv1=", len(dfv1))


    N = 10_000_000

    d = 128
    index1 = faiss.IndexBinaryHNSW(d, 64)
    index1.metric_type = faiss.METRIC_Jaccard
    index1.hnsw.efConstruction = 64
    index1.hnsw.efSearch = 128



    no_batch = 1_000_000
    batches = N  // no_batch
    st1 = 0

    #print(batches)
    start_t = time.time()
    for j in range(batches):
        if j == batches - 1:
             no_batch = no_batch + (N % no_batch)
        print(f"inserting from {st1} to {st1+no_batch-1}")
        vectors = df1.loc[st1:st1+no_batch-1, 'v'].tolist()
        data = np.array(vectors).astype(np.uint8)
        byte_data = np.packbits(data, axis=-1)
        index1.add(byte_data)
        st1 = st1+no_batch
    end_t= time.time()
    total_time = end_t - start_t
    print(f"Total time for building the index {round(total_time,3)}")

    matches = 3_000_000
    n = 0
    k = 10
    tp = 0
    fp = 0
    no_batch =  50_000
    N = len(df2)
    batches = N  // no_batch
    st1 = 0
    qtimes = []
    i = 0
    ids = set()
    for j in range(batches):
          start_t = time.time()
          if j == batches - 1:
               no_batch = no_batch + (N % no_batch)
          print(f"querying from {st1} to {st1+no_batch-1}")

          vectors = df2.loc[st1:st1+no_batch-1, 'v'].tolist()
          data = np.array(vectors).astype(np.uint8)
          byte_data = np.packbits(data, axis=-1)
          distances, replies = index1.search(byte_data, k)  #index.search(query, k)
          distances = distances/128
          for rs, dss in zip(replies, distances):
            df02 = df2.loc[i, ]
            id2 = df02["id"]

            #true_count = df02["true_count"]
            i+=1
            #ltp = 0
            for ind, ds in zip(rs, dss):
                df01 = df1.loc[ind, ]
                id1 = df01["id"]

                if ds < 0.45:
                   if id2 == id1:
                        tp+=1
                   else:
                        fp+=1

          st1 = st1+no_batch
          end_t= time.time()
          qtimes.append(end_t - start_t)

    mean_q = np.mean(qtimes)
    print(f"recall={round(tp/matches, 2)} precision={round(tp/(tp+fp), 2)} query time for batches of {no_batch} records={round(mean_q,4)}")





