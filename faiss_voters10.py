import pandas as pd
import numpy as np
import faiss
import time
from faiss import write_index, read_index

if __name__ == '__main__':
    model = "minhash"
    df1 = pd.read_parquet(f"./data/voters10_{model}.pqt")
   
    id_counts = df1["id"].value_counts()
    #print("max=",np.max(id_counts))


    duplicate_ids = id_counts[id_counts >= 2].index

    df_duplicates = df1[df1["id"].isin(duplicate_ids)]

    #df2 = df1[df1["id"].isin(duplicate_ids)].reset_index(drop=True)
    df2 = df1[df1["id"].isin(duplicate_ids)].drop_duplicates(subset="id").reset_index(drop=True)
    print("Number of rows of the initial data set", len(df1))
    df2["true_count"] = df2["id"].map(id_counts) - 1


    print("Number of rows of the query data set", len(df2))
    duplicate_ids = id_counts[id_counts >= 2]
    total_instances = duplicate_ids.sum() - duplicate_ids.count()
    total_instances2 =  df2["true_count"].sum()
    print("matches:", total_instances2)
    matches = total_instances2

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
    #start_t = time.time()
    #faiss.write_index_binary(index1, "./data/voters10.index")
    #end_t= time.time()
    #total_time = end_t - start_t 
    #print(f"Total time for persisting the index {round(total_time,3)}")
    
    #index1 = faiss.read_index_binary("./data/voters10.index")

    n = 0
    k = 10
    tp = 0
    fp = 0
    no_batch =  50_000
    #batches = N  // no_batch 
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
          #print(data.shape)
          byte_data = np.packbits(data, axis=-1)
          distances, replies = index1.search(byte_data, k)  #index.search(query, k)
          
          for rs, dss in zip(replies, distances):
            df02 = df2.loc[i, ]
            id2 = df02["id"]    
            true_count = df02["true_count"]
            i+=1
            ltp = 0        
            print(len(rs))
            for ind, ds in zip(rs, dss):
                df01 = df1.loc[ind, ]
                id1 = df01["id"]
                #if ind in tps:
                 #   continue
                #if ind == i:
                 #   continue
                
                if id2 == id1:                         
                     #aprint(f"    found {id2} for {id1}")
                     tp+=1
                     ltp+=1
                     #print("=======================================================================")
                     #print(f"Searching {id}  {givenname} {surname} {suburb} {postcode}")
                     #print("Retrieved author",df0["id"], df0["givenname"],df0["surname"], df0["suburb"], df0["postcode"], ds)
                else:
                    fp+=1 
            #print(f"true_count={true_count} found {ltp} {ltp/true_count}  ")    

          st1 = st1+no_batch   
          end_t= time.time()
          qtimes.append(end_t - start_t)

    mean_q = np.mean(qtimes)
    print(f"recall={round(tp/matches, 2)} precision={round(tp/(tp+fp), 2)} query time={round(mean_q,4)}")






