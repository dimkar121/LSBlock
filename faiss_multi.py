import pandas as pd
import numpy as np
import faiss
import time
import multiprocessing
import pyarrow.parquet as pq


if __name__ == '__main__':

    num_cores = multiprocessing.cpu_count()

    print(f"Number of CPU cores: {num_cores}")

    '''
    df1 = pd.read_parquet(f"./data/voters10_minhash_A.pqt")
    chunks = np.array_split(df1, num_cores)
    for i, chunk in enumerate(chunks):
      chunk.to_parquet(f"./data/data_chunk_{i}.pqt")  # Save as Parquet
      print(f"Saved chunk {i} to data_chunk_{i}.pqt")
    exit(1)
    '''

    def build(i, queue):
           df1 = pd.read_parquet(f"./data/data_chunk_{i}.pqt")  # Load chunk
           vectors = df1.loc[: , 'v'].tolist()
           data = np.array(vectors).astype(np.uint8)
           byte_data = np.packbits(data, axis=-1)           
           index = faiss.IndexBinaryHNSW(128, 64)
           index.metric_type = faiss.METRIC_Jaccard
           index.hnsw.efConstruction = 64
           index.hnsw.efSearch = 128
           index.add(byte_data)
           print(f"{i}-th index created.")

           table2 = pq.read_table(f"./data/voters10_minhash_B.pqt")
           df2 = table2.to_pandas()
           tp = 0
           fp = 0
           k = 2
           no_batch = 50_000
           N = len(df2)
           batches = N  // no_batch
           st1 = 0
           for j in range(batches):
             if j == batches - 1:
                no_batch = no_batch + (N % no_batch)
             #print(f"{i} querying from {st1} to {st1+no_batch-1}")

             vectors = df2.loc[st1:st1+no_batch-1, 'v'].tolist()
             ids = df2.loc[st1:st1+no_batch-1, 'id'].tolist()

             data = np.array(vectors).astype(np.uint8)
             byte_data = np.packbits(data, axis=-1)
             ii = 0
             distances, replies = index.search(byte_data, k)  #index.search(query, k)
             distances = distances/128
             for rs, dss in zip(replies, distances):
                 id2 = ids[ii]
                 ii+=1
                 for ind, ds in zip(rs, dss):
                     ind = i*len(df1) + ind
                     df01 = df1.loc[ind, ]
                     id1 = df01["id"]
                     #if ds < 0.45:
                     if id2 == id1:
                             tp+=1
                     else:
                             fp+=1
             st1 = st1+no_batch   

           queue.put((tp, fp))



    #start  = time.time()
    #with Pool(num_cores) as pool:
     #   results = pool.map(build, chunks)
 
    
    start  = time.time()
    
    queue = multiprocessing.Queue() 
    processes = []
    for i in range(num_cores):
        p = multiprocessing.Process(target=build, args=(i,queue,))
        p.start()
        processes.append(p)


    for p in processes:
              p.join()

    end = time.time()
    print(f"Time for building the indexes {end-start} seconds.")

 
    results = [queue.get() for _ in range(num_cores)]

    for p in processes:
         p.join()
   
              
    end  = time.time()          
    tp=0
    fp=0
    for tp_, fp_ in results:
        tp+=tp_
        fp+=fp_
    matches = 3_000_000
    print(f"tp={tp} fp={fp}  recall={round(tp/matches, 2)} precision={round(tp/(tp+fp), 2)} time for querying={end-start} seconds.")


