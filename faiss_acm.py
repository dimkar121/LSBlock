import pandas as pd
import numpy as np
import faiss
import thresholds as thr
import time

if __name__ == '__main__':
    truth = pd.read_csv("./data/truth_ACM_DBLP.csv", sep=",", encoding="utf-8", keep_default_na=False)
    truthD = dict()
    a = 0

    for i, r in truth.iterrows():
        idDBLP = r["idDBLP"]
        idACM = r["idACM"]
        if idACM in truthD:
            ids = truthD[idACM]
            ids.append(idDBLP)
            a += 1
        else:
            truthD[idACM] = [idDBLP]
            a+=1 
    matches =  a
    print("No of matches=", matches)

    #====================================================================--
    model = "minhash"
    d = 120
    df1 = pd.read_parquet(f"./data/ACM_embedded_{model}.pqt")
    df2 = pd.read_parquet(f"./data/DBLP_embedded_{model}.pqt")

    theta = thr.opt_jaccard_threshold(df2, df1, truth)

    n = 0
    dblps = []
    index = faiss.IndexBinaryHNSW(d, 32)
    index.metric_type = faiss.METRIC_Jaccard
    #print("Default Metric:", index.metric_type)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16

    for i1, r1 in df2.iterrows():
        id = r1["id"]
        authors1 = r1["authors"]
        title1 = r1["title"]        
        venue1 = r1["venue"]
        de1 = r1["v"]
        dblps.append([id, authors1, title1, venue1])

    vectors = df2['v'].tolist()
    data = np.array(vectors).astype(np.uint8)    
    byte_data = np.packbits(data, axis=1)    
    index.add(byte_data)  # Reshape to 2D array
    #=========================================================================

    model = "mini"
    d = 384
    df11 = pd.read_parquet(f"./data/ACM_embedded_{model}.pqt")
    df22 = pd.read_parquet(f"./data/DBLP_embedded_{model}.pqt")

    phi = thr.opt_inner_threshold(df22, df11, truth)
    vectors_dblp = df22['v'].tolist()
    vectors_acm = df11['v'].tolist()
   


    tp = 0
    fp = 0    
    qtimes = []
    for i2, r2 in df1.iterrows():
          start_t = time.time()
          authors2 = r2["authors"]
          title2 = r2["title"]
          venue2 = r2["venue"]
          id = r2["id"]
          de2 = np.array([r2["v"]]).astype(np.uint8)
          byte_data = np.packbits(de2, axis=1)
          distances, indices1 = index.search(byte_data, 3)    # return scholars           
          distances = distances/120
          i=0 
          v_acm = vectors_acm[i2]                         
          for ind, ds in zip(indices1[0], distances[0]): 
                  if ds < theta:
                    v_dblp = vectors_dblp[ind]
                    sim = np.inner(v_acm, v_dblp)           
                    if sim > phi: 
                       idACM = id   #ids2[ind][0]
                       if idACM in truthD.keys():
                          idDBLPs = truthD[idACM]
                          for idDBLP in idDBLPs:
                             if idDBLP == dblps[ind][0]:   
                                   tp += 1
                                   #print("not found by the dense vectors=======================================================================================")
                                   #print(f"{authors2} {title2} {venue2}  ")
                                   #print(f"{dblps[ind][1]} {dblps[ind][2]} {dblps[ind][3]} ")
                                    
                             else:
                                   fp += 1

                    i+=1       
          end_t= time.time()         
          qtimes.append(end_t - start_t)

      
    mean_q = np.mean(qtimes)
    print(f"recall={round(tp/matches, 2)} precision={round(tp/(tp+fp), 2)} query time={round(mean_q,4)}")


