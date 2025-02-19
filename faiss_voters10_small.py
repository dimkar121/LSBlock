import pandas as pd
import numpy as np
import faiss
import thresholds as thr
import time

if __name__ == '__main__':
    truth = pd.read_csv("./data/truth_voters.csv", sep=",", encoding="utf-8", keep_default_na=False)
    truthD = dict()
    a = 0

    for i, r in truth.iterrows():
        id1 = r["id1"]
        id2 = r["id2"]
        if id1 in truthD:
            ids = truthD[id1]
            if id2 in ids:
                print(f" {id2} already exists for {id1}")
            ids.append(id2)
            a += 1
        else:
            if id1 != id2:
                 print(f" different ids {id1} and {id2}")
            truthD[id1] = [id2]
            a+=1 
    matches =  a
    print("No of matches=", matches)
    #====================================================================--
    model = "minhash"
    d = 120
    df1 = pd.read_parquet(f"./data/votersA_embedded_{model}.pqt")
    df2 = pd.read_parquet(f"./data/votersB_embedded_{model}.pqt")

    theta = thr.opt_jaccard_threshold(df1, df2, truth)

    n = 0
    votersA = []
    index = faiss.IndexBinaryHNSW(d, 32)
    index.metric_type = faiss.METRIC_Jaccard
    #print("Default Metric:", index.metric_type)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16

    for i1, r1 in df1.iterrows():
        id = r1["id"]
        givenname = r1["givenname"]
        surname = r1["surname"]        
        postcode = r1["postcode"]
        suburb = r1["suburb"]
        de1 = r1["v"]
        votersA.append([id, givenname, surname, postcode, suburb])

    vectors = df1['v'].tolist()
    data = np.array(vectors).astype(np.uint8)    
    byte_data = np.packbits(data, axis=1)    
    index.add(byte_data)
    #=========================================================================

    model = "mini"
    d = 384
    df11 = pd.read_parquet(f"./data/votersA_embedded_{model}.pqt")
    df22 = pd.read_parquet(f"./data/votersB_embedded_{model}.pqt")

    phi = thr.opt_inner_threshold(df11, df22, truth)
    vectors_votersA = df11['v'].tolist()    
    vectors_votersB = df22['v'].tolist()
   


    tp = 0
    fp = 0    
    qtimes = []
    tps = set()
    for i2, r2 in df2.iterrows():
          start_t = time.time()
          givenname2 = r2["givenname"]
          surname2 = r2["surname"]
          postcode2 = r2["postcode"]
          suburb2 = r2["suburb"]
          id = r2["id"]
          de2 = np.array([r2["v"]]).astype(np.uint8)
          byte_data = np.packbits(de2, axis=1)
          distances, indices1 = index.search(byte_data, 5)    # return scholars
          distances = distances/120
          i=0 
          v_votersB = vectors_votersB[i2]           
          
          for ind, ds in zip(indices1[0], distances[0]): 
                  if ds < theta:
                    v_votersA = vectors_votersA[ind]
                    sim = np.inner(v_votersA, v_votersB)           
                    if sim > phi: 
                      idVotersA = votersA[ind][0]
                      if idVotersA in truthD.keys():
                          idVotersB = truthD[idVotersA][0]                          
                          if idVotersB == id:   
                              if not idVotersB in tps:   
                                   tp += 1                                   
                                   tps.add(idVotersB)
                                   #print("TP=======================================================================================")
                                   #print(f"{givenname2} {surname2} {postcode2} {suburb2} ")
                                   #print(f"{votersA[ind][1]} {votersA[ind][2]} {votersA[ind][3]}  {votersA[ind][4]} ")                                    
                          else:
                                   fp += 1
                    #else:
                        #if ds < 0.15 and i==0:
                         # print(f"{ds}=======================================================")
                         # print(f"searching {authors2} {title2} {venue2}  ")
                         # print(f" {dblps[ind][1]} {dblps[ind][2]} {dblps[ind][3]} ")
                    i+=1       
          end_t= time.time()         
          qtimes.append(end_t - start_t)

      
    mean_q = np.mean(qtimes)
    print(f"{tp} {matches}  recall={round(tp/matches, 2)} precision={round(tp/(tp+fp), 2)} query time={round(mean_q,4)}")


