import pandas as pd
import numpy as np
import faiss
import thresholds as thr
import time

if __name__ == '__main__':

    truth = pd.read_csv("./data/truth_abt_buy.csv", sep=",", encoding="unicode_escape", keep_default_na=False)
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        idAbt = r["idAbt"]
        idBuy = r["idBuy"]
        if idAbt in truthD:
            ids = truthD[idAbt]
            ids.append(idBuy)
            a += 1
        else:
            truthD[idAbt] = [idBuy]
    matches = len(truthD.keys()) + a
    print("No of matches=", matches)

    df1 = pd.read_parquet(f"./data/Abt_embedded_minhash.pqt")
    df2 = pd.read_parquet(f"./data/Buy_embedded_minhash.pqt")

    abts=[]
    for i1, r1 in df1.iterrows():
          id = r1["id"]
          name= r1["name"]
          description = r1["description"]
          abts.append([id, name, description])

 
    d = 120
    index1 = faiss.IndexBinaryHNSW(d, 32)
    index1.metric_type = faiss.METRIC_Jaccard
    index1.hnsw.efConstruction = 60
    index1.hnsw.efSearch = 16
    vectors = df1['v'].tolist()
    data = np.array(vectors).astype(np.uint8)
    byte_data = np.packbits(data, axis=-1)
    index1.add(byte_data)  

    theta = thr.opt_jaccard_threshold(df1, df2, truth)

    model="mini"
    d=384
    df11 = pd.read_parquet(f"./data/Abt_embedded_{model}.pqt")
    df22 = pd.read_parquet(f"./data/Buy_embedded_{model}.pqt")    

    vectors_abt = df11['v'].tolist()
    vectors_buy = df22['v'].tolist()

    phi = thr.opt_inner_threshold(df11, df22, truth)


    tp = 0
    fp = 0
    qtimes=[]
    for i2, r2 in df2.iterrows():
           start_t = time.time()
           name = r2["name"]
           description = r2["description"]
           id2 = r2["id"]
           de2 = np.array([r2["v"]]).astype(np.uint8)
           byte_query = np.packbits(de2, axis=-1)
           distances, indices = index1.search(byte_query, 10)
           distances = distances/120
           v_buy = vectors_buy[i2]          
           for ind, ds in zip(indices[0], distances[0]):
             if ds < theta:  
               v_abt = vectors_abt[ind]
               sim = np.inner(v_buy, v_abt)
               if sim > phi:
                 idAbt = abts[ind][0]
                 if idAbt in truthD.keys():
                        idBuys = truthD[idAbt]
                        for idBuy in idBuys:
                            if idBuy == id2:
                                # print(f"{name}")
                                # print(f"{description} ")
                                # print(f"{abts[ind][1]}")
                                # print(f"{abts[ind][2]} ") 
                               tp += 1
                            else:
                               fp += 1
           end_t= time.time()
           qtimes.append(end_t - start_t)      
    mean_q = np.mean(qtimes)       
    print("recall=", round(tp / matches, 2), "precision=", round(tp / (tp + fp), 2),"query time=", round(mean_q,3) )

    

