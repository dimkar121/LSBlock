import pandas as pd
import numpy as np
import faiss
import thresholds as thr
import time

if __name__ == '__main__':

    truth = pd.read_csv("./data/truth_Amazon_googleProducts.csv", sep=",", encoding="unicode_escape", keep_default_na=False)
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        idAmazon = r["idAmazon"]
        idGoogle = r["idGoogleBase"]
        if idAmazon in truthD.keys():
            ids = truthD[idAmazon]
            ids.append(idGoogle)
            a +=1
        else:
          truthD[idAmazon] = [idGoogle]
          a+=1
    matches = a
    print("No of matches=", matches)

    d = 384
    df1 = pd.read_parquet(f"./data/Amazon_embedded_mini.pqt")
    df2 = pd.read_parquet(f"./data/Google_embedded_mini.pqt")
 
    amazons = []
    for i1, r1 in df1.iterrows():
             id = r1["id"]
             name = r1["name"]
             description = r1["description"]
             amazons.append([id, name, description])
    
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16
    vectors = df1['v'].tolist()
    data = np.array(vectors).astype(np.float32)
    index.add(data)

    vectors_amazon = df1['v'].tolist()
    vectors_google = df2['v'].tolist()

    phi = thr.opt_inner_threshold(df1, df2, truth)


    tp = 0
    fp = 0    
    qtimes = []
    for i2, r2 in df2.iterrows():
           start_t = time.time()
           name = r2["name"]
           description = r2["description"]
           id2 = r2["id"]
           de2 = np.array([r2["v"]]).astype(np.float32)
           distances, indices = index.search(de2, 5)

           v_google = vectors_google[i2] 
           for ind, ds in zip(indices[0], distances[0]):
                 v_amazon = vectors_amazon[ind]
                 idAmazon = amazons[ind][0]
                 sim = np.inner(v_amazon, v_google)
                 if sim > phi:
                    if idAmazon in truthD.keys():
                        idGoogles = truthD[idAmazon]
                        for idGoogle in idGoogles:
                          if idGoogle == id2:
                                #print("===============================================================================================================")
                                #print(f"{name} ")
                                #print(f"{description} ")
                                #print("-----------------------------------")
                                #print(f" {amazons[ind][1]}")
                                #print(f"{amazons[ind][2]} ")
                                tp += 1
                          else:
                                fp += 1
                 
           end_t= time.time()
           qtimes.append(end_t - start_t)
    mean_q = np.mean(qtimes)         
    print("recall=", round(tp / matches, 2), "precision=", round(tp / (tp + fp), 2), "query time=", round(mean_q,3 ) )

