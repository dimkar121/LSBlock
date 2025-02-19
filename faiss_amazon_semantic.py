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

    d = 120
    df1 = pd.read_parquet(f"./data/Amazon_embedded_minhash.pqt")
    df2 = pd.read_parquet(f"./data/Google_embedded_minhash.pqt")
 
    theta = thr.opt_jaccard_threshold(df1, df2, truth)

    amazons = []
    for i1, r1 in df1.iterrows():
             id = r1["id"]
             name = r1["name"]
             description = r1["description"]
             amazons.append([id, name, description])
    
    index1 = faiss.IndexBinaryHNSW(d, 32)
    index1.metric_type = faiss.METRIC_Jaccard
    index1.hnsw.efConstruction = 60
    index1.hnsw.efSearch = 16
    vectors = df1['v'].tolist()
    data = np.array(vectors).astype(np.uint8)
    byte_data = np.packbits(data, axis=-1)
    index1.add(byte_data)  # Reshape to 2D array

    model="mini"
    df11 = pd.read_parquet(f"./data/Amazon_embedded_{model}.pqt")
    df22 = pd.read_parquet(f"./data/Google_embedded_{model}.pqt")    
    vectors_amazon =  df11['v'].tolist()
    vectors_google = df22['v'].tolist()

    phi = thr.opt_inner_threshold(df11, df22, truth)


    tp = 0
    fp = 0    
    qtimes = []
    for i2, r2 in df2.iterrows():
           start_t = time.time()
           name = r2["name"]
           description = r2["description"]
           id2 = r2["id"]
           de2 = np.array([r2["v"]]).astype(np.uint8)
           byte_query = np.packbits(de2, axis=-1)
           distances, indices = index1.search(byte_query, 10)
           distances = distances/120
           v_google = vectors_google[i2] 
           for ind, ds in zip(indices[0], distances[0]):
              if ds < theta:
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

