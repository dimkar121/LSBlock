import pandas as pd
import numpy as np
import faiss
import time
import thresholds as thr



if __name__ == '__main__':
    truth = pd.read_csv("./data/truth_fodors_zagats.csv", sep=",", encoding="utf-8", keep_default_na=False)
    truthD = dict()
    a = 0
    for i, r in truth.iterrows():
        idFodors = r["idFodors"]
        idZagats = r["idZagats"]
        truthD[idFodors] = idZagats
    matches = len(truthD.keys())
    print("No of matches=", matches)
    #====================================================================--
    model = "minhash"
    d = 120
    df1 = pd.read_parquet(f"./data/fodors_embedded_{model}.pqt")
    df2 = pd.read_parquet(f"./data/zagats_embedded_{model}.pqt")

    theta = thr.opt_jaccard_threshold(df1, df2, truth)

    n = 0
    num_rows = df1.shape[0]
    data = np.zeros((num_rows, d), dtype='float32')
    fodors = []
    index = faiss.IndexBinaryHNSW(d, 32)
    index.metric_type = faiss.METRIC_Jaccard
    index.hnsw.efConstruction = 60
    index.hnsw.efSearch = 16

    for i1, r1 in df1.iterrows():
        id = r1["id"]
        name = r1["name"]
        address = r1["address"]        
        de1 = r1["v"]
        city = r1["city"]
        type1 = r1["type"]
        fodors.append([id, name, address, city, type1])

    vectors = df1['v'].tolist()
    data = np.array(vectors).astype(np.uint8)
    byte_data = np.packbits(data, axis=-1)
    index.add(byte_data)
    #=========================================================================

    model = "mini"
    d = 384
    df11 = pd.read_parquet(f"./data/fodors_embedded_{model}.pqt")
    df22 = pd.read_parquet(f"./data/zagats_embedded_{model}.pqt")
    
    phi = thr.opt_inner_threshold(df11, df22, truth)

    vectors_fodors = df11['v'].tolist()
    vectors_zagats = df22['v'].tolist()

    tp = 0
    fp = 0
    tps=set()
    qtimes = []
    for i2, r2 in df2.iterrows():
           start_t = time.time()
           name2 = r2["name"]
           address2 = r2["address"]
           id2 = r2["id"]
           city2 = r2["city"]
           type2 = r2["type"]
           de2 = np.array([r2["v"]]).astype(np.uint8)
           byte_data = np.packbits(de2, axis=1)
           distances, indices1 = index.search(byte_data, 5)
           distances = distances/120
           v_zagats = vectors_zagats[i2]
           for ind, ds in zip(indices1[0], distances[0]): 
                 if ds < theta:
                    v_fodors = vectors_fodors[ind]
                    sim = np.inner(v_zagats, v_fodors)
                    if sim > phi:
                       idFodors = fodors[ind][0]
                       if idFodors in truthD.keys():
                            idZagats = truthD[idFodors]
                            if idZagats == id2:
                                tp += 1
                                #print(f"TP=======================================================")
                                #print(f"searching{id2}    {name2} {address2} {city2} {type2}  ")
                                #print(f"  {fodors[ind][1]} {fodors[ind][2]} {fodors[ind][3]} {fodors[ind][4]}  ")
                            else:
                                fp += 1
                                #print(f"FP=======================================================")
                                #print(f"{name2} {address2} {city2} {type2}  ")
                                #print(f"{fodors[ind][1]} {fodors[ind][2]} {fodors[ind][3]} {fodors[ind][4]}  ")

                    #else:
                         #print(f"Missed Inner=======================================================")
                         #print(f"{name2} {address2} {city2} {type2}  ")
                         #print(f"{fodors[ind][1]} {fodors[ind][2]} {fodors[ind][3]} {fodors[ind][4]}  ")

  

           end_t= time.time()
           qtimes.append(end_t - start_t)


    mean_q = np.mean(qtimes)
    print(f"recall={round(tp / matches, 2)} precision={round(tp / (tp + fp), 2)} query time={round(mean_q,4)}  ")

