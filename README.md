
# LSBlock
LSBlock is a novel hybrid blocking system that combines lexical and semantic similarity to improve record linkage accuracy. By leveraging minhash-based blocking keys and dense embeddings, LSBlock effectively balances precision and recall. 
This repo contains the implementation of LSBlock and the benchmark datasets used to evaluate its performance. 
The title of the corresponding manuscript is:

"LSBlock: A Hybrid Blocking System Combining Lexical and Semantic Similarity Search for Record Linkage", 

co-authored by D. Karapiperis (IHU), C. Tjortjis (IHU), and V. Verykios (HOU).


## Abstract
Record linkage plays a crucial role in data integration by identifying and merging records that correspond to the same entity across different datasets. Traditional linkage methods are either
lexically driven, relying on surface-level textual matching, or semantically driven, capturing contextual meaning. In this paper, we propose LSBlock, a hybrid blocking system that integrates lexical
and semantic similarity search to enhance record linkage accuracy by improving recall while maintaining high levels of precision. LSBlock utilizes minhash-based lexical similarity for efficient block-
ing key formulation and applies dense embeddings with cosine similarity to refine candidate matches. A Hierarchical Navigable Small-World (HNSW) index is leveraged for efficient approximate
nearest neighbor searches. The retrieved results are further refined using two thresholds: one in the Jaccard space and another in the cosine similarity space, both derived after training a model using
labeled matches. Our experimental evaluation on six real-world datasets demonstrates that LSBlock significantly outperforms state-of-the-art blocking techniques in terms of recall, precision, and
F1-score, achieving a balanced trade-off between efficiency and effectiveness. LSBlock consistently achieves recall above 0.95 and precision above 0.65, when using structured data, outperforming
its competitors by a large margin. These findings highlight the advantages of integrating lexical and semantic similarity, making LSBlock a robust solution for scalable record linkage.


## Running the artifact
Clone the repo and install project's dependencies:
```
pip3 install -r requirements.txt
```
The source has been tested with Python versions 3.10 and 3.12

 
The following scripts evaluate the performance of LSBlock using the following benchmark datasets:
| paired dataset | linkage | SHAP (for deriving the blocking keys) | Recall-Precision (averages) |
| --- | --- | --- | :---: |
| ACM and DBLP | `faiss_acm.py` | `acm_blocking_keys.py` | 0.98 - 0.69 |
| Google Scholar and DBLP | `faiss_Scholar_DBLP.py` | `scholar_blocking_keys.py` | 0.96 - 0.66 |
| Abt and Buy | `faiss_abt.py` | `abt_blocking_keys.py` | 0.95 - 0.12 |
| Fodor's and Zagat's | `faiss_restaurants.py` | `restaurants_blocking_keys.py` | 0.99 - 0.68 |
| Amazon and Google products | `faiss_amazon.py` | `amazon_blocking_keys.py` | 0.89 - 0.08 | 
| two subsets of the Voters dataset | `faiss_voters10_small.py` | `voters_blocking_keys.py` | 0.97 - 0.69 |
| Voters10 (10M records) | `faiss_voters10.py` | | 0.95 - 0.60 |

In order to run `faiss_voters10.py` download the required parquet files from [here](https://1drv.ms/f/c/5d21c94cfd3ca26c/EruRIWVgq3BAiPsXt5RHnpABL5g0asPNHfAbcTSGAjPfsg?e=EPsIjT).
  
   
