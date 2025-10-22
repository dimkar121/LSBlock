
# LSBlock
LSBlock is a novel hybrid blocking system that combines lexical and semantic similarity to improve record linkage accuracy. By leveraging minhash-based blocking keys and dense embeddings, LSBlock effectively balances precision and recall. 
This repo contains the implementation of LSBlock and the benchmark datasets used to evaluate its performance. 
The title of the corresponding manuscript is:

"LSBlock: A Hybrid Blocking System Combining Lexical and Semantic Similarity Search for Record Linkage"  and "Towards a Hybrid Embedding Model for Robust Entity Resolution" co-authored by D. Karapiperis (IHU), C. Tjortjis (IHU), and V. Verykios (HOU).

## Hybrid Vectors
We are currently working on a model that creates a single, robust vector for search (e.g., for e-commerce) that is resilient to both typos and semantic ambiguity. It achieves this by fusing two different types of embeddings using a trainable gated network. This allows the model to learn when to pay attention to spelling (to correct typos) and when to pay attention to meaning (to find related items), all within a single vector representation. We utilize (a) a transformer model (like all-MiniLM-L6-v2) to get a context-aware vector that understands meaning, and (b) FastText (cc.en.300.bin) to get a sub-word vector that understands spelling and is robust to typos. 


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

`faiss_multi.py` first splits voters10_minhash_A.pqt, whose size is 10M minhash vectors, into multiple chunks based on the number of available CPU cores, ensuring that each chunk is assigned to a separate worker process using `multiprocessing.Process()`. This allows efficient parallel processing by leveraging all available computational resources for FAISS index construction and querying without shared memory conflicts. The FAISS indexes are stored in the local memory of each process to avoid serialization overhead. Once the indexes are built, queries are dispatched in parallel to all processes, where each worker searches its local FAISS index. The individual search results are collected via an inter-process queue (multiprocessing.Queue), and the main process merges them to return the final ranked results. This approach maximizes parallelism while minimizing data transfer bottlenecks, making it ideal for large-scale similarity search applications.

In order to run `faiss_voters10.py` download the required parquet files from [here](https://1drv.ms/f/c/5d21c94cfd3ca26c/EruRIWVgq3BAiPsXt5RHnpABL5g0asPNHfAbcTSGAjPfsg?e=EPsIjT).
  
   
