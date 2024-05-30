from sentence_transformers import SentenceTransformer, util
import numpy as np
import ir_datasets as ird
import torch
import tqdm

from utils import batched_iterator

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = '/home/busolin/dense_data/msmarco/tasb/'
    model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b').to(device)
    dataset = ird.load("msmarco-passage")
    n_docs = dataset.docs_count()
    batch_size = 2048
    i = 0
    memmap = np.memmap(data_dir + 'embeddings.dat', dtype='float32', mode='w+', shape=(n_docs, 768))
    for batch in tqdm.tqdm(batched_iterator(dataset.docs_iter(), batch_size=batch_size), total=n_docs // batch_size):
        batch_text = [doc.text for doc in batch]
        embeddings = model.encode(batch_text, convert_to_tensor=True, device=device)
        nv = embeddings.shape[0]
        memmap[i:i+nv] = embeddings.cpu().numpy()
        i += nv

    data = np.memmap(data_dir + f'embeddings.dat', dtype='float32', mode='r').reshape(-1, 768)
    print(data.shape, i, nv)