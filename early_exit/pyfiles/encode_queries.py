from sentence_transformers import SentenceTransformer, util
import numpy as np
import ir_datasets as ird
import torch
import tqdm

from utils import batched_iterator
split2tag = {
    'train': 'train',
    'dev-small': 'dev/small',
    'dev': 'dev',
}
models = {
    'contriver': 'facebook/contriever-msmarco',
    'tasb': 'sentence-transformers/msmarco-distilbert-base-tas-b',
}

if __name__ == '__main__':
    encoder = 'tasb'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = f'/home/busolin/dense_data/msmarco/{encoder}/'
    split = 'dev'
    model = SentenceTransformer(models[encoder]).to(device)
    split_name = f"msmarco-passage/{split2tag[split]}"

    dataset = ird.load(split_name)
    n_q = dataset.queries_count()
    batch_size = 1024
    i = 0
    memmap = np.memmap(data_dir + f'queries_{split}.dat', dtype='float32', mode='w+', shape=(n_q, 768))
    for batch in tqdm.tqdm(batched_iterator(dataset.queries_iter(), batch_size=batch_size), total=n_q // batch_size):
        batch_text = [doc.text for doc in batch]
        embeddings = model.encode(batch_text, convert_to_tensor=True, device=device)
        nv = embeddings.shape[0]
        memmap[i:i+nv] = embeddings.cpu().numpy()
        i += nv

    data = np.memmap(data_dir + f'queries_{split}.dat', dtype='float32', mode='r').reshape(-1, 768)
    print(data.shape, i, nv)