import argparse
import os
import pickle

import faiss
import numpy as np

from utils import load_msmarco_from_star_vectors
from utils import load_msmarco_from_tasb
from utils import load_msmarco_queries_from_star_vectors
from utils import build_ivfindex_from_embeddings
from utils import build_index_from_embeddings
from utils import load_msmarco_from_contriver


def load_msmarco(DATA_DIR, encoder='star'):
    if encoder == 'star':
        embs, ids = load_msmarco_from_star_vectors(DATA_DIR)
    if encoder == 'contriver':
        embs, ids = load_msmarco_from_contriver(DATA_DIR)
    if encoder == 'tasb':
        embs, ids = load_msmarco_from_tasb(DATA_DIR)
    return embs, ids


def get_offset_id_mappings(encoder, splits=['train', 'dev']):
    qid2offset = {}
    offset2qid = {}
    pid2offset = {}
    offset2pid = {}
    if encoder == 'star':
        for split in splits:
            qid2offset[split] = pickle.load(open(os.path.join(DATA_DIR, f"{split}-qid2offset.pickle"), 'rb'))
            offset2qid[split] = {v: k for k, v in qid2offset[split].items()}
        # get docid offsets
        pid2offset = pickle.load(open(os.path.join(DATA_DIR, "pid2offset.pickle"), 'rb'))
        offset2pid = {v: k for k, v in pid2offset.items()}
    if (encoder == 'contriver') or (encoder == 'tasb'):
        pid2offset = dict(enumerate(np.arange(doc_embeddings.shape[0])))
        offset2pid = {v: k for k, v in pid2offset.items()}
        pass
    return qid2offset, offset2qid, pid2offset, offset2pid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='/home/busolin/star_vectors/')
    parser.add_argument("--index_dir", type=str, default='/home/busolin/indexes/')
    parser.add_argument("--index_key", type=str, default='IVF65535,Flat')
    parser.add_argument("--encoder", type=str, default='star')
    parser.add_argument("--use_adaptive", type=bool, default=False)
    parser.add_argument("--metric", type=str, default='cosine')
    args = parser.parse_args()

    use_adaptive = args.use_adaptive
    index_key = args.index_key
    dataset_dir = args.dataset_dir
    index_dir = args.index_dir
    metric = args.metric
    encoder = args.encoder

    build_metric = 'L2'
    if metric == 'cosine' or metric == 'dot':
        build_metric = 'IP'
    elif metric == 'euclidean':
        build_metric = 'L2'
    else:
        print(f"[ERROR]: metric {metric} not supported!")
        exit(1)

    DATA_DIR = dataset_dir

    doc_embeddings, doc_ids = load_msmarco(DATA_DIR, encoder=encoder)
    print(f'[INFO] Loaded {doc_embeddings.shape[0]} embeddings from {DATA_DIR}.')

    #qid2offset, offset2qid, pid2offset, offset2pid = get_offset_id_mappings(encoder)

    if metric == 'cosine':
        norm_doc_embeddings = doc_embeddings.copy()
        faiss.normalize_L2(norm_doc_embeddings)
    else:
        norm_doc_embeddings = doc_embeddings

    if use_adaptive and index_key.startswith('IVF'):
        n = norm_doc_embeddings.shape[0]
        nlist = 4 * np.sqrt(n).astype(int)
        print(f"[WARN] Using adaptive: ignoring {index_key} and using IVF{nlist},Flat instead")
        index_key = f"IVF{nlist},Flat"
    else:
        nlist = int(index_key.split(',')[0].split('IVF')[1]) if index_key.startswith('IVF') else 0

    assert (nlist > 0 and index_key.startswith('IVF')) or (nlist == 0 and not index_key.startswith('IVF'))

    print("[INFO] Index build")
    index_tag = '.'.join(index_key.split(','))
    save_path = f"{index_dir}/{'msmarco'}.{encoder}.{index_tag}.{metric}.faiss"
    print(f'[INFO] save_path: {save_path}')
    if index_key.startswith('IVF'):
        index = build_ivfindex_from_embeddings(norm_doc_embeddings, nlist, n_samples=50, ids=doc_ids,
                                               metric=build_metric,
                                               index_type=index_key, verbose=True)
    else:
        index = build_index_from_embeddings(norm_doc_embeddings, doc_ids, build_metric, index_key, verbose=True)
    faiss.write_index(index, save_path)
