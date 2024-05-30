# compute Nearest Neighbour ground truth element for a dataset

import faiss
import numpy as np
import os
import pickle
import argparse
import ir_datasets

from utils import load_msmarco_from_star_vectors, load_msmarco_queries_from_star_vectors


def load_query_embeddings(DATA_DIR, encoder):
    dataset_dev = ir_datasets.load('msmarco-passage/dev')
    dataset_small = ir_datasets.load('msmarco-passage/dev/small')
    if encoder == 'star':
        query_embeddings, query_ids = load_msmarco_queries_from_star_vectors(DATA_DIR, queries='dev-full')
        dev_query_embeddings, dev_query_ids = load_msmarco_queries_from_star_vectors(DATA_DIR, queries='dev-small')
        return query_embeddings, query_ids, dev_query_embeddings, dev_query_ids
    if encoder == 'contriver' or encoder == 'tasb':
        query_ids = np.array([int(query.query_id) for query in dataset_dev.queries_iter()])
        query_small_ids = np.array([int(query.query_id) for query in dataset_small.queries_iter()])
        query_embeddings = np.memmap(DATA_DIR + f'queries_dev.dat', dtype='float32', mode='r').reshape(-1, 768)
        dev_embeddings = np.memmap(DATA_DIR + f'queries_dev-small.dat', dtype='float32', mode='r').reshape(-1, 768)
        return query_embeddings, query_ids, dev_embeddings, query_small_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='/home/busolin/star_vectors/')
    parser.add_argument("--index_dir", type=str, default='/home/busolin/indexes/')
    parser.add_argument("--similarity", type=str, default='euclidean')
    parser.add_argument("--encoder", type=str, default='star')
    # parser.add_argument("--num_clusters", type=int, default=65535)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    index_dir = args.index_dir
    similarity = args.similarity
    encoder = args.encoder
    # num_clusters = args.num_clusters

    index_name = f'msmarco.{encoder}.Flat.{similarity}.fidx'
    index = faiss.read_index(os.path.join(index_dir, index_name))
    print(f'[INFO] Loaded index {index_name} with {index.ntotal} elements.')
    # load dataset

    DATA_DIR = dataset_dir

    train_embeddings, train_ids, test_embeddings, test_ids = load_query_embeddings(DATA_DIR, encoder)

    print(f'[INFO] Loaded {train_embeddings.shape[0]} train queries and {test_embeddings.shape[0]} dev queries.')

    # search closest neighbours
    # faiss.ParameterSpace().set_index_parameter(index, "nprobe",num_clusters)  # bit of a hack, slow but finds nearest neighbour
    print(f'[INFO] Searching for ground truth of train queries using {index_name}')
    res_faiss_train = index.search(train_embeddings, 100)
    print(f'[INFO] Searching for ground truth of dev queries using {index_name}')
    res_faiss_dev = index.search(test_embeddings, 100)

    print(f'[INFO] Saving results')

    # save results in dataset_dir
    np.save(os.path.join('/home/busolin/datasets/msmarco/', f'{encoder}.dev-small_gt_{similarity}.npy'),
            res_faiss_dev)
    np.save(os.path.join('/home/busolin/datasets/msmarco/', f'{encoder}.dev-full_gt_{similarity}.npy'),
            res_faiss_train)
