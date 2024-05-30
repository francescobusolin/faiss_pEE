import faiss
import numpy as np


def ivecs_read(fname):
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def bvecs_read(fname):
    a = np.fromfile(fname, dtype="uint8")
    d = a.view('int32')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy()


def batched_iterator(iterator, batch_size):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def name2file(name):
    if name == "sift":
        return "sift", "fvecs"
    elif name == "gist":
        return "gist", "fvecs"
    elif name == "sift1B":
        return "bigann", "bvecs"
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def readvecs(fname):
    _, fext = fname.split(".")
    if fext == "fvecs":
        return fvecs_read(fname)
    elif fext == "bvecs":
        return bvecs_read(fname)
    elif fext == "ivecs":
        return ivecs_read(fname)
    else:
        raise ValueError(f"Unknown file extension: {fext}")


# xt = fvecs_read("sift1M/sift_learn.fvecs")
# xb = fvecs_read("sift1M/sift_base.fvecs")
# xq = fvecs_read("sift1M/sift_query.fvecs")
#
# d = xt.shape[1]
#
# print("load GT")
#
# gt = ivecs_read("sift1M/sift_groundtruth.ivecs")
# gt = gt.astype('int64')
# k = gt.shape[1]
# index = faiss.index_factory(d, index_key)
# index.train(xt)
# index.add(xb)
#

def load_msmarco_from_star_vectors(data):
    # loading MS-MARCO passage v1.0 collection
    doc_memmap_path = data + "passages.memmap"
    docid_memmap_path = data + "passages-id.memmap"

    doc_embeddings = np.memmap(doc_memmap_path, dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, 768)
    doc_ids = np.memmap(docid_memmap_path, dtype=np.int32, mode="r")

    # moving memmaps to real np.arrays
    # doc_embeddings = np.copy(doc_embeddings)
    # doc_ids = np.copy(doc_ids)

    return doc_embeddings, doc_ids

def load_msmarco_from_tasb(data):
    doc_memmap_path = data + "embeddings.dat"
    doc_embeddings = np.memmap(doc_memmap_path, dtype=np.float32, mode="r").reshape(-1, 768)
    doc_ids_collection = np.arange(doc_embeddings.shape[0])

    return doc_embeddings, doc_ids_collection

def load_msmarco_from_contriver(data):
    doc_memmap_path = data + "contriever.dat"
    doc_embeddings = np.memmap(doc_memmap_path, dtype=np.float32, mode="r").reshape(-1, 768)
    doc_ids_collection = np.arange(doc_embeddings.shape[0])

    return doc_embeddings, doc_ids_collection


def load_msmarco_queries_from_star_vectors(data_dir, queries='dev-small'):
    if queries == 'dev-full':
        query_memmap_path = data_dir + "dev-full-query.memmap"
        queryids_memmap_path = data_dir + "dev-full-query-id.memmap"
    elif queries == 'dev-small':
        query_memmap_path = data_dir + "dev-query.memmap"
        queryids_memmap_path = data_dir + "dev-query-id.memmap"
    elif queries == 'train':
        query_memmap_path = data_dir + "train-query.memmap"
        queryids_memmap_path = data_dir + "train-query-id.memmap"
    else:
        return None, None

    query_embeddings = np.memmap(query_memmap_path, dtype=np.float32, mode="r")
    query_embeddings = query_embeddings.reshape(-1, 768)
    query_ids = np.memmap(queryids_memmap_path, dtype=np.int32, mode="r")

    # moving memmaps to real np.arrays
    # query_embeddings = np.copy(query_embeddings)
    # query_ids = np.copy(query_ids)

    return query_embeddings, query_ids


def build_index_from_embeddings(embeddings, ids=None, metric='L2', index_type="Flat", verbose=False):
    dim = embeddings.shape[1]

    if verbose:
        print("* Embedding shape:", embeddings.shape)
        if ids is not None:
            print("* Ids shape and type:", ids.shape, ids.dtype)

    if metric == "L2":
        metric = faiss.METRIC_L2
    elif metric == "IP":
        metric = faiss.METRIC_INNER_PRODUCT
    else:
        print("ERROR: please give a valid metric!")
        return None

    index = faiss.index_factory(dim, index_type, metric)
    index.verbose = True

    # index.train(embeddings)

    if ids is not None:
        ids = ids.astype(np.int64)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(embeddings, ids)
    else:
        index.add(embeddings)

    return index


def build_ivfindex_from_embeddings(embeddings, n_clusters=16384, n_samples=50, ids=None, metric='L2', index_type="Flat",
                                   verbose=False):
    dim = embeddings.shape[1]

    if verbose:
        print("* Embedding shape:", str(embeddings.shape))
        if ids is not None:
            print("* Ids shape and type:", ids.shape, ids.dtype)

    if metric == "L2":
        metric = faiss.METRIC_L2
    elif metric == "IP":
        metric = faiss.METRIC_INNER_PRODUCT
    else:
        print("ERROR: please give a valid metric!")
        return None

    n_samples_train = n_clusters * n_samples

    # sampling data
    train_indexes = np.random.choice(embeddings.shape[0], n_samples_train, replace=False)
    training_embeddings = embeddings[train_indexes]

    if verbose:
        print("* Training sample shape:", training_embeddings.shape)

    index = faiss.index_factory(dim, index_type, metric)
    index.verbose = True

    index.train(training_embeddings)

    if verbose:
        print("* Is index trained?:", index.is_trained)

    if ids is not None:
        ids = ids.astype(np.int64)
        # index = faiss.IndexIDMap2(index)
        index.add_with_ids(embeddings, ids)
    else:
        index.add(embeddings)

    if verbose:
        print("* Total number of embeddings indexed:", index.ntotal)

    return index
