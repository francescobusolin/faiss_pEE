#include <chrono>

#include <faiss/IndexIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/index_io.h>

#include <LightGBM/boosting.h>
#include <LightGBM/prediction_early_stop.h>
#include "npy.h"
#include "cxxopts.hpp"


// 64-bit int
using idx_t = faiss::idx_t;

std::string str_append(const char* a, const char* b) {
    std::string str(a);
    str.append(b);
    return str;
}


int main(int argc, char** argv ) {

    std::string index_path;
    std::string data_path;
    std::string model_path;
    std::string model_name;
    std::string masker_name;
    std::string test_offsets_path;
    size_t np;
    size_t k;
    size_t n_runs;
    size_t patience;
    float patience_tol;
    idx_t sentinel;
    bool is_classifier;

    cxxopts::Options options("faiss_padaknn", "Search using a faiss index with Earlu stopping");
    options.add_options()
            ("i,index", "Index path", cxxopts::value<std::string>(index_path))
            ("d,data", "Data path", cxxopts::value<std::string>(data_path))
            ("m,model", "Model path", cxxopts::value<std::string>(model_path))
            ("mn,model_name", "Model name", cxxopts::value<std::string>(model_name))
            ("masker", "Masker name", cxxopts::value<std::string>(masker_name))
            ("np", "Number of probes", cxxopts::value<size_t>(np))
            ("k", "Number of results", cxxopts::value<size_t>(k))
            ("r,n_runs", "Number of runs", cxxopts::value<size_t>(n_runs))
            ("p,patience", "Patience", cxxopts::value<size_t>(patience))
            ("t,patience_tol", "Patience tolerance", cxxopts::value<float>(patience_tol))
            ("e,exit", "Exit Point", cxxopts::value<idx_t>(sentinel))
            ("c,is_classifier", "Is classifier", cxxopts::value<bool>(is_classifier))
            ("test_offsets", "test offsets of data", cxxopts::value<std::string>(test_offsets_path))
            ("h,help", "Print usage")
            ;



    options.parse(argc, argv);

    const size_t n_features = 768 + 14 + ( 2 * (10 - 1) ) + 10; // d + 14 + intersections + close clusters

    auto searchParams = faiss::SearchParametersIVF();
    searchParams.nprobe = np;
    searchParams.patience = patience;
    searchParams.tolerance = patience_tol;
    searchParams.exit_index  = sentinel;
    searchParams.n_features = n_features;


    printf("Loading model from %s\n", model_path.c_str());
    LightGBM::Boosting *regressor = nullptr;
    LightGBM::Boosting *first_stage = nullptr;

    if (model_name != "") {
        regressor = LightGBM::Boosting::CreateBoosting(std::string("gbdt"), (model_path + model_name).c_str());
        printf("R Model of %d trees loaded\n", regressor->NumberOfTotalModel());
    }
    if (masker_name != "") {
        first_stage = LightGBM::Boosting::CreateBoosting(std::string("gbdt"), (model_path + masker_name).c_str());
        printf("C Model of %d trees loaded\n", first_stage->NumberOfTotalModel());
    }



    LightGBM::PredictionEarlyStopConfig tree_config;
    LightGBM::PredictionEarlyStopInstance tree_early_stop =
            LightGBM::CreatePredictionEarlyStopInstance(std::string("none"),
                                                        tree_config);


    printf("Loading data from %s\n", data_path.c_str());

    npy::npy_data<float> full_data = npy::read_npy<float>(data_path);
    printf("Data shape: %lu x %lu\n", full_data.shape[0], full_data.shape[1]);
    npy::npy_data<int64_t> offsets = npy::read_npy<int64_t>(test_offsets_path);

    size_t nq = offsets.shape[0];
    float* const d = new float[nq * 768];
    for (int i = 0; i < nq; i++){
        memcpy(d + i * 768, full_data.data.data() + offsets.data[i] * 768, 768 * sizeof(float));
    }


    printf("Data shape: %lu x %d\n", nq, 768);
    printf("Loading index from %s\n", index_path.c_str());
    // auto index_ivf = faiss::read_index(index_path); // << This should be the correct way to load the index
    auto index_idmap = dynamic_cast<faiss::IndexIDMap*>(faiss::read_index(index_path.c_str()));
    auto index_ivf = index_idmap->index;
    printf("Loaded index with: %lu elements\n", index_ivf->ntotal);


    idx_t* const prev_search_buffer = new idx_t[nq * k * 10];
    idx_t* const first_search_buffer = new idx_t[nq * k];
    idx_t* const patience_buffer = new idx_t[nq];
    double* const feature_buffer = new double[nq * n_features];

    memset(prev_search_buffer, 0, nq * k * 10 * sizeof(idx_t));
    memset(first_search_buffer, 0, nq * k * sizeof(idx_t));
    memset(patience_buffer, 0, nq * sizeof(idx_t));
    memset(feature_buffer, 0, nq * n_features * sizeof(double));

    if (regressor){
        regressor->InitPredict(0, regressor->NumberOfTotalModel(), true);
        searchParams.probe_predictor = regressor;
        searchParams.is_classifier = model_name[0] == 'C';
    }
    if (first_stage){
        first_stage->InitPredict(0, first_stage->NumberOfTotalModel(), true);
        searchParams.first_stage_clf = first_stage;
    }


    searchParams.lgb_tree_config = tree_config;
    searchParams.lgb_tree_early_stop = tree_early_stop;

    searchParams.feature_buffer = feature_buffer;
    searchParams.previous_search_buffer = prev_search_buffer;
    searchParams.first_search_buffer = first_search_buffer;
    searchParams.stable_probes_buffer = patience_buffer;

    printf("Searching\n");
    printf("Search parameters@%d: nprobe=%d, patience=%d, tolerance=%f, predictor=%p (clf: %d)\n",
           searchParams.exit_index,
           searchParams.nprobe,
           searchParams.patience,
           searchParams.tolerance,
           searchParams.probe_predictor,
           searchParams.is_classifier);
    idx_t* I = new idx_t[k * nq];
    float* D = new float[k * nq];
    auto embeddings = d;


    index_ivf->search(nq, embeddings, k, D, I, &searchParams); // warm-up; throw away first run
    auto start = std::chrono::high_resolution_clock::now();
    for (int runs = 0; runs < n_runs; runs++) {
        index_ivf->search(nq, embeddings, k, D, I, &searchParams);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = (end - start) / n_runs;

    for (idx_t i = 0; i < 100 * np; i++) { // useful if id of doc is not sequential; ours is
        I[i] = I[i] < 0 ? I[i] : index_idmap->id_map[I[i]];
    }

    for (int i = 0; i < 10; i++) {
        printf("Q %lu:", i);
        for (int j = 0; j < 10; j++) {
            printf(" %lu", I[i * k + j]);
        }
        printf("\n");
    }
    printf("Search time: %.4f s\n", elapsed.count());
    printf("Search time per query: %.4f ms\n", 1000 * elapsed.count() / nq);
    printf("Documents scanned: %lu\n", faiss::indexIVF_stats.ndis);
    printf("Cluster scanned: %lu\n", faiss::indexIVF_stats.nlist);

    delete[] I;
    delete[] D;
    delete[] d;

    delete[] prev_search_buffer;
    delete[] first_search_buffer;
    delete[] patience_buffer;
    delete[] feature_buffer;
    return 0;
}

