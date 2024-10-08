#!/bin/bash

INDEX_PATH=PATH_TO_INDEXES # Path to the indexes
DATA_PATH=PATH_TO_VECTORS # Path to the data
MODELS_PATH=PATH_TO_MODELS # Path to the models
TEST_OFFSETS=SOME_OTHER_PATH/test_offsets.npy # Path to the test offsets
DIM_LIGHTGBM=100 # Dimension of the LightGBM model

for encoder in contriver star tasb; do
  echo "Running $encoder"
  if [ $encoder = contriver ]; then
      patience=10
      probes=140
      w=7
  fi

  if [ $encoder = star ]; then
      patience=7
      probes=80
      w=3
  fi

  if [ $encoder = tasb ]; then
      patience=14
      probes=190
      w=3
  fi

  echo "base aknn"
    ./faiss_paknn --index $INDEX_PATH/msmarco.$encoder.IVF65535.Flat.dot.fidx \
    --data $DATA_PATH/$encoder/query.dev.npy \
    --np $probes -k 100 --n_runs 5 --patience 0 --patience_tol 0 --exit 10 --test_offsets $TEST_OFFSETS
  echo "-----------------------------------"

  echo "Li Regressor"
    # regressor without intersections
    ./faiss_paknn --index $INDEX_PATH/msmarco.$encoder.IVF65535.Flat.dot.fidx \
    --data $DATA_PATH/$encoder/query.dev.npy \
    --model $MODELS_PATH --model_name R.$encoder.S-10.I-NI.D-$DIM_LIGHTGBM.txt  \
    --np $probes -k 100 --n_runs 5 --patience 0 --patience_tol 0 --exit 10 --test_offsets $TEST_OFFSETS --is_classifier false
  echo "-----------------------------------"

  echo "Int Regressor"
    # regressor with intersections
    ./faiss_paknn --index $INDEX_PATH/msmarco.$encoder.IVF65535.Flat.dot.fidx \
    --data $DATA_PATH/$encoder/query.dev.npy \
    --model $MODELS_PATH --model_name R.$encoder.S-10.I-I.D-$DIM_LIGHTGBM.txt  \
    --np $probes -k 100 --n_runs 5 --patience 0 --patience_tol 0 --exit 10 --test_offsets $TEST_OFFSETS --is_classifier false
  echo "-----------------------------------"

  echo "Patience"
    # patience
    ./faiss_paknn --index $INDEX_PATH/msmarco.$encoder.IVF65535.Flat.dot.fidx \
    --data $DATA_PATH/$encoder/query.dev.npy \
    --np $probes -k 100 --n_runs 5 --patience $patience --patience_tol 0.95 --exit 0 --test_offsets $TEST_OFFSETS
  echo "-----------------------------------"

  echo "Classifier w = 1"
    # classifier
    ./faiss_paknn --index $INDEX_PATH/msmarco.$encoder.IVF65535.Flat.dot.fidx \
    --data $DATA_PATH/$encoder/query.dev.npy \
    --model $MODELS_PATH --model_name C.$encoder.S-10.I-I.W-1.D-$DIM_LIGHTGBM.txt  \
    --np $probes -k 100 --n_runs 5 --patience 0 --patience_tol 0 --exit 10 --test_offsets $TEST_OFFSETS --is_classifier true
  echo "-----------------------------------"

  echo "Classifier w = $w"
    # classifier with w=3
    ./faiss_paknn --index $INDEX_PATH/msmarco.$encoder.IVF65535.Flat.dot.fidx \
    --data $DATA_PATH/$encoder/query.dev.npy \
    --model $MODELS_PATH --model_name C.$encoder.S-10.I-I.W-$w.D-$DIM_LIGHTGBM.txt  \
    --np $probes -k 100 --n_runs 5 --patience 0 --patience_tol 0 --exit 10 --test_offsets $TEST_OFFSETS --is_classifier true
  echo "-----------------------------------"

  echo "Clf > Regressor (w = $w)"
    # Clf > Regressor
    ./faiss_paknn --index $INDEX_PATH/msmarco.$encoder.IVF65535.Flat.dot.fidx \
    --data $DATA_PATH/$encoder/query.dev.npy \
    --model $MODELS_PATH \
    --model_name R.$encoder.S-10.I-I.D-$DIM_LIGHTGBM.txt  \
    --masker C.$encoder.S-10.I-I.W-$w.D-$DIM_LIGHTGBM.txt \
    --np $probes -k 100 --n_runs 5 --patience 0 --patience_tol 0 --exit 10 --test_offsets $TEST_OFFSETS --is_classifier false
  echo "-----------------------------------"

  echo "Clf > Patience"
    # clf > patience
    ./faiss_paknn --index $INDEX_PATH/msmarco.$encoder.IVF65535.Flat.dot.fidx \
    --data $DATA_PATH/$encoder/query.dev.npy \
    --model $MODELS_PATH \
    --masker C.$encoder.S-10.I-I.W-$w.D-$DIM_LIGHTGBM.txt \
    --np $probes -k 100 --n_runs 5 --patience $patience --patience_tol 0.95 --exit 10 --test_offsets $TEST_OFFSETS
  echo "-----------------------------------"

done
