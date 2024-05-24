# External Libraries
We use LightGBM v4.3.0 to train and evaluate our learned models: see [https://github.com/microsoft/LightGBM/releases/tag/v4.3.0]
Also, we tested our solution using openBLAS: see [https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.27]
So you need to have both these libraries available to be linked against our code.

# Compilation
In general, you can follow the instructions provided by FAISS to build from the source; however, note that we exclusively used and thus tested the CPU-based version as customization, and rebuilding from the source the GPU variant can and *will* be tedious and error-prone.
The CMake command that we used to compile our solution is:
```
cmake -DFAISS_ENABLE_GPU=OFF \
-DFAISS_ENABLE_PYTHON=OFF \
-DBUILD_TESTING=OFF \
-DBUILD_SHARED_LIBS=ON \
-DFAISS_ENABLE_C_API=OFF \
-DCMAKE_BUILD_TYPE=Release \
-DBLA_VENDOR=OpenBLAS \
-DMKL_LIBRARIES=<PATH_TO_BLAS_LIB>/openBLAS/libopenblas.a \
-DLIGHTGBM_LIBRARIES=<PATH_TO_LIGHTGBM_LIB>/LightGBM/lib_lightgbm.a \
-DLIGHTGBM_INCLUDE=<PATH_TO_LIGHTGBM>/LightGBM/include \
-DCMAKE_INSTALL_LIBDIR=lib  -B build .
```
From which you will need to specify the libraries and headers locations
Then, to build faiss run
```
make -j8 -C build faiss
```

Finally, you can go ahead and build our experiment.cpp file inside the execs folder using:
```
g++ experiment.cpp -I../faiss -I<PATH_TO_LIGHTGBM>/LightGBM/include  -L../faiss/build/faiss -L<PATH_TO_LIGHTGBM>/LightGBM -lfaiss -lgfortran -g -l_lightgbm -fopenmp -g -o faiss_paknn
```
Again, make sure that all the paths are correct.
