# External Libraries
We use LightGBM v4.3.0 to train and evaluate our learned models: see [https://github.com/microsoft/LightGBM/releases/tag/v4.3.0]
Also, we tested our solution using openBLAS: see [https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.27]
So you need to have both these libraries available to be linked against our code.

# Compilation
In general you can follow the instruction provided by FAISS to build from source; however note that we exclusively used the CPU-based version as customize and rebuild from source the GPU variant can and *will* be tedious and errore prone.
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
