# External Libraries
We use LightGBM v4.3.0 to train and evaluate our learned models: see [https://github.com/microsoft/LightGBM/releases/tag/v4.3.0] (LightGBM). 
Also, we tested our solution using openBLAS: see [https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.27] (openBLAS)
So you need to have both these libraries available to be linked against our code.

# Compilation
In general you can follow the instruction provided by FAISS to build from source; however note that we exclusively used the CPU-based version as customize and rebuild from source the GPU variant can and *will* be tedious and errore prone.
