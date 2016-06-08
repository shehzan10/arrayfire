/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <SparseArray.hpp>
#include <mkl_spblas.h>

namespace cpu
{

#ifdef USE_MKL
typedef char sp_op_t;
typedef MKL_Complex8  sp_cfloat;
typedef MKL_Complex16 sp_cdouble;
#endif

template<typename T, af_sparse_storage storage>
common::SparseArray<T> sparseConvertDenseToStorage(const Array<T> &in);

template<typename T, af_sparse_storage storage>
Array<T> sparseConvertStorageToDense(const common::SparseArray<T> &in);

template<typename T, af_sparse_storage src, af_sparse_storage dest>
common::SparseArray<T> sparseConvertStorageToStorage(const common::SparseArray<T> &in);

}
