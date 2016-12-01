/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/traits.hpp>
#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <testHelpers.hpp>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::abs;
using af::cfloat;
using af::cdouble;

///////////////////////////////// CPP ////////////////////////////////////
//

template<typename T>
af::array makeSparse(af::array A, int factor)
{
    A = floor(A * 1000);
    A = A * ((A % factor) == 0) / 1000;
    return A;
}

template<>
af::array makeSparse<cfloat>(af::array A, int factor)
{
    af::array r = real(A);
    r = floor(r * 1000);
    r = r * ((r % factor) == 0) / 1000;

    af::array i = r / 2;

    A = af::complex(r, i);
    return A;
}

template<>
af::array makeSparse<cdouble>(af::array A, int factor)
{
    af::array r = real(A);
    r = floor(r * 1000);
    r = r * ((r % factor) == 0) / 1000;

    af::array i = r / 2;

    A = af::complex(r, i);
    return A;
}

double calc_norm(af::array lhs, af::array rhs)
{
    return af::max<double>(af::abs(lhs - rhs) / (af::abs(lhs) + af::abs(rhs) + 1E-5));
}

template<typename T>
void sparseTester(const int m, const int n, const int k, int factor, double eps)
{
    af::deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    af::array A = cpu_randu<T>(af::dim4(m, n));
    af::array B = cpu_randu<T>(af::dim4(n, k));
#else
    af::array A = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array B = af::randu(n, k, (af::dtype)af::dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    // Result of GEMM
    af::array dRes1 = matmul(A, B);

    // Create Sparse Array From Dense
    af::array sA = af::sparse(A, AF_STORAGE_CSR);

    // Sparse Matmul
    af::array sRes1 = matmul(sA, B);

    // Verify Results
    ASSERT_NEAR(0, calc_norm(real(dRes1), real(sRes1)), eps);
    ASSERT_NEAR(0, calc_norm(imag(dRes1), imag(sRes1)), eps);
}

template<typename T>
void sparseTransposeTester(const int m, const int n, const int k, int factor, double eps)
{
    af::deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    af::array A = cpu_randu<T>(af::dim4(m, n));
    af::array B = cpu_randu<T>(af::dim4(m, k));
#else
    af::array A = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array B = af::randu(m, k, (af::dtype)af::dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    // Result of GEMM
    af::array dRes2 = matmul(A, B, AF_MAT_TRANS, AF_MAT_NONE);
    af::array dRes3 = matmul(A, B, AF_MAT_CTRANS, AF_MAT_NONE);

    // Create Sparse Array From Dense
    af::array sA = af::sparse(A, AF_STORAGE_CSR);

    // Sparse Matmul
    af::array sRes2 = matmul(sA, B, AF_MAT_TRANS, AF_MAT_NONE);
    af::array sRes3 = matmul(sA, B, AF_MAT_CTRANS, AF_MAT_NONE);

    // Verify Results
    ASSERT_NEAR(0, calc_norm(real(dRes2), real(sRes2)), eps);
    ASSERT_NEAR(0, calc_norm(imag(dRes2), imag(sRes2)), eps);

    ASSERT_NEAR(0, calc_norm(real(dRes3), real(sRes3)), eps);
    ASSERT_NEAR(0, calc_norm(imag(dRes3), imag(sRes3)), eps);
}

template<typename T>
void convertCSR(const int M, const int N, const float ratio)
{
    if (noDoubleTests<T>()) return;
#if 1
    af::array a = cpu_randu<T>(af::dim4(M, N));
#else
    af::array a = af::randu(M, N);
#endif
    a = a * (a > ratio);

    af::array s = af::sparse(a, AF_STORAGE_CSR);
    af::array aa = af::dense(s);

    ASSERT_EQ(0, af::max<double>(af::abs(a - aa)));
}

#define SPARSE_TESTS(T, eps)                                \
    TEST(SPARSE, T##Square)                                 \
    {                                                       \
        sparseTester<T>(1000, 1000, 100, 5, eps);           \
    }                                                       \
    TEST(SPARSE, T##RectMultiple)                           \
    {                                                       \
        sparseTester<T>(2048, 1024, 512, 3, eps);           \
    }                                                       \
    TEST(SPARSE, T##RectDense)                              \
    {                                                       \
        sparseTester<T>(500, 1000, 250, 1, eps);            \
    }                                                       \
    TEST(SPARSE, T##MatVec)                                 \
    {                                                       \
        sparseTester<T>(625, 1331, 1, 2, eps);              \
    }                                                       \
    TEST(SPARSE_TRANSPOSE, T##MatVec)                       \
    {                                                       \
        sparseTransposeTester<T>(625, 1331, 1, 2, eps);     \
    }                                                       \
    TEST(SPARSE_TRANSPOSE, T##Square)                       \
    {                                                       \
        sparseTransposeTester<T>(1000, 1000, 100, 5, eps);  \
    }                                                       \
    TEST(SPARSE_TRANSPOSE, T##RectMultiple)                 \
    {                                                       \
        sparseTransposeTester<T>(2048, 1024, 512, 3, eps);  \
    }                                                       \
    TEST(SPARSE_TRANSPOSE, T##RectDense)                    \
    {                                                       \
        sparseTransposeTester<T>(453, 751, 397, 1, eps);    \
    }                                                       \
    TEST(SPARSE, T##ConvertCSR)                             \
    {                                                       \
        convertCSR<T>(2345, 5678, 0.5);                     \
    }                                                       \

SPARSE_TESTS(float, 1E-3)
SPARSE_TESTS(double, 1E-5)
SPARSE_TESTS(cfloat, 1E-3)
SPARSE_TESTS(cdouble, 1E-5)

#undef SPARSE_TESTS

// This test essentially verifies that the sparse structures have the correct
// dimensions and indices using a very basic test
template<af_storage stype>
void createFunction()
{
    af::array in = af::sparse(af::identity(3, 3), stype);

    af::array values = sparseGetValues(in);
    af::array rowIdx = sparseGetRowIdx(in);
    af::array colIdx = sparseGetColIdx(in);
    dim_t     nNZ    = sparseGetNNZ(in);

    ASSERT_EQ(nNZ, values.elements());

    ASSERT_EQ(0, af::max<double>(values - af::constant(1, nNZ)));
    ASSERT_EQ(0, af::max<int   >(rowIdx - af::range(af::dim4(rowIdx.elements()), 0, s32)));
    ASSERT_EQ(0, af::max<int   >(colIdx - af::range(af::dim4(colIdx.elements()), 0, s32)));
}

#define CREATE_TESTS(STYPE)                                         \
    TEST(SPARSE_CREATE, STYPE)                                      \
    {                                                               \
        createFunction<STYPE>();                                    \
    }

CREATE_TESTS(AF_STORAGE_CSR)
CREATE_TESTS(AF_STORAGE_COO)

#undef CREATE_TESTS

template<typename T, af_storage src, af_storage dest>
void sparseConvertTester(const int m, const int n, int factor)
{
    af::deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    af::array A = cpu_randu<T>(af::dim4(m, n));
#else
    af::array A = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    // Create Sparse Array of type src and dest From Dense
    af::array sA = af::sparse(A, src);

    // Convert src to dest format and dest to src
    af::array s2d = sparseConvertTo(sA, dest);

    // Create the dest type from dense - gold
    af::array dA = af::sparse(A, dest);

    // Verify nnZ
    dim_t dNNZ   = sparseGetNNZ(dA);
    dim_t s2dNNZ = sparseGetNNZ(s2d);

    ASSERT_EQ(dNNZ, s2dNNZ);

    // Verify Types
    af_storage dType   = sparseGetStorage(dA);
    af_storage s2dType = sparseGetStorage(s2d);

    ASSERT_EQ(dType, s2dType);

    // Get the individual arrays and verify equality
    af::array dValues = sparseGetValues(dA);
    af::array dRowIdx = sparseGetRowIdx(dA);
    af::array dColIdx = sparseGetColIdx(dA);

    af::array s2dValues = sparseGetValues(s2d);
    af::array s2dRowIdx = sparseGetRowIdx(s2d);
    af::array s2dColIdx = sparseGetColIdx(s2d);

    // Verify values
    ASSERT_EQ(0, af::max<double>(af::abs(dValues - s2dValues)));

    // Verify row and col indices
    ASSERT_EQ(0, af::max<int   >(dRowIdx - s2dRowIdx));
    ASSERT_EQ(0, af::max<int   >(dColIdx - s2dColIdx));
}

#define CONVERT_TESTS_TYPES(T, STYPE, DTYPE, SUFFIX, M, N, F)                   \
    TEST(SPARSE_CONVERT, T##_##STYPE##_##DTYPE##_##SUFFIX)                      \
    {                                                                           \
        sparseConvertTester<T, STYPE, DTYPE>(M, N, F);                          \
    }                                                                           \
    TEST(SPARSE_CONVERT, T##_##DTYPE##_##STYPE##_##SUFFIX)                      \
    {                                                                           \
        sparseConvertTester<T, DTYPE, STYPE>(M, N, F);                          \
    }                                                                           \

#define CONVERT_TESTS(T, STYPE, DTYPE)                                          \
    CONVERT_TESTS_TYPES(T, STYPE, DTYPE, 1, 1000, 1000,  5)                     \
    CONVERT_TESTS_TYPES(T, STYPE, DTYPE, 2,  512,  512,  1)                     \
    CONVERT_TESTS_TYPES(T, STYPE, DTYPE, 3,  512, 1024,  2)                     \
    CONVERT_TESTS_TYPES(T, STYPE, DTYPE, 4, 2048, 1024, 10)                     \

CONVERT_TESTS(float  , AF_STORAGE_CSR, AF_STORAGE_COO)
CONVERT_TESTS(double , AF_STORAGE_CSR, AF_STORAGE_COO)
CONVERT_TESTS(cfloat , AF_STORAGE_CSR, AF_STORAGE_COO)
CONVERT_TESTS(cdouble, AF_STORAGE_CSR, AF_STORAGE_COO)

template<typename Ti, typename To>
void sparseCastTester(const int m, const int n, int factor)
{
    af::deviceGC();

    if (noDoubleTests<Ti>()) return;
    if (noDoubleTests<To>()) return;

#if 1
    af::array A = cpu_randu<Ti>(af::dim4(m, n));
#else
    af::array A = af::randu(m, n, (af::dtype)af::dtype_traits<Ti>::af_type);
#endif

    A = makeSparse<Ti>(A, factor);

    af::array sTi = af::sparse(A, AF_STORAGE_CSR);

    // Cast
    af::array sTo = sTi.as((af::dtype)af::dtype_traits<To>::af_type);

    // Verify nnZ
    dim_t iNNZ = sparseGetNNZ(sTi);
    dim_t oNNZ = sparseGetNNZ(sTo);

    ASSERT_EQ(iNNZ, oNNZ);

    // Verify Types
    dim_t iSType = sparseGetStorage(sTi);
    dim_t oSType = sparseGetStorage(sTo);

    ASSERT_EQ(iSType, oSType);

    // Get the individual arrays and verify equality
    af::array iValues = sparseGetValues(sTi);
    af::array iRowIdx = sparseGetRowIdx(sTi);
    af::array iColIdx = sparseGetColIdx(sTi);

    af::array oValues = sparseGetValues(sTo);
    af::array oRowIdx = sparseGetRowIdx(sTo);
    af::array oColIdx = sparseGetColIdx(sTo);

    // Verify values
    ASSERT_EQ(0, af::max<int>(af::abs(iRowIdx - oRowIdx)));
    ASSERT_EQ(0, af::max<int>(af::abs(iColIdx - oColIdx)));

    if(iValues.iscomplex() && !oValues.iscomplex()) {
        ASSERT_NEAR(0, af::max<double>(af::abs(af::abs(iValues) - oValues)), 1e-6);
    } else if(!iValues.iscomplex() && oValues.iscomplex()) {
        ASSERT_NEAR(0, af::max<double>(af::abs(iValues - af::abs(oValues))), 1e-6);
    } else {
        ASSERT_NEAR(0, af::max<double>(af::abs(iValues - oValues)), 1e-6);
    }
}

#define CAST_TESTS_TYPES(Ti, To, SUFFIX, M, N, F)                               \
    TEST(SPARSE_CAST, Ti##_##To##_##SUFFIX)                                     \
    {                                                                           \
        sparseCastTester<Ti, To>(M, N, F);                                      \
    }                                                                           \

#define CAST_TESTS(Ti, To)                                                      \
    CAST_TESTS_TYPES(Ti, To, 1, 1000, 1000,  5)                                 \
    CAST_TESTS_TYPES(Ti, To, 2,  512, 1024,  2)                                 \

CAST_TESTS(float  , float   )
CAST_TESTS(float  , double  )
CAST_TESTS(float  , cfloat  )
CAST_TESTS(float  , cdouble )

CAST_TESTS(double , float   )
CAST_TESTS(double , double  )
CAST_TESTS(double , cfloat  )
CAST_TESTS(double , cdouble )

CAST_TESTS(cfloat , cfloat  )
CAST_TESTS(cfloat , cdouble )

CAST_TESTS(cdouble, cfloat  )
CAST_TESTS(cdouble, cdouble )


typedef enum {
    af_add_t,
    af_sub_t,
    af_mul_t,
    af_div_t,
} af_op_t;

template<af_op_t op>
struct arith_op
{
    af::array operator()(af::array v1, af::array v2)
    {
        return v1;
    }
};

template<>
struct arith_op<af_add_t>
{
    af::array operator()(af::array v1, af::array v2)
    {
        return v1 + v2;
    }
};

template<>
struct arith_op<af_sub_t>
{
    af::array operator()(af::array v1, af::array v2)
    {
        return v1 - v2;
    }
};

template<>
struct arith_op<af_mul_t>
{
    af::array operator()(af::array v1, af::array v2)
    {
        return v1 * v2;
    }
};

template<>
struct arith_op<af_div_t>
{
    af::array operator()(af::array v1, af::array v2)
    {
        return v1 / v2;
    }
};

template<typename T, af_op_t op>
void sparseArithTester(const int m, const int n, int factor, const double eps)
{
    af::deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    af::array A = cpu_randu<T>(af::dim4(m, n));
    af::array B = cpu_randu<T>(af::dim4(m, n));
#else
    af::array A = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array B = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    af::array SA = af::sparse(A, AF_STORAGE_CSR);
    af::array OA = af::sparse(A, AF_STORAGE_COO);

    // Arith Op
    af::array resS = arith_op<op>()(SA, B);
    af::array resO = arith_op<op>()(OA, B);
    af::array resD = arith_op<op>()( A, B);

    af::array revS = arith_op<op>()(B, SA);
    af::array revO = arith_op<op>()(B, OA);
    af::array revD = arith_op<op>()(B,  A);

    ASSERT_NEAR(0, af::sum<double>(af::abs(real(resS - resD))) / (m * n), eps);
    ASSERT_NEAR(0, af::sum<double>(af::abs(imag(resS - resD))) / (m * n), eps);

    ASSERT_NEAR(0, af::sum<double>(af::abs(real(resO - resD))) / (m * n), eps);
    ASSERT_NEAR(0, af::sum<double>(af::abs(imag(resO - resD))) / (m * n), eps);

    ASSERT_NEAR(0, af::sum<double>(af::abs(real(revS - revD))) / (m * n), eps);
    ASSERT_NEAR(0, af::sum<double>(af::abs(imag(revS - revD))) / (m * n), eps);

    ASSERT_NEAR(0, af::sum<double>(af::abs(real(revO - revD))) / (m * n), eps);
    ASSERT_NEAR(0, af::sum<double>(af::abs(imag(revO - revD))) / (m * n), eps);
}

template<typename T>
void sparseArithTesterDiv(const int m, const int n, int factor, const double eps)
{
    af::deviceGC();

    if (noDoubleTests<T>()) return;

#if 1
    af::array A = cpu_randu<T>(af::dim4(m, n));
    af::array B = cpu_randu<T>(af::dim4(m, n));
#else
    af::array A = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
    af::array B = af::randu(m, n, (af::dtype)af::dtype_traits<T>::af_type);
#endif

    A = makeSparse<T>(A, factor);

    af::array SA = af::sparse(A, AF_STORAGE_CSR);
    af::array OA = af::sparse(A, AF_STORAGE_COO);

    // Arith Op
    af::array resS = arith_op<af_div_t>()(SA, B);
    af::array resO = arith_op<af_div_t>()(OA, B);
    af::array resD = arith_op<af_div_t>()( A, B);

    af::array revS = arith_op<af_div_t>()(B, SA);
    af::array revO = arith_op<af_div_t>()(B, OA);
    af::array revD = arith_op<af_div_t>()(B,  A);

    T *hResS = resS.host<T>();
    T *hResO = resO.host<T>();
    T *hResD = resD.host<T>();
    T *hRevS = revS.host<T>();
    T *hRevO = revO.host<T>();
    T *hRevD = revD.host<T>();

    for(int i = 0; i < B.elements(); i++) {
        ASSERT_EQ(hResS[i], hResD[i]) << "at : " << i;
        ASSERT_EQ(hResO[i], hResD[i]) << "at : " << i;
        ASSERT_EQ(hRevS[i], hRevD[i]) << "at : " << i;
        ASSERT_EQ(hRevO[i], hRevD[i]) << "at : " << i;
    }

    af::freeHost(hResS);
    af::freeHost(hResO);
    af::freeHost(hResD);
    af::freeHost(hRevS);
    af::freeHost(hRevO);
    af::freeHost(hRevD);
}

#define ARITH_TESTS_OPS(T, M, N, F, EPS)                                    \
    TEST(SPARSE_ARITH, T##_ADD_##M##_##N)                                   \
    {                                                                       \
        sparseArithTester<T, af_add_t>(M, N, F, EPS);                       \
    }                                                                       \
    TEST(SPARSE_ARITH, T##_SUB_##M##_##N)                                   \
    {                                                                       \
        sparseArithTester<T, af_sub_t>(M, N, F, EPS);                       \
    }                                                                       \
    TEST(SPARSE_ARITH, T##_MUL_##M##_##N)                                   \
    {                                                                       \
        sparseArithTester<T, af_mul_t>(M, N, F, EPS);                       \
    }                                                                       \
    TEST(SPARSE_ARITH, T##_DIV_##M##_##N)                                   \
    {                                                                       \
        sparseArithTesterDiv<T>(M, N, F, EPS);                              \
    }                                                                       \

#define ARITH_TESTS(T)                                                      \
    ARITH_TESTS_OPS(T, 10  , 10  , 5, 1e-6)                                 \
    ARITH_TESTS_OPS(T, 1024, 1024, 5, 1e-6)                                 \
    ARITH_TESTS_OPS(T, 100 , 100 , 1, 1e-6)                                 \
    ARITH_TESTS_OPS(T, 2048, 1000, 6, 1e-6)                                 \
    ARITH_TESTS_OPS(T, 123 , 278 , 5, 1e-6)                                 \

ARITH_TESTS(float  )
ARITH_TESTS(double )
ARITH_TESTS(cfloat )
ARITH_TESTS(cdouble)
