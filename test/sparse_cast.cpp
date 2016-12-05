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
