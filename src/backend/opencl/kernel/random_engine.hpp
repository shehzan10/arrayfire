/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <platform.hpp>
#include <af/defines.h>
#include <kernel_headers/random_engine_philox.hpp>
#include <kernel_headers/random_engine_threefry.hpp>
#include <kernel_headers/random_engine_write.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>
#include <dispatch.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <program.hpp>
#include <type_util.hpp>
#include <cache.hpp>
#include <random_engine.hpp>

#include <kernel_headers/random_engine_mersenne_init.hpp>
#include <kernel_headers/random_engine_mersenne.hpp>

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::KernelFunctor;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

static const int N = 351;
static const int TABLE_SIZE = 16;
static const int MAX_BLOCKS = 32;
static const int STATE_SIZE = (256*3);

namespace opencl
{
    namespace kernel
    {
        static const uint THREADS = 256;

        template <typename T>
        static Kernel get_random_engine_kernel(const af_random_type type, const int kerIdx, const uint elementsPerBlock)
        {
            using std::string;
            using std::to_string;
            string engineName;
            const char *ker_strs[2];
            int ker_lens[2];
            ker_strs[0] = random_engine_write_cl;
            ker_lens[0] = random_engine_write_cl_len;
            switch (type) {
                case AF_RANDOM_PHILOX_4X32_10   : engineName = "Philox";
                                                ker_strs[1] = random_engine_philox_cl;
                                                ker_lens[1] = random_engine_philox_cl_len;
                                                break;
                case AF_RANDOM_THREEFRY_2X32_16 : engineName = "Threefry";
                                                ker_strs[1] = random_engine_threefry_cl;
                                                ker_lens[1] = random_engine_threefry_cl_len;
                                                break;
                case AF_RANDOM_MERSENNE_GP11213 : engineName = "Mersenne";
                                                ker_strs[1] = random_engine_mersenne_cl;
                                                ker_lens[1] = random_engine_mersenne_cl_len;
                                                break;
                default                         : AF_ERROR("Random Engine Type Not Supported", AF_ERR_NOT_SUPPORTED);
            }

            string ref_name =
                "random_engine_kernel_" + engineName +
                "_" + string(dtype_traits<T>::getName()) +
                "_" + to_string(kerIdx);
            int device = getActiveDeviceId();
            kc_t::iterator idx = kernelCaches[device].find(ref_name);
            kc_entry_t entry;
            if (idx == kernelCaches[device].end()) {
                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName()
                        << " -D THREADS=" << THREADS
                        << " -D RAND_DIST=" << kerIdx;
                if (type == AF_RANDOM_MERSENNE_GP11213) {
                    options << " -D STATE_SIZE=" << STATE_SIZE
                            << " -D TABLE_SIZE=" << TABLE_SIZE
                            << " -D N=" << N;
                } else {
                    options << " -D ELEMENTS_PER_BLOCK=" << elementsPerBlock;
                }
                if (std::is_same<T, double>::value) {
                    options << " -D USE_DOUBLE";
                }
#if defined(OS_MAC) // Because apple is "special"
                options << " -D IS_APPLE"
                        << " -D log10_val=" << std::log(10.0);
#endif
                cl::Program prog;
                buildProgram(prog, 2, ker_strs, ker_lens, options.str());
                entry.prog = new Program(prog);
                entry.ker = new Kernel(*entry.prog, "generate");
                kernelCaches[device][ref_name] = entry;
            } else {
                entry = idx->second;
            }

            return *entry.ker;
        }

        static Kernel get_mersenne_init_kernel(void)
        {
            using std::string;
            using std::to_string;
            string engineName;
            const char *ker_str = random_engine_mersenne_init_cl;
            int ker_len = random_engine_mersenne_init_cl_len;
            string ref_name = "mersenne_init";
            int device = getActiveDeviceId();
            kc_t::iterator idx = kernelCaches[device].find(ref_name);
            kc_entry_t entry;
            if (idx == kernelCaches[device].end()) {
                std::ostringstream options;
                options << " -D N=" << N << " -D TABLE_SIZE=" << TABLE_SIZE;
                cl::Program prog;
                buildProgram(prog, 1, &ker_str, &ker_len, options.str());
                entry.prog = new Program(prog);
                entry.ker = new Kernel(*entry.prog, "initState");
                kernelCaches[device][ref_name] = entry;
            } else {
                entry = idx->second;
            }

            return *entry.ker;
        }

        template <typename T>
        static void randomDistribution(cl::Buffer out, const size_t elements,
                const af_random_type type, const uintl &seed, uintl &counter, int kerIdx)
        {
            try {
                uint elementsPerBlock = THREADS*4*sizeof(uint)/sizeof(T);
                uint groups = divup(elements, elementsPerBlock);

                uint hi = seed>>32;
                uint lo = seed;

                NDRange local(THREADS, 1);
                NDRange global(THREADS * groups, 1);

                if ((type == AF_RANDOM_PHILOX_4X32_10) || (type == AF_RANDOM_THREEFRY_2X32_16)) {
                    Kernel ker = get_random_engine_kernel<T>(type, kerIdx, elementsPerBlock);
                    auto randomEngineOp = KernelFunctor<cl::Buffer, uint, uint, uint, uint>(ker);
                    randomEngineOp(EnqueueArgs(getQueue(), global, local),
                            out, elements, counter, hi, lo);
                }

                counter += elements;
                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
            }
        }

        template <typename T>
        void randomDistribution(cl::Buffer out, const size_t elements,
                cl::Buffer state, cl::Buffer pos, cl::Buffer sh1, cl::Buffer sh2,
                const uint mask, cl::Buffer recursion_table, cl::Buffer temper_table,
                int kerIdx)
        {
            try {
                int threads = THREADS;
                int min_elements_per_block = 32*THREADS*4*sizeof(uint)/sizeof(T);
                int blocks = divup(elements, min_elements_per_block);
                blocks = (blocks > MAX_BLOCKS)? MAX_BLOCKS : blocks;
                int elementsPerBlock = divup(elements, blocks);

                NDRange local(threads, 1);
                NDRange global(threads * blocks, 1);
                Kernel ker = get_random_engine_kernel<T>(AF_RANDOM_MERSENNE_GP11213, kerIdx, elementsPerBlock);
                auto randomEngineOp = KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                     uint, cl::Buffer, cl::Buffer, uint, uint>(ker);
                randomEngineOp(EnqueueArgs(getQueue(), global, local),
                        out, state, pos, sh1, sh2, mask, recursion_table, temper_table, elementsPerBlock, elements);
                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
            }
        }

        template <typename T>
        void uniformDistributionCBRNG(cl::Buffer out, const size_t elements,
                const af_random_type type, const uintl &seed, uintl &counter)
        {
            randomDistribution<T>(out, elements, type, seed, counter, 0);
        }

        template <typename T>
        void normalDistributionCBRNG(cl::Buffer out, const size_t elements,
                const af_random_type type, const uintl &seed, uintl &counter)
        {
            randomDistribution<T>(out, elements, type, seed, counter, 1);
        }

        template <typename T>
        void uniformDistributionMT(cl::Buffer out, const size_t elements,
                cl::Buffer state, cl::Buffer pos, cl::Buffer sh1, cl::Buffer sh2,
                const uint mask, cl::Buffer recursion_table, cl::Buffer temper_table)
        {
            randomDistribution<T>(out, elements, state, pos, sh1, sh2, mask, recursion_table, temper_table, 0);
        }

        template <typename T>
        void normalDistributionMT(cl::Buffer out, const size_t elements,
                cl::Buffer state, cl::Buffer pos, cl::Buffer sh1, cl::Buffer sh2,
                const uint mask, cl::Buffer recursion_table, cl::Buffer temper_table)
        {
            randomDistribution<T>(out, elements, state, pos, sh1, sh2, mask, recursion_table, temper_table, 1);
        }

        void initMersenneState(cl::Buffer state, cl::Buffer table, const uintl &seed)
        {
            try{
                NDRange local(N, 1);
                NDRange global(N * MAX_BLOCKS, 1);

                Kernel ker = get_mersenne_init_kernel();
                auto initOp = KernelFunctor<cl::Buffer, cl::Buffer, uintl>(ker);
                initOp(EnqueueArgs(getQueue(), global, local), state, table, seed);
                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
            }
        }
    }
}
