/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/cudaUtils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
int invokesliceKernel(const T* input,T* output, int max_batch_size, int max_beam_width, int num_heads, int max_seqlen,
    int hidden_dim_per_head, int window_size, cudaStream_t stream = 0);

} // namespace kernels
    } // namespace kernels
    