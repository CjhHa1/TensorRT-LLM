#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/recentCache.h"


using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
template <typename T>
__global__ void sliceKernel(
    const T*  input,
    T*  output,
    int max_batch_size,
    int max_beam_width,
    int num_heads,
    int max_seqlen,
    int hidden_dim_per_head,
    int window_size)
{
    // global index
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int hidden_idx = blockIdx.z * blockDim.z + threadIdx.z;

    // offset 
    int num_elements = max_batch_size * max_beam_width * num_heads * hidden_dim_per_head;
    int input_offset = batch_idx * num_heads * max_seqlen * hidden_dim_per_head + head_idx * max_seqlen * hidden_dim_per_head + (max_seqlen - window_size) * hidden_dim_per_head;
    int output_offset = batch_idx * num_heads * window_size * hidden_dim_per_head + head_idx * window_size * hidden_dim_per_head;

    // check
    if (batch_idx < max_batch_size * max_beam_width && head_idx < num_heads && hidden_idx < hidden_dim_per_head)
    {

        for (int i = 0; i < window_size; ++i)
        {
            int input_idx = input_offset + i * hidden_dim_per_head + hidden_idx;
            int output_idx = output_offset + i * hidden_dim_per_head + hidden_idx;

            // copy
            if (input_idx < num_elements)
            {
                output[output_idx] = input[input_idx];
            }
        }
    }
}

template <typename T>
int invokesliceKernel(
    const T*  input,
    T*  output,
    int max_batch_size,
    int max_beam_width,
    int num_heads,
    int max_seqlen,
    int hidden_dim_per_head,
    int window_size,
    cudaStream_t stream)
{
    // 计算grid和block的大小
    dim3 block_size(8, 8, 8); 
    dim3 grid_size((max_batch_size * max_beam_width + block_size.x - 1) / block_size.x,
        (num_heads + block_size.y - 1) / block_size.y, (hidden_dim_per_head + block_size.z - 1) / block_size.z);

    // invoke kernel
    sliceKernel<T><<<grid_size, block_size, 0, stream>>>(
        input, output, max_batch_size, max_beam_width, num_heads, max_seqlen, hidden_dim_per_head, window_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return -1; 
    }

    return 0; 
}


#define INSTANTIATE_GENERAL_SLICE_KERNEL(T)  \
    template int invokesliceKernel(const T*  input, T*  output,int max_batch_size, int max_beam_width,\
    int num_heads,int max_seqlen,int hidden_dim_per_head,int window_size,\
    cudaStream_t stream);

INSTANTIATE_GENERAL_SLICE_KERNEL(float);
INSTANTIATE_GENERAL_SLICE_KERNEL(half);

#ifdef ENABLE_BF16
INSTANTIATE_GENERAL_SLICE_KERNEL(__nv_bfloat16);
#endif


}//namespace: kernels

}// namespace: tensorrt_llm

