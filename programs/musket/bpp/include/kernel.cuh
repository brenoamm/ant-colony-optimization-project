#include <limits.h>
#include <float.h>
	
namespace mkt{
namespace kernel{

template <typename T, typename R, typename F>
__global__ void map(T* in, R* out, unsigned int size, F func);

template <typename T, typename F>
__global__ void map_in_place(T* inout, unsigned int size, F func);		

template <typename T, typename R, typename F>
__global__ void map_index(T* in, R* out, unsigned int size, unsigned int offset, F func);

template <typename T, typename F>
__global__ void map_index_in_place(T* inout, unsigned int size, unsigned int offset, F func);

template <typename T, typename R, typename F>
__global__ void map_index(T* in, R* out, unsigned int rows, unsigned int columns, unsigned int row_offset, unsigned int column_offset, F func);

template <typename T, typename F>
__global__ void map_index_in_place(T* inout, unsigned int rows, unsigned int columns, unsigned int row_offset, unsigned int column_offset, F func);
template<typename T, typename F>
void fold_call(unsigned int size, T* d_idata, T* d_odata, int threads, int blocks, F& f, cudaStream_t& stream, int gpu);

template<typename T, typename F, unsigned int blockSize>
__global__ void fold(T *g_idata, T *g_odata, unsigned int n, F func);
template<typename T>
void reduce_plus_call(unsigned int size, T* d_idata, T* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu);
template<typename T>
void reduce_multiply_call(unsigned int size, T* d_idata, T* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu);
void reduce_min_call(unsigned int size, int* d_idata, int* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu);
void reduce_min_call(unsigned int size, float* d_idata, float* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu);
void reduce_min_call(unsigned int size, double* d_idata, double* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu);
void reduce_max_call(unsigned int size, int* d_idata, int* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu);
void reduce_max_call(unsigned int size, float* d_idata, float* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu);
void reduce_max_call(unsigned int size, double* d_idata, double* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu);

template<typename T, typename R, typename Functor>
void map_reduce_plus_call(unsigned int size, T* d_idata, R* d_odata, int threads, int blocks, Functor f, cudaStream_t& stream, int gpu);

template<typename T, unsigned int blockSize>
__global__ void reduce_plus(T *g_idata, T *g_odata, unsigned int n);

template<typename T, unsigned int blockSize>
__global__ void reduce_multiply(T *g_idata, T *g_odata, unsigned int n);

template<unsigned int blockSize>
__global__ void reduce_max(int *g_idata, int *g_odata, unsigned int n);

template<unsigned int blockSize>
__global__ void reduce_max(float *g_idata, float *g_odata, unsigned int n);

template<unsigned int blockSize>
__global__ void reduce_max(double *g_idata, double *g_odata, unsigned int n);

template<unsigned int blockSize>
__global__ void reduce_min(int *g_idata, int *g_odata, unsigned int n);

template<unsigned int blockSize>
__global__ void reduce_min(float *g_idata, float *g_odata, unsigned int n);

template<unsigned int blockSize>
__global__ void reduce_min(double *g_idata, double *g_odata, unsigned int n);

template<typename T, typename R, unsigned int blockSize, typename Functor>
__global__ void map_reduce_plus(T *g_idata, R *g_odata, unsigned int n, Functor f);

} //namespace kernel
} //namespace mkt

template <typename T, typename R, typename F>
__global__ void mkt::kernel::map(T* in, R* out, unsigned int size, F func)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < size) {
    out[x] = func(in[x]);
  }
}

template <typename T, typename F>
__global__ void mkt::kernel::map_in_place(T* inout, unsigned int size, F func)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < size) {
    inout[x] = func(inout[x]);
  }
}

template <typename T, typename R, typename F>
__global__ void mkt::kernel::map_index(T* in, R* out, unsigned int size, unsigned int offset, F func)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x < size) {
    out[x] = func(x + offset, in[x]);
  }
}

template <typename T, typename F>
__global__ void mkt::kernel::map_index_in_place(T* inout, unsigned int size, unsigned int offset, F func)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (x < size) {
    inout[x] = func(x + offset, inout[x]);
  }
}

template <typename T, typename R, typename F>
__global__ void mkt::kernel::map_index(T* in, R* out, unsigned int rows, unsigned int columns, unsigned int row_offset, unsigned int column_offset, F func)
{
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (y < rows) {
    if (x < columns) {
      out[y * columns + x] = func(y + row_offset, x + column_offset, in[y * columns + x]);
    }
  }
}

template <typename T, typename F>
__global__ void mkt::kernel::map_index_in_place(T* inout, unsigned int rows, unsigned int columns, unsigned int row_offset, unsigned int column_offset, F func)
{
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (y < rows) {
    if (x < columns) {
      inout[y * columns + x] = func(y + row_offset, x + column_offset, inout[y * columns + x]);
    }
  }
}
template<typename T, typename F>
void mkt::kernel::fold_call(unsigned int size, T* d_idata, T* d_odata, int threads, int blocks, T identity, F& f, cudaStream_t& stream, int gpu) {
  cudaSetDevice(gpu);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  unsigned int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    switch (threads) {
      case 1024:
        mkt::kernel::fold<T, F, 1024> <<<dimGrid, dimBlock, smemSize, stream>>>(
            d_idata, d_odata, size, identity, f);
        break;
      case 512:
        mkt::kernel::fold<T, F, 512> <<<dimGrid, dimBlock, smemSize, stream>>>(
            d_idata, d_odata, size, identity, f);
        break;
      case 256:
        mkt::kernel::fold<T, F, 256> <<<dimGrid, dimBlock, smemSize, stream>>>(
            d_idata, d_odata, size, identity, f);
        break;
      case 128:
        mkt::kernel::fold<T, F, 128> <<<dimGrid, dimBlock, smemSize, stream>>>(
            d_idata, d_odata, size, identity, f);
        break;
      case 64:
        mkt::kernel::fold<T, F, 64> <<<dimGrid, dimBlock, smemSize, stream>>>(
            d_idata, d_odata, size, identity, f);
        break;
      case 32:
        mkt::kernel::fold<T, F, 32> <<<dimGrid, dimBlock, smemSize, stream>>>(
            d_idata, d_odata, size, identity, f);
        break;
      case 16:
        mkt::kernel::fold<T, F, 16> <<<dimGrid, dimBlock, smemSize, stream>>>(
            d_idata, d_odata, size, identity, f);
        break;
      case 8:
        mkt::kernel::fold<T, F, 8> <<<dimGrid, dimBlock, smemSize, stream>>>(
            d_idata, d_odata, size, identity, f);
        break;
      case 4:
        mkt::kernel::fold<T, F, 4> <<<dimGrid, dimBlock, smemSize, stream>>>(
            d_idata, d_odata, size, identity, f);
        break;
      case 2:
        mkt::kernel::fold<T, F, 2> <<<dimGrid, dimBlock, smemSize, stream>>>(
            d_idata, d_odata, size, identity, f);
        break;
      case 1:
        mkt::kernel::fold<T, F, 1> <<<dimGrid, dimBlock, smemSize, stream>>>(
            d_idata, d_odata, size, identity, f);
        break;
    }
  }

	template<typename T, typename F, unsigned int blockSize>
	__global__ void mkt::kernel::fold(T *g_idata, T *g_odata, unsigned int n, T identity, F func) {
	  extern __shared__ T sdata_t[];
	
	  // perform first level of reduction,
	  // reading from global memory, writing to shared memory
	  unsigned int tid = threadIdx.x;
	  unsigned int i = blockIdx.x * blockSize + threadIdx.x;
	  unsigned int gridSize = blockSize * gridDim.x;
	
	  // we reduce multiple elements per thread.  The number is determined by the
	  // number of active thread blocks (via gridDim). More blocks will result
	  // in a larger gridSize and therefore fewer elements per thread.
	  sdata_t[tid] = identity;
	
	  while (i < n) {
	    sdata_t[tid] = func(sdata_t[tid], g_idata[i]);
	    i += gridSize;
	  }
	  __syncthreads();
	
	  // perform reduction in shared memory
	  if ((blockSize >= 1024) && (tid < 512)) {
	    sdata_t[tid] = func(sdata_t[tid], sdata_t[tid + 512]);
	  }
	  __syncthreads();
	
	  if ((blockSize >= 512) && (tid < 256)) {
	    sdata_t[tid] = func(sdata_t[tid], sdata_t[tid + 256]);
	  }
	  __syncthreads();
	
	  if ((blockSize >= 256) && (tid < 128)) {
	    sdata_t[tid] = func(sdata_t[tid], sdata_t[tid + 128]);
	  }
	  __syncthreads();
	
	  if ((blockSize >= 128) && (tid < 64)) {
	    sdata_t[tid] = func(sdata_t[tid], sdata_t[tid + 64]);
	  }
	  __syncthreads();
	
	  if ((blockSize >= 64) && (tid < 32)) {
	    sdata_t[tid] = func(sdata_t[tid], sdata_t[tid + 32]);
	  }
	  __syncthreads();
	
	  if ((blockSize >= 32) && (tid < 16)) {
	    sdata_t[tid] = func(sdata_t[tid], sdata_t[tid + 16]);
	  }
	  __syncthreads();
	
	  if ((blockSize >= 16) && (tid < 8)) {
	    sdata_t[tid] = func(sdata_t[tid], sdata_t[tid + 8]);
	  }
	  __syncthreads();
	
	  if ((blockSize >= 8) && (tid < 4)) {
	    sdata_t[tid] = func(sdata_t[tid], sdata_t[tid + 4]);
	  }
	  __syncthreads();
	
	  if ((blockSize >= 4) && (tid < 2)) {
	    sdata_t[tid] = func(sdata_t[tid], sdata_t[tid + 2]);
	  }
	  __syncthreads();
	
	  if ((blockSize >= 2) && (tid < 1)) {
	    sdata_t[tid] = func(sdata_t[tid], sdata_t[tid + 1]);
	  }
	  __syncthreads();
	
	  // write result for this block to global mem
	  if (tid == 0) {
	    g_odata[blockIdx.x] = sdata_t[0];
	  }
	}
template<typename T>
void mkt::kernel::reduce_plus_call(unsigned int size, T* d_idata, T* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu) {
  cudaSetDevice(gpu);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  unsigned int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    switch (threads) {
      case 1024:
        mkt::kernel::reduce_plus<T, 1024> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 512:
        mkt::kernel::reduce_plus<T, 512> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 256:
        mkt::kernel::reduce_plus<T, 256> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 128:
        mkt::kernel::reduce_plus<T, 128> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 64:
        mkt::kernel::reduce_plus<T, 64> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 32:
        mkt::kernel::reduce_plus<T, 32> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 16:
        mkt::kernel::reduce_plus<T, 16> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 8:
        mkt::kernel::reduce_plus<T, 8> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 4:
        mkt::kernel::reduce_plus<T, 4> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 2:
        mkt::kernel::reduce_plus<T, 2> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 1:
        mkt::kernel::reduce_plus<T, 1> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
    }
}
template<typename T>
void mkt::kernel::reduce_multiply_call(unsigned int size, T* d_idata, T* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu) {
  cudaSetDevice(gpu);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  unsigned int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    switch (threads) {
      case 1024:
        mkt::kernel::reduce_multiply<T, 1024> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 512:
        mkt::kernel::reduce_multiply<T, 512> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 256:
        mkt::kernel::reduce_multiply<T, 256> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 128:
        mkt::kernel::reduce_multiply<T, 128> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 64:
        mkt::kernel::reduce_multiply<T, 64> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 32:
        mkt::kernel::reduce_multiply<T, 32> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 16:
        mkt::kernel::reduce_multiply<T, 16> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 8:
        mkt::kernel::reduce_multiply<T, 8> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 4:
        mkt::kernel::reduce_multiply<T, 4> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 2:
        mkt::kernel::reduce_multiply<T, 2> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 1:
        mkt::kernel::reduce_multiply<T, 1> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
    }
}
void mkt::kernel::reduce_min_call(unsigned int size, int* d_idata, int* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu) {
  cudaSetDevice(gpu);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  unsigned int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

    switch (threads) {
      case 1024:
        mkt::kernel::reduce_min<1024> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 512:
        mkt::kernel::reduce_min<512> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 256:
        mkt::kernel::reduce_min<256> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 128:
        mkt::kernel::reduce_min<128> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 64:
        mkt::kernel::reduce_min<64> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 32:
        mkt::kernel::reduce_min<32> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 16:
        mkt::kernel::reduce_min<16> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 8:
        mkt::kernel::reduce_min<8> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 4:
        mkt::kernel::reduce_min<4> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 2:
        mkt::kernel::reduce_min<2> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 1:
        mkt::kernel::reduce_min<1> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
    }
}
void mkt::kernel::reduce_min_call(unsigned int size, float* d_idata, float* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu) {
  cudaSetDevice(gpu);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  unsigned int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    switch (threads) {
      case 1024:
        mkt::kernel::reduce_min<1024> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 512:
        mkt::kernel::reduce_min<512> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 256:
        mkt::kernel::reduce_min<256> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 128:
        mkt::kernel::reduce_min<128> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 64:
        mkt::kernel::reduce_min<64> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 32:
        mkt::kernel::reduce_min<32> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 16:
        mkt::kernel::reduce_min<16> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 8:
        mkt::kernel::reduce_min<8> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 4:
        mkt::kernel::reduce_min<4> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 2:
        mkt::kernel::reduce_min<2> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 1:
        mkt::kernel::reduce_min<1> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
    }
}
void mkt::kernel::reduce_min_call(unsigned int size, double* d_idata, double* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu) {
  cudaSetDevice(gpu);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  unsigned int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

    switch (threads) {
      case 1024:
        mkt::kernel::reduce_min<1024> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 512:
        mkt::kernel::reduce_min<512> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 256:
        mkt::kernel::reduce_min<256> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 128:
        mkt::kernel::reduce_min<128> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 64:
        mkt::kernel::reduce_min<64> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 32:
        mkt::kernel::reduce_min<32> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 16:
        mkt::kernel::reduce_min<16> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 8:
        mkt::kernel::reduce_min<8> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 4:
        mkt::kernel::reduce_min<4> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 2:
        mkt::kernel::reduce_min<2> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 1:
        mkt::kernel::reduce_min<1> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
    }
}
void mkt::kernel::reduce_max_call(unsigned int size, int* d_idata, int* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu) {
  cudaSetDevice(gpu);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  unsigned int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

    switch (threads) {
      case 1024:
        mkt::kernel::reduce_max<1024> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 512:
        mkt::kernel::reduce_max<512> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 256:
        mkt::kernel::reduce_max<256> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 128:
        mkt::kernel::reduce_max<128> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 64:
        mkt::kernel::reduce_max<64> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 32:
        mkt::kernel::reduce_max<32> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 16:
        mkt::kernel::reduce_max<16> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 8:
        mkt::kernel::reduce_max<8> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 4:
        mkt::kernel::reduce_max<4> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 2:
        mkt::kernel::reduce_max<2> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 1:
        mkt::kernel::reduce_max<1> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
    }
}
void mkt::kernel::reduce_max_call(unsigned int size, float* d_idata, float* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu) {
  cudaSetDevice(gpu);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  unsigned int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    switch (threads) {
      case 1024:
        mkt::kernel::reduce_max<1024> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 512:
        mkt::kernel::reduce_max<512> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 256:
        mkt::kernel::reduce_max<256> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 128:
        mkt::kernel::reduce_max<128> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 64:
        mkt::kernel::reduce_max<64> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 32:
        mkt::kernel::reduce_max<32> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 16:
        mkt::kernel::reduce_max<16> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 8:
        mkt::kernel::reduce_max<8> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 4:
        mkt::kernel::reduce_max<4> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 2:
        mkt::kernel::reduce_max<2> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 1:
        mkt::kernel::reduce_max<1> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
    }
}
void mkt::kernel::reduce_max_call(unsigned int size, double* d_idata, double* d_odata, int threads, int blocks, cudaStream_t& stream, int gpu) {
  cudaSetDevice(gpu);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  unsigned int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

    switch (threads) {
      case 1024:
        mkt::kernel::reduce_max<1024> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 512:
        mkt::kernel::reduce_max<512> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 256:
        mkt::kernel::reduce_max<256> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 128:
        mkt::kernel::reduce_max<128> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 64:
        mkt::kernel::reduce_max<64> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 32:
        mkt::kernel::reduce_max<32> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 16:
        mkt::kernel::reduce_max<16> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 8:
        mkt::kernel::reduce_max<8> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 4:
        mkt::kernel::reduce_max<4> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 2:
        mkt::kernel::reduce_max<2> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
      case 1:
        mkt::kernel::reduce_max<1> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
        break;
    }
}

template<typename T, typename R, typename Functor>
void mkt::kernel::map_reduce_plus_call(unsigned int size, T* d_idata, R* d_odata, int threads, int blocks, Functor f, cudaStream_t& stream, int gpu) {
  cudaSetDevice(gpu);
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  unsigned int smemSize = (threads <= 32) ? 2 * threads * sizeof(R) : threads * sizeof(R);

    switch (threads) {
      case 1024:
        mkt::kernel::map_reduce_plus<T, R,1024, Functor> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, f);
        break;
      case 512:
        mkt::kernel::map_reduce_plus<T, R,512, Functor> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, f);
        break;
      case 256:
        mkt::kernel::map_reduce_plus<T, R,256, Functor> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, f);
        break;
      case 128:
        mkt::kernel::map_reduce_plus<T, R,128, Functor> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, f);
        break;
      case 64:
        mkt::kernel::map_reduce_plus<T, R,64, Functor> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, f);
        break;
      case 32:
        mkt::kernel::map_reduce_plus<T, R,32, Functor> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, f);
        break;
      case 16:
        mkt::kernel::map_reduce_plus<T, R,16, Functor> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, f);
        break;
      case 8:
        mkt::kernel::map_reduce_plus<T, R,8, Functor> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, f);
        break;
      case 4:
        mkt::kernel::map_reduce_plus<T, R,4, Functor> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, f);
        break;
      case 2:
        mkt::kernel::map_reduce_plus<T, R,2, Functor> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, f);
        break;
      case 1:
        mkt::kernel::map_reduce_plus<T, R,1, Functor> <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, f);
        break;
    }
}

    
template<typename T, unsigned int blockSize>
__global__ void mkt::kernel::reduce_plus(T *g_idata, T *g_odata, unsigned int n) {
  extern __shared__ T sdata_t[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  sdata_t[tid] = static_cast<T>(0);

  while (i < n) {
    sdata_t[tid] += g_idata[i];
    i += gridSize;
  }
  __syncthreads();

  // perform reduction in shared memory
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata_t[tid] += sdata_t[tid + 512];
  }
  __syncthreads();

  if ((blockSize >= 512) && (tid < 256)) {
    sdata_t[tid] += sdata_t[tid + 256];
  }
  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata_t[tid] += sdata_t[tid + 128];
  }
  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata_t[tid] += sdata_t[tid + 64];
  }
  __syncthreads();

  if ((blockSize >= 64) && (tid < 32)) {
    sdata_t[tid] += sdata_t[tid + 32];
  }
  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata_t[tid] += sdata_t[tid + 16];
  }
  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata_t[tid] += sdata_t[tid + 8];
  }
  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata_t[tid] += sdata_t[tid + 4];
  }
  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata_t[tid] += sdata_t[tid + 2];
  }
  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata_t[tid] += sdata_t[tid + 1];
  }
  __syncthreads();

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata_t[0];
  }
}

template<typename T, unsigned int blockSize>
__global__ void mkt::kernel::reduce_multiply(T *g_idata, T *g_odata, unsigned int n) {
  extern __shared__ T sdata_t[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  sdata_t[tid] = static_cast<T>(1);

  while (i < n) {
    sdata_t[tid] *= g_idata[i];
    i += gridSize;
  }
  __syncthreads();

  // perform reduction in shared memory
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata_t[tid] *= sdata_t[tid + 512];
  }
  __syncthreads();

  if ((blockSize >= 512) && (tid < 256)) {
    sdata_t[tid] *= sdata_t[tid + 256];
  }
  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata_t[tid] *= sdata_t[tid + 128];
  }
  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata_t[tid] *= sdata_t[tid + 64];
  }
  __syncthreads();

  if ((blockSize >= 64) && (tid < 32)) {
    sdata_t[tid] *= sdata_t[tid + 32];
  }
  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata_t[tid] *= sdata_t[tid + 16];
  }
  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata_t[tid] *= sdata_t[tid + 8];
  }
  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata_t[tid] *= sdata_t[tid + 4];
  }
  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata_t[tid] *= sdata_t[tid + 2];
  }
  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata_t[tid] *= sdata_t[tid + 1];
  }
  __syncthreads();

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata_t[0];
  }
}

template<unsigned int blockSize>
__global__ void mkt::kernel::reduce_max(int *g_idata, int *g_odata, unsigned int n) {
  extern __shared__ int sdata_int[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  sdata_int[tid] = INT_MIN;

  while (i < n) {
    sdata_int[tid] = max(sdata_int[tid], g_idata[i]);
    i += gridSize;
  }
  __syncthreads();

  // perform reduction in shared memory
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata_int[tid] = max(sdata_int[tid], sdata_int[tid + 512]);
  }
  __syncthreads();

  if ((blockSize >= 512) && (tid < 256)) {
    sdata_int[tid] = max(sdata_int[tid], sdata_int[tid + 256]);
  }
  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata_int[tid] = max(sdata_int[tid], sdata_int[tid + 128]);
  }
  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata_int[tid] = max(sdata_int[tid], sdata_int[tid + 64]);
  }
  __syncthreads();

  if ((blockSize >= 64) && (tid < 32)) {
    sdata_int[tid] = max(sdata_int[tid], sdata_int[tid + 32]);
  }
  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata_int[tid] = max(sdata_int[tid], sdata_int[tid + 16]);
  }
  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata_int[tid] = max(sdata_int[tid], sdata_int[tid + 8]);
  }
  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata_int[tid] = max(sdata_int[tid], sdata_int[tid + 4]);
  }
  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata_int[tid] = max(sdata_int[tid], sdata_int[tid + 2]);
  }
  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata_int[tid] = max(sdata_int[tid], sdata_int[tid + 1]);
  }
  __syncthreads();

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata_int[0];
  }
}
template<unsigned int blockSize>
__global__ void mkt::kernel::reduce_max(float *g_idata, float *g_odata, unsigned int n) {
  extern __shared__ float sdata_float[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  sdata_float[tid] = FLT_MIN;

  while (i < n) {
    sdata_float[tid] = fmaxf(sdata_float[tid], g_idata[i]);
    i += gridSize;
  }
  __syncthreads();

  // perform reduction in shared memory
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata_float[tid] = fmaxf(sdata_float[tid], sdata_float[tid + 512]);
  }
  __syncthreads();

  if ((blockSize >= 512) && (tid < 256)) {
    sdata_float[tid] = fmaxf(sdata_float[tid], sdata_float[tid + 256]);
  }
  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata_float[tid] = fmaxf(sdata_float[tid], sdata_float[tid + 128]);
  }
  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata_float[tid] = fmaxf(sdata_float[tid], sdata_float[tid + 64]);
  }
  __syncthreads();

  if ((blockSize >= 64) && (tid < 32)) {
    sdata_float[tid] = fmaxf(sdata_float[tid], sdata_float[tid + 32]);
  }
  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata_float[tid] = fmaxf(sdata_float[tid], sdata_float[tid + 16]);
  }
  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata_float[tid] = fmaxf(sdata_float[tid], sdata_float[tid + 8]);
  }
  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata_float[tid] = fmaxf(sdata_float[tid], sdata_float[tid + 4]);
  }
  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata_float[tid] = fmaxf(sdata_float[tid], sdata_float[tid + 2]);
  }
  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata_float[tid] = fmaxf(sdata_float[tid], sdata_float[tid + 1]);
  }
  __syncthreads();

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata_float[0];
  }
}
template<unsigned int blockSize>
__global__ void mkt::kernel::reduce_max(double *g_idata, double *g_odata, unsigned int n) {
  extern __shared__ double sdata_double[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  sdata_double[tid] = DBL_MIN;

  while (i < n) {
    sdata_double[tid] = fmax(sdata_double[tid], g_idata[i]);
    i += gridSize;
  }
  __syncthreads();

  // perform reduction in shared memory
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata_double[tid] = fmax(sdata_double[tid], sdata_double[tid + 512]);
  }
  __syncthreads();

  if ((blockSize >= 512) && (tid < 256)) {
    sdata_double[tid] = fmax(sdata_double[tid], sdata_double[tid + 256]);
  }
  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata_double[tid] = fmax(sdata_double[tid], sdata_double[tid + 128]);
  }
  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata_double[tid] = fmax(sdata_double[tid], sdata_double[tid + 64]);
  }
  __syncthreads();

  if ((blockSize >= 64) && (tid < 32)) {
    sdata_double[tid] = fmax(sdata_double[tid], sdata_double[tid + 32]);
  }
  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata_double[tid] = fmax(sdata_double[tid], sdata_double[tid + 16]);
  }
  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata_double[tid] = fmax(sdata_double[tid], sdata_double[tid + 8]);
  }
  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata_double[tid] = fmax(sdata_double[tid], sdata_double[tid + 4]);
  }
  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata_double[tid] = fmax(sdata_double[tid], sdata_double[tid + 2]);
  }
  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata_double[tid] = fmax(sdata_double[tid], sdata_double[tid + 1]);
  }
  __syncthreads();

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata_double[0];
  }
}

template<unsigned int blockSize>
__global__ void mkt::kernel::reduce_min(int *g_idata, int *g_odata, unsigned int n) {
  extern __shared__ int sdata_int[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  sdata_int[tid] = INT_MAX;

  while (i < n) {
    sdata_int[tid] = min(sdata_int[tid], g_idata[i]);
    i += gridSize;
  }
  __syncthreads();

  // perform reduction in shared memory
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata_int[tid] = min(sdata_int[tid], sdata_int[tid + 512]);
  }
  __syncthreads();

  if ((blockSize >= 512) && (tid < 256)) {
    sdata_int[tid] = min(sdata_int[tid], sdata_int[tid + 256]);
  }
  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata_int[tid] = min(sdata_int[tid], sdata_int[tid + 128]);
  }
  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata_int[tid] = min(sdata_int[tid], sdata_int[tid + 64]);
  }
  __syncthreads();

  if ((blockSize >= 64) && (tid < 32)) {
    sdata_int[tid] = min(sdata_int[tid], sdata_int[tid + 32]);
  }
  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata_int[tid] = min(sdata_int[tid], sdata_int[tid + 16]);
  }
  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata_int[tid] = min(sdata_int[tid], sdata_int[tid + 8]);
  }
  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata_int[tid] = min(sdata_int[tid], sdata_int[tid + 4]);
  }
  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata_int[tid] = min(sdata_int[tid], sdata_int[tid + 2]);
  }
  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata_int[tid] = min(sdata_int[tid], sdata_int[tid + 1]);
  }
  __syncthreads();

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata_int[0];
  }
}
template<unsigned int blockSize>
__global__ void mkt::kernel::reduce_min(float *g_idata, float *g_odata, unsigned int n) {
  extern __shared__ float sdata_float[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  sdata_float[tid] = FLT_MAX;

  while (i < n) {
    sdata_float[tid] = fminf(sdata_float[tid], g_idata[i]);
    i += gridSize;
  }
  __syncthreads();

  // perform reduction in shared memory
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata_float[tid] = fminf(sdata_float[tid], sdata_float[tid + 512]);
  }
  __syncthreads();

  if ((blockSize >= 512) && (tid < 256)) {
    sdata_float[tid] = fminf(sdata_float[tid], sdata_float[tid + 256]);
  }
  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata_float[tid] = fminf(sdata_float[tid], sdata_float[tid + 128]);
  }
  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata_float[tid] = fminf(sdata_float[tid], sdata_float[tid + 64]);
  }
  __syncthreads();

  if ((blockSize >= 64) && (tid < 32)) {
    sdata_float[tid] = fminf(sdata_float[tid], sdata_float[tid + 32]);
  }
  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata_float[tid] = fminf(sdata_float[tid], sdata_float[tid + 16]);
  }
  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata_float[tid] = fminf(sdata_float[tid], sdata_float[tid + 8]);
  }
  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata_float[tid] = fminf(sdata_float[tid], sdata_float[tid + 4]);
  }
  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata_float[tid] = fminf(sdata_float[tid], sdata_float[tid + 2]);
  }
  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata_float[tid] = fminf(sdata_float[tid], sdata_float[tid + 1]);
  }
  __syncthreads();

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata_float[0];
  }
}
template<unsigned int blockSize>
__global__ void mkt::kernel::reduce_min(double *g_idata, double *g_odata, unsigned int n) {
  extern __shared__ double sdata_double[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  sdata_double[tid] = DBL_MAX;

  while (i < n) {
    sdata_double[tid] = fmin(sdata_double[tid], g_idata[i]);
    i += gridSize;
  }
  __syncthreads();

  // perform reduction in shared memory
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata_double[tid] = fmin(sdata_double[tid], sdata_double[tid + 512]);
  }
  __syncthreads();

  if ((blockSize >= 512) && (tid < 256)) {
    sdata_double[tid] = fmin(sdata_double[tid], sdata_double[tid + 256]);
  }
  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata_double[tid] = fmin(sdata_double[tid], sdata_double[tid + 128]);
  }
  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata_double[tid] = fmin(sdata_double[tid], sdata_double[tid + 64]);
  }
  __syncthreads();

  if ((blockSize >= 64) && (tid < 32)) {
    sdata_double[tid] = fmin(sdata_double[tid], sdata_double[tid + 32]);
  }
  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata_double[tid] = fmin(sdata_double[tid], sdata_double[tid + 16]);
  }
  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata_double[tid] = fmin(sdata_double[tid], sdata_double[tid + 8]);
  }
  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata_double[tid] = fmin(sdata_double[tid], sdata_double[tid + 4]);
  }
  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata_double[tid] = fmin(sdata_double[tid], sdata_double[tid + 2]);
  }
  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata_double[tid] = fmin(sdata_double[tid], sdata_double[tid + 1]);
  }
  __syncthreads();

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata_double[0];
  }
}

//// MapReduce
template<typename T, typename R, unsigned int blockSize, typename Functor>
__global__ void mkt::kernel::map_reduce_plus(T *g_idata, R *g_odata, unsigned int n, Functor f) {
  extern __shared__ R sdata_t[];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize + threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim). More blocks will result
  // in a larger gridSize and therefore fewer elements per thread.
  sdata_t[tid] = static_cast<R>(0);

  while (i < n) {
    sdata_t[tid] += f(g_idata[i]);
    i += gridSize;
  }
  __syncthreads();

  // perform reduction in shared memory
  if ((blockSize >= 1024) && (tid < 512)) {
    sdata_t[tid] += sdata_t[tid + 512];
  }
  __syncthreads();

  if ((blockSize >= 512) && (tid < 256)) {
    sdata_t[tid] += sdata_t[tid + 256];
  }
  __syncthreads();

  if ((blockSize >= 256) && (tid < 128)) {
    sdata_t[tid] += sdata_t[tid + 128];
  }
  __syncthreads();

  if ((blockSize >= 128) && (tid < 64)) {
    sdata_t[tid] += sdata_t[tid + 64];
  }
  __syncthreads();

  if ((blockSize >= 64) && (tid < 32)) {
    sdata_t[tid] += sdata_t[tid + 32];
  }
  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sdata_t[tid] += sdata_t[tid + 16];
  }
  __syncthreads();

  if ((blockSize >= 16) && (tid < 8)) {
    sdata_t[tid] += sdata_t[tid + 8];
  }
  __syncthreads();

  if ((blockSize >= 8) && (tid < 4)) {
    sdata_t[tid] += sdata_t[tid + 4];
  }
  __syncthreads();

  if ((blockSize >= 4) && (tid < 2)) {
    sdata_t[tid] += sdata_t[tid + 2];
  }
  __syncthreads();

  if ((blockSize >= 2) && (tid < 1)) {
    sdata_t[tid] += sdata_t[tid + 1];
  }
  __syncthreads();

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata_t[0];
  }
}
