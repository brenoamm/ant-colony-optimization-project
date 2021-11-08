#pragma once
#include <string>
#include "aco_iroulette.cuh"
#include "kernel.cuh"

namespace mkt {
enum Distribution {DIST, COPY};

// Musket variables
cudaStream_t cuda_streams[4];

// Musket functions
void init();
void sync_streams();

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template<typename T>
class DArray {
 public:

  // CONSTRUCTORS / DESTRUCTOR
  DArray(int pid, size_t size, size_t size_local, T init_value, int partitions, int partition_pos, size_t offset, mkt::Distribution d = DIST, mkt::Distribution device_dist = DIST);
  ~ DArray();

  template<std::size_t N>
  void operator=(const std::array<T, N>& a);

   void update_self();
   void update_devices();

// Getter and Setter

  T get_global(size_t index);
  void set_global(size_t index, const T& value);

  T get_local(size_t index);
  void set_local(size_t index, const T& value);

  T get_host_local(size_t index);
  void set_host_local(size_t index, T v);

  T& operator[](size_t local_index);
  const T& operator[](size_t local_index) const;

  size_t get_size() const;
  size_t get_size_local() const;
  size_t get_size_gpu() const;
  size_t get_bytes_gpu() const;

  size_t get_offset() const;

  mkt::Distribution get_distribution() const;
  mkt::Distribution get_device_distribution() const;

  T* get_data();
  const T* get_data() const;

  T* get_device_pointer(int gpu) const;

 private:

  int get_gpu_by_local_index(size_t local_index) const;
  int get_gpu_by_global_index(size_t global_index) const;

  int get_pid_by_global_index(size_t global_index) const;
  bool is_local(size_t global_index) const;

  //
  // Attributes
  //

  // position of processor in data parallel group of processors; zero-base
  int _pid;

  size_t _size;
  size_t _size_local;
  size_t _size_gpu;
  size_t _bytes_gpu;

  // number of (local) partitions in array
  int _partitions;

  // position of processor in data parallel group of processors
  int _partition_pos;

  // first index in local partition
  size_t _offset;

  // checks whether data is copy distributed among all processes
  mkt::Distribution _dist;
  mkt::Distribution _device_dist;

  T* _data;
  std::array<T*, 1> _host_data;
  std::array<T*, 1> _gpu_data;
};
template<typename T, typename R, typename Functor>
void map(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f);

template<typename T, typename R, typename Functor>
void map_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f);

template<typename T, typename R, typename Functor>
void map_local_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f);

template<typename T, typename Functor>
void map_in_place(mkt::DArray<T>& a, Functor f);

template<typename T, typename Functor>
void map_index_in_place(mkt::DArray<T>& a, Functor f);

template<typename T, typename Functor>
void map_index_in_place64(mkt::DArray<T>& a, Functor f);

template<typename T, typename Functor>
void map_local_index_in_place(mkt::DArray<T>& a, Functor f);

template<typename T, typename Functor>
void fold(const mkt::DArray<T>& a, T& out, const T identity, const Functor f);

template<typename T, typename Functor>
void fold_copy(const mkt::DArray<T>& a, T& out, const T identity, const Functor f);

template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
void map_fold(const mkt::DArray<T>& a, R& out, const MapFunctor& f_map, const R identity, const FoldFunctor f_fold);

template<typename T, typename R, typename MapFunctor, typename FoldFunctor>
void map_fold_copy(const mkt::DArray<T>& a, R& out, const MapFunctor& f_map, const R identity, const FoldFunctor f_fold);

template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
void map_fold(const mkt::DArray<T>& a, mkt::DArray<R>& out, const MapFunctor& f_map, const I identity, const FoldFunctor f_fold);

template<typename T, typename R, typename I, typename MapFunctor, typename FoldFunctor>
void map_fold_copy(const mkt::DArray<T>& a, mkt::DArray<R>& out, const MapFunctor f_map, const I identity, const FoldFunctor f_fold);
template<typename T>
class DeviceArray {
 public:

  // CONSTRUCTORS / DESTRUCTOR
  DeviceArray(const DArray<T>& da);
  DeviceArray(const DeviceArray<T>& da);
  ~DeviceArray();

  void init(int device_id);


// Getter and Setter
	size_t get_bytes_device() const;

__device__ const T& get_data_device(size_t device_index) const;
__device__ const T& get_data_local(size_t local_index) const;
__device__ T get_global(size_t local_index);
__device__ T set_global(size_t local_index, T value);

 private:

  //
  // Attributes
  //

  size_t _size;
  size_t _size_local;
  size_t _size_device;
  size_t _bytes_device;

  size_t _offset;

  size_t _device_offset;

  mkt::Distribution _dist;
  mkt::Distribution _device_dist;

  T* _device_data;

  std::array<T*, 1> _gpu_data;

  };



template<typename T>
void print(std::ostringstream& stream, const T& a);


template<typename T>
void gather(mkt::DArray<T>& in, mkt::DArray<T>& out);

template<typename T>
void scatter(mkt::DArray<T>& in, mkt::DArray<T>& out);



template<typename T>
T reduce_plus(mkt::DArray<T>& a);

template<typename T>
T reduce_multiply(mkt::DArray<T>& a);

template<typename T>
T reduce_max(mkt::DArray<T>& a);

template<typename T>
T reduce_min(mkt::DArray<T>& a);



} // namespace mkt


void mkt::init(){

	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		cudaStreamCreate(&cuda_streams[gpu]);
	}
}

void mkt::sync_streams(){
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		cudaStreamSynchronize(cuda_streams[gpu]);
	}
}


	template<typename T>
	mkt::DArray<T>::DArray(int pid, size_t size, size_t size_local, T init_value, int partitions, int partition_pos, size_t offset, mkt::Distribution d, mkt::Distribution device_dist)
	    : _pid(pid),
	      _size(size),
	      _size_local(size_local),
	      _size_gpu(0),
	      _partitions(partitions),
	      _partition_pos(partition_pos),
	      _offset(offset),
	      _dist(d),
	      _device_dist(device_dist) {

	    if(device_dist == mkt::Distribution::DIST){
	    	_size_gpu = size_local / 1; // assume even distribution for now
	    }else if(device_dist == mkt::Distribution::COPY){
	    	_size_gpu = size_local;
	    }
	    _bytes_gpu = _size_gpu * sizeof(T);

	    cudaMallocHost((void**)&_data, _size_local * sizeof(T));

		for(int gpu = 0; gpu < 1; ++gpu){
			cudaSetDevice(gpu);

			// allocate memory
			T* devptr;
			cudaMalloc((void**)&devptr, _size_gpu * sizeof(T));

			// store pointer to device memory and host memory
			_gpu_data[gpu] = devptr;
			if(device_dist == mkt::Distribution::DIST){
		    	_host_data[gpu] = _data + gpu * _size_gpu;
		    }else if(device_dist == mkt::Distribution::COPY){
		    	_host_data[gpu] = _data; // all gpus have complete data, thus point to the beginning of host vector
		    }
		}

		//init data
		for(size_t i = 0; i< _size_local; ++i){
			_data[i] = init_value;
		}
		update_devices();
	}

	template<typename T>
	mkt::DArray<T>::~DArray(){
		cudaFreeHost(_data);
		for(int gpu = 0; gpu < 1; ++gpu){
			cudaSetDevice(gpu);
			cudaFree(_gpu_data[gpu]);
		}
	}

	template<typename T>
	template<std::size_t N>
	void mkt::DArray<T>::operator=(const std::array<T, N>& a) {
	  mkt::sync_streams();
	  for(size_t element = 0; element < _size_local; ++element){
		_data[element] = a[element];
	  }
	  update_devices();
	}

	template<typename T>
	void mkt::DArray<T>::update_self() {
		if(_device_dist == mkt::Distribution::DIST){
			for(int gpu = 0; gpu < 1; ++gpu){
				cudaSetDevice(gpu);
				cudaMemcpyAsync(_host_data[gpu], _gpu_data[gpu], _bytes_gpu, cudaMemcpyDeviceToHost, mkt::cuda_streams[gpu]);
			}
		}else{
			cudaSetDevice(0);
			cudaMemcpyAsync(_host_data[0], _gpu_data[0], _bytes_gpu, cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
            //gpuErrchk( cudaPeekAtLastError() );
            //gpuErrchk( cudaDeviceSynchronize() );
		}
		mkt::sync_streams();
	}

	template<typename T>
	void mkt::DArray<T>::update_devices() {
        if(_device_dist == mkt::Distribution::DIST){
            for(int gpu = 0; gpu < 1; ++gpu){
                cudaSetDevice(gpu);
                cudaMemcpyAsync(_gpu_data[gpu], _host_data[gpu], _bytes_gpu, cudaMemcpyHostToDevice, mkt::cuda_streams[gpu]);
            }
            mkt::sync_streams();

        }else{
            cudaSetDevice(0);
            cudaMemcpyAsync(_gpu_data[0], _host_data[0], _bytes_gpu, cudaMemcpyHostToDevice, mkt::cuda_streams[0]);
        }
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
	}

	template<typename T>
	T mkt::DArray<T>::get_local(size_t index) {
		// One GPU is configured if the datastructure is up-to-date it can be returned.
		T* host_pointer = _data + index;
		T* gpu_pointer = _gpu_data[0] + (index % _size_gpu );
		cudaMemcpyAsync(host_pointer, gpu_pointer, sizeof(T), cudaMemcpyDeviceToHost, mkt::cuda_streams[0]);
		mkt::sync_streams();
		return _data[index];
	}

	template<typename T>
	void mkt::DArray<T>::set_local(size_t index, const T& v) {
		mkt::sync_streams();
		_data[index] = v;
		T* host_pointer = _data + index;
		if(_device_dist == mkt::Distribution::COPY){
			for(int gpu = 0; gpu < 1; ++gpu){
				cudaSetDevice(gpu);
				T* gpu_pointer = _gpu_data[gpu] + index;
				cudaMemcpyAsync(gpu_pointer, host_pointer, sizeof(T), cudaMemcpyHostToDevice, mkt::cuda_streams[gpu]);
			}
		}else if(_device_dist == mkt::Distribution::DIST){
			int gpu = get_gpu_by_local_index(index);
			cudaSetDevice(gpu);
			T* gpu_pointer = _gpu_data[gpu] + (index % _size_gpu );
			cudaMemcpyAsync(gpu_pointer, host_pointer, sizeof(T), cudaMemcpyHostToDevice, mkt::cuda_streams[gpu]);
		}
	}

	template<typename T>
	T mkt::DArray<T>::get_host_local(size_t index) {
		return _data[index];
	}

	template<typename T>
	void mkt::DArray<T>::set_host_local(size_t index, T v) {
		_data[index] = v;
	}

	template<typename T>
	int mkt::DArray<T>::get_pid_by_global_index(size_t global_index) const {
	  return global_index / _size_local;
	}

	template<typename T>
	bool mkt::DArray<T>::is_local(size_t global_index) const {
		int pid = get_pid_by_global_index(global_index);
	  return (pid == _pid);
	}

	template<typename T>
	T mkt::DArray<T>::get_global(size_t index) {
		// One GPU is configured if the datastructure is up-to-date it can be returned.
		return get_local(index);
	}

	template<typename T>
	void mkt::DArray<T>::set_global(size_t index, const T& v) {
	  // TODO
	}

	template<typename T>
	T& mkt::DArray<T>::operator[](size_t local_index) {
	  	return _data[local_index];
	}

	template<typename T>
	const T& mkt::DArray<T>::operator[](size_t local_index) const {
	  	return _data[local_index];
	}

	template<typename T>
	size_t mkt::DArray<T>::get_size() const {
	  return _size;
	}

	template<typename T>
	size_t mkt::DArray<T>::get_size_local() const {
	  return _size_local;
	}

	template<typename T>
	size_t mkt::DArray<T>::get_size_gpu() const {
	  return _size_gpu;
	}

	template<typename T>
	size_t mkt::DArray<T>::get_bytes_gpu() const {
	  return _bytes_gpu;
	}

	template<typename T>
	size_t mkt::DArray<T>::get_offset() const {
	  return _offset;
	}

	template<typename T>
	mkt::Distribution mkt::DArray<T>::get_distribution() const {
	  return _dist;
	}

	template<typename T>
	mkt::Distribution mkt::DArray<T>::get_device_distribution() const {
	  return _device_dist;
	}

	template<typename T>
	const T* mkt::DArray<T>::get_data() const {
	  return _data;
	}

	template<typename T>
	T* mkt::DArray<T>::get_data() {
	  return _data;
	}

	template<typename T>
	T* mkt::DArray<T>::get_device_pointer(int gpu) const{
	  return _gpu_data[gpu];
	}

	template<typename T>
	int mkt::DArray<T>::get_gpu_by_local_index(size_t local_index) const {
		if(_device_dist == mkt::Distribution::COPY){
			return 0;
		}else if(_device_dist == mkt::Distribution::DIST){
			return local_index / _size_gpu;
		}
		else{
			return -1;
		}
	}

	template<typename T>
	int mkt::DArray<T>::get_gpu_by_global_index(size_t global_index) const {
		// TODO
		return -1;
	}
template<typename T, typename R, typename Functor>
void mkt::map(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f) {
	size_t gpu_elements = in.get_size_gpu();

	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);

		size_t smem_bytes = f.get_smem_bytes();

		dim3 dimBlock(1024);
		dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
		mkt::kernel::map<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(in_devptr, out_devptr, gpu_elements, f);
	}
}

template<typename T, typename R, typename Functor>
void mkt::map_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f) {
	size_t offset = in.get_offset();
	size_t gpu_elements = in.get_size_gpu();

	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);

		size_t gpu_offset = offset;
		if(in.get_device_distribution() == mkt::Distribution::DIST){
			gpu_offset += gpu * gpu_elements;
		}

		size_t smem_bytes = f.get_smem_bytes();

		dim3 dimBlock(1024);
		dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
		mkt::kernel::map_index<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(in_devptr, out_devptr, gpu_elements, gpu_offset, f);
	}
}

template<typename T, typename R, typename Functor>
void mkt::map_local_index(const mkt::DArray<T>& in, mkt::DArray<R>& out, Functor f) {
	size_t gpu_elements = in.get_size_gpu();

	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* in_devptr = in.get_device_pointer(gpu);
		R* out_devptr = out.get_device_pointer(gpu);

		size_t gpu_offset = 0;
		if(in.get_device_distribution() == mkt::Distribution::DIST){
			gpu_offset += gpu * gpu_elements;
		}

		size_t smem_bytes = f.get_smem_bytes();

		dim3 dimBlock(1024);
		dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
		mkt::kernel::map_index<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(in_devptr, out_devptr, gpu_elements, gpu_offset, f);
	}
}

template<typename T, typename Functor>
void mkt::map_in_place(mkt::DArray<T>& a, Functor f){
	size_t gpu_elements = a.get_size_gpu();

	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* devptr = a.get_device_pointer(gpu);

		size_t smem_bytes = f.get_smem_bytes();

		dim3 dimBlock(1024);
		dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
		mkt::kernel::map_in_place<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(devptr, gpu_elements, f);
	}
}
template<typename T, typename Functor>
void mkt::map_index_in_place(mkt::DArray<T>& a, Functor f){
	size_t offset = a.get_offset();
	size_t gpu_elements = a.get_size_gpu();
	//printf("%zu, %zu\n", offset, gpu_elements);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* devptr = a.get_device_pointer(gpu);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
		size_t gpu_offset = offset;
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			gpu_offset += gpu * gpu_elements;
		}
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
		size_t smem_bytes = f.get_smem_bytes();

		dim3 dimBlock(1024);
		dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
		mkt::kernel::map_index_in_place<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(devptr, gpu_elements, gpu_offset, f);
        //printf("count\n");
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
	}
}

template<typename T, typename Functor>
void mkt::map_index_in_place64(mkt::DArray<T>& a, Functor f){
	size_t offset = a.get_offset();
	size_t gpu_elements = a.get_size_gpu();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* devptr = a.get_device_pointer(gpu);

		size_t gpu_offset = offset;
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			gpu_offset += gpu * gpu_elements;
		}

		size_t smem_bytes = f.get_smem_bytes();

		dim3 dimBlock(64);
		dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
		mkt::kernel::map_index_in_place<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(devptr, gpu_elements, gpu_offset, f);
		//printf("%d", (gpu_elements+dimBlock.x-1)/dimBlock.x);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
	}
}

template<typename T, typename Functor>
void mkt::map_local_index_in_place(mkt::DArray<T>& a, Functor f){
	size_t gpu_elements = a.get_size_gpu();

	for(int gpu = 0; gpu < 1; ++gpu){
		cudaSetDevice(gpu);
		f.init(gpu);
		T* devptr = a.get_device_pointer(gpu);

		size_t gpu_offset = 0;
		if(a.get_device_distribution() == mkt::Distribution::DIST){
			gpu_offset = gpu * gpu_elements;
		}

		size_t smem_bytes = f.get_smem_bytes();

		dim3 dimBlock(1024);
		dim3 dimGrid((gpu_elements+dimBlock.x-1)/dimBlock.x);
		mkt::kernel::map_index_in_place<<<dimGrid, dimBlock, smem_bytes, mkt::cuda_streams[gpu]>>>(devptr, gpu_elements, gpu_offset, f);
	}
}
						template<typename T>
						mkt::DeviceArray<T>::DeviceArray(const DArray<T>& da)
						    : _size(da.get_size()),
						      _size_local(da.get_size_local()),
						      _size_device(da.get_size_gpu()),
						      _bytes_device(da.get_bytes_gpu()),
						      _offset(da.get_offset()),
						      _device_offset(0),
						      _dist(da.get_distribution()),
						      _device_dist(da.get_device_distribution())
						{
							_device_data = nullptr;
							for(int i = 0; i < 1; ++i){
								_gpu_data[i] = da.get_device_pointer(i);
							}
						}

						template<typename T>
						mkt::DeviceArray<T>::DeviceArray(const DeviceArray<T>& da)
						    : _size(da._size),
						      _size_local(da._size_local),
						      _size_device(da._size_device),
						      _bytes_device(da._bytes_device),
						      _offset(da._offset),
						      _device_offset(da._device_offset),
						      _dist(da._dist),
						      _device_dist(da._device_dist)
						{
							_device_data = da._device_data;
							for(int i = 0; i < 1; ++i){
								_gpu_data[i] = da._gpu_data[i];
							}
						}


						template<typename T>
						mkt::DeviceArray<T>::~DeviceArray(){
						}

						template<typename T>
						void mkt::DeviceArray<T>::init(int gpu) {
							if(_device_dist == mkt::Distribution::COPY){
								_device_offset = 0;
							} else {
								_device_offset = _size_device * gpu;
							}

							_device_data = _gpu_data[gpu];
						}

						template<typename T>
						__device__ T mkt::DeviceArray<T>::get_global(size_t index) {
							// One GPU is configured if the datastructure is up-to-date it can be returned.
							return get_data_local(index);
						}


						template<typename T>
						__device__ T mkt::DeviceArray<T>::set_global(size_t local_index, T value) {
						  _device_data[local_index] = value;
						}

						template<typename T>
						size_t mkt::DeviceArray<T>::get_bytes_device() const {
						  return _bytes_device;
						}

						template<typename T>
						__device__ const T& mkt::DeviceArray<T>::get_data_device(size_t device_index) const {
						  return _device_data[device_index];
						}


						template<typename T>
						__device__ const T& mkt::DeviceArray<T>::get_data_local(size_t local_index) const {
						  return this->get_data_device(local_index - _device_offset);
						}



template<typename T>
void mkt::print(std::ostringstream& stream, const T& a) {
	if(std::is_fundamental<T>::value){
		stream << a;
	}
}



template<>
void mkt::gather<double>(mkt::DArray<double>& in, mkt::DArray<double>& out){
	in.update_self();
	std::copy(in.get_data(), in.get_data() + in.get_size_local(), out.get_data());
	out.update_devices();
}
template<>
void mkt::gather<int>(mkt::DArray<int>& in, mkt::DArray<int>& out){
	in.update_self();
	std::copy(in.get_data(), in.get_data() + in.get_size_local(), out.get_data());
	out.update_devices();
}
template<typename T>
void mkt::scatter(mkt::DArray<T>& in, mkt::DArray<T>& out){
	in.update_self();
	std::copy(in.get_data(), in.get_data() + in.get_size(), out.get_data());
	out.update_devices();
}
