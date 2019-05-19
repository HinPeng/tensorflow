/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>

using perftools::gputools::DeviceMemoryBase;
// using perftools::gputools::Stream;

// #define _DEBUG
// #define _DEBUGV2

/* #define cudaCheckError(cudaCall) {                                                  \
    cudaError_t err = cudaCall;                                                       \
    if(err!=cudaSuccess) {                                                            \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
      exit(0);                                                                        \
    }                                                                                 \
} */


namespace tensorflow {

std::string GetEnv(const std::string& env_name) {
  const char* env_p = std::getenv(env_name.c_str());
  if (env_p == nullptr) return "";
  return env_p;
}

const std::string swap_policy_env = "SWAP_POLICY_FILE";

const int64 kCopyThreshold = 2 << 20;    // 2M

/* cudaStream_t GPUBFCAllocator::device_to_device_stream_;
cudaStream_t GPUBFCAllocator::host_to_device_stream_;
cudaStream_t GPUBFCAllocator::device_to_host_stream_;
cudaEvent_t GPUBFCAllocator::cuda_event_; */

GPUBFCAllocator::GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                                 const string& name)
    : GPUBFCAllocator(cuda_gpu_id, total_memory, GPUOptions(), name) {}

GPUBFCAllocator::GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                                 const GPUOptions& gpu_options,
                                 const string& name)
    : BFCAllocator(
          new GPUMemAllocator(
              GpuIdUtil::ExecutorForCudaGpuId(cuda_gpu_id).ValueOrDie()),
          //total_memory, gpu_options.allow_growth(), name) {
          total_memory, true, name) {
      LoadSwapPolicy();
      /* cudaStreamCreate(&device_to_device_stream_);
      cudaStreamCreate(&host_to_device_stream_);
      cudaStreamCreate(&device_to_host_stream_);
      cudaEventCreate(&cuda_event_); */
    }

GPUBFCAllocator::~GPUBFCAllocator() {
  /* const std::string invalid_swap_filename = "/tmp/invalid_swap.txt";
  std::fstream fout(invalid_swap_filename, fout.out);
  if (!fout.is_open()) {
    LOG(ERROR) << "Fail to open invalid swap file";
    return;
  }
  for (auto& name : invalid_swap_) {
    fout << name << "\n";
  } */
}

/*----------------to be deprecated---------------*/
/* void GPUBFCAllocator::GetOrCreateHashBuffer(const Tensor* tensor, const string& tensor_name, HashBuffer** hash_buf){
  // Only record the necessary tensor
  if (tensor_swap_params_map_.count(tensor_name) == 0) return;
  std::lock_guard<std::mutex> l(lock_);

  auto t_buf = tensor->buf_;
  // auto t_name = tensor->Name();  // this field can be empty
  if (t_buf == nullptr) {
    LOG(FATAL) << "Buffer should not be null!";
    return;
  }

  if (hash_bufs_.count(tensor_name) == 0) {
    *hash_buf = new HashBuffer(t_buf, tensor_name);
    hash_bufs_[tensor_name] = *hash_buf;
    return;
  } else {
    *hash_buf = hash_bufs_[tensor_name];
    return;
  }
} */

void GPUBFCAllocator::RecordTensorAccess(const string& tensor_name,
                                         TensorBuffer* buf,
                                         const uint64 _time) {
  if (tensor_swap_params_map_.count(tensor_name)) {
    SwapIn(tensor_name);
  }

  if (swap_triggers_.count(tensor_name) == 0) {
    return;
  }

  auto& swap_trigger = swap_triggers_[tensor_name];
  int cnt;
  {
    std::lock_guard<std::mutex> l(mu_);
    swap_trigger.access_count++;
    cnt = swap_trigger.access_count;
    if (swap_trigger.access_count == swap_trigger.total_access_count) {
      swap_trigger.access_count = 0;
    }
  }

  if (swap_trigger.out_trigger_count != 0) {
    if (cnt == swap_trigger.out_trigger_count) {
      SwapOut(tensor_name);
    } else if (cnt > swap_trigger.out_trigger_count) {
      // when a tensor has been swapped out, we need to set the tensor_buffer->data() to the swapin corresponding memory addr
      DCHECK(tensor_swap_params_map_.count(tensor_name));
      auto& swap_params = tensor_swap_params_map_[tensor_name];
      auto& cv_mu = swap_params.cv_mu;
      {
        std::lock_guard<std::mutex> l(*(cv_mu.second));
        if (swap_params.need_in_addr) {
          void* in_gpu_src = swap_params.in_gpu_src;
          if (in_gpu_src == nullptr) {
            // TODO(px): if this branch happen, seems the in_trigger is totally wrong, can judge according to SwapStatus, if not happen, can leave here
            LOG(FATAL) << "Weird, the correspoding SwapIn in_gpu_src is not set!";
          }
          buf->set_data(in_gpu_src);
          swap_params.need_in_addr = false;
        #ifdef _DEBUGV2
          LOG(INFO) << "Set " << tensor_name << " buffer addr";
        #endif
        }
      }
    }
  }

  if (swap_trigger.in_trigger_count != 0 && cnt == swap_trigger.in_trigger_count) {
    SwapIn(swap_trigger.in_tensor);
  }
}

void GPUBFCAllocator::RecordSwapContext(const TensorParams& params, TensorBuffer* tensor_buf) {
  std::lock_guard<std::mutex> l(lock_);
  if (tensor_swap_params_map_.count(params.name) == 0) return;
  const string &tensor_name = params.name;
  TensorSwapParams& swap_params = tensor_swap_params_map_[tensor_name];
  swap_params.device = params.device;
  swap_params.device_context = params.device_context;
  swap_params.tensor_buffer = tensor_buf;
  swap_params.in_gpu_src = nullptr;
  swap_params.data_ready = SwapStatus::IN;
  swap_params.need_in_addr = false;
  swap_params.can_deallocate_after_swap_out = true;
  swap_params.then_deallocate = false;
  buffer_tensor_map_[tensor_buf] = tensor_name;
}

void GPUBFCAllocator::Notify(TensorBuffer* tensor_buffer) {
  if (buffer_tensor_map_.count(tensor_buffer) == 0) return;
  const string& tensor_name = buffer_tensor_map_[tensor_buffer];
  auto& swap_params = tensor_swap_params_map_[tensor_name];
  auto& cv_mu = swap_params.cv_mu;
  std::lock_guard<std::mutex> l(*(cv_mu.second));
  int64 gpu_part_size = swap_params.swapped_gpu_buffer.second;
  if (swap_params.can_deallocate_after_swap_out && swap_params.then_deallocate) {
    if (gpu_part_size <= 0)
      DeallocateRaw(tensor_buffer->data());
    else
      SplitBuffer(tensor_buffer->data(), gpu_part_size);
    tensor_buffer->set_data(nullptr);
    swap_params.data_ready = SwapStatus::OUT;
    swap_params.then_deallocate = false;
  }

  if (!swap_params.can_deallocate_after_swap_out && swap_params.then_deallocate) {
    LOG(INFO) << "Set status IN for " << swap_params.tensor_name;
    swap_params.data_ready = SwapStatus::IN;
    cv_mu.first->notify_all();
  }
}

// Check inputs again when enqueuing computation
// 1. see if need to deallocate swapped-out tensor's memory and wait for device_to_host_stream (just the computation which trigger swapped-out)
// 2. see if need to wait for host_to_device_stream (the computation which need swapped-out tensor) (set tensor_buf->data() need to be before the enqueuing of computation)
void GPUBFCAllocator::CheckInput(const string& tensor_name,
                                  TensorBuffer* tensor_buf,
                                  bool* flag,
                                  bool before) {
  if (tensor_swap_params_map_.count(tensor_name) == 0) return;

  auto& swap_params = tensor_swap_params_map_[tensor_name];
  auto& cv_mu = swap_params.cv_mu;
  {
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    int ready = swap_params.data_ready;
    if (before) {
      // check tensor iff swapping-in before comp
      if (ready == SwapStatus::SWAPPING_IN) {
        *flag = true;
      #ifdef _DEBUGV2
        LOG(INFO) << tensor_name << " : wait h2d to true";
      #endif
      } else if (ready = SwapStatus::IN) {
      #ifdef _DEBUGV2
        LOG(INFO) << tensor_name << " is already been swapped-in, dont wait h2d";
      #endif
      }
    } else {
      // check tensor iff swapped-out after comp
      if (ready == SwapStatus::SWAPPING_OUT) {
        DeallocateRaw(tensor_buf->data());
        *flag = true;
      #ifdef _DEBUGV2
        LOG(INFO) << tensor_name << " : wait d2h to true";
      #endif
      } else if (ready == SwapStatus::OUT) {
        DeallocateRaw(tensor_buf->data());
      #ifdef _DEBUGV2
        LOG(INFO) << tensor_name << " is already been swapped-out, dont wait d2h";
      #endif
      }
    }
  }
}

void GPUBFCAllocator::LoadSwapPolicy() {
  std::string swap_policy_file = GetEnv(swap_policy_env);
  if (swap_policy_file.empty()) {
  #ifdef _DEBUG
    LOG(INFO) << "No swap policy specified";
  #endif
    return;
  }
  std::fstream fin(swap_policy_file, fin.in);
  if (!fin.is_open()) {
  #ifdef _DEBUG
    LOG(INFO) << "open " << swap_policy_file << " failed.";
  #endif
    return;
  }

  LOG(INFO) << "Load " << swap_policy_file << " succeeded.";

  string out_tensor_name, in_trigger_name;
  int out_trigger_count, in_trigger_count;
  int out_tensor_total_access, in_trigger_total_access;
  while(fin >> out_tensor_name >> out_tensor_total_access >> out_trigger_count
            >> in_trigger_name >> in_trigger_total_access >> in_trigger_count) {
    if (out_tensor_name[0] == '#') {
      continue;
    }
    auto& swap_params = tensor_swap_params_map_[out_tensor_name];
    swap_params.tensor_name = out_tensor_name;
    swap_params.cv_mu = std::make_pair(std::make_shared<std::condition_variable>(), std::make_shared<std::mutex>());
    swap_params.out_fraction = 1.0f;

    auto& swap_out_trigger = swap_triggers_[out_tensor_name];
    swap_out_trigger.tensor_name = out_tensor_name;
    swap_out_trigger.out_trigger_count = out_trigger_count;
    swap_out_trigger.out_params = &swap_params;
    swap_out_trigger.in_trigger = in_trigger_name;
    swap_out_trigger.access_count = 0;
    swap_out_trigger.total_access_count = out_tensor_total_access;

    auto& swap_in_trigger = swap_triggers_[in_trigger_name];
    swap_in_trigger.tensor_name = in_trigger_name;
    swap_in_trigger.in_trigger_count = in_trigger_count;
    swap_in_trigger.in_tensor = out_tensor_name;
    swap_in_trigger.in_params = &swap_params;
    swap_in_trigger.access_count = 0;
    swap_in_trigger.total_access_count = in_trigger_total_access;
  }
  fin.close();
}

Status PrepareCopy(Device* device, const DeviceContext* ctx,
    const DeviceBase::GpuDeviceInfo** dev_info, gpu::Stream** stream) {
  if (device == nullptr) {
    return errors::Internal("Unexpected null device.");
  }
  auto di = device->tensorflow_gpu_device_info();
  if (di == nullptr) {
    return errors::Internal("Unexpected null device info.");
  }
  *dev_info = di;
  if (ctx == nullptr) {
    return errors::Internal("Unexpected null device context.");
  }
  auto gs = static_cast<const GPUDeviceContext*>(ctx)->stream();
  if (gs == nullptr) {
    return errors::Internal("No gpu stream is available.");
  }
  *stream = gs;
  return Status::OK();
}

/*----------------current implem SwapOut---------------*/

void GPUBFCAllocator::SwapOut(const string& tensor_name, const int64 retain_size) {
  /* if (invalid_swap_.count(tensor_name) != 0){
  #ifdef _DEBUGV2
    LOG(INFO) << "Ignore the invalid swap out: " << tensor_name;
  #endif
    return;
  } */

  DCHECK(tensor_swap_params_map_.count(tensor_name));
  auto &swap_params = tensor_swap_params_map_[tensor_name];
  auto &cv_mu = swap_params.cv_mu;
  {
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    swap_params.data_ready = SwapStatus::SWAPPING_OUT;
  }

  LOG(INFO) << "Start to swap out: " << tensor_name;

  TensorBuffer* tensor_buffer = swap_params.tensor_buffer;
  // HashBuffer* hash_buffer = swap_params.hash_buffer;
  float out_fraction = swap_params.out_fraction;

  void* src_ptr = (void*)(tensor_buffer->data());
  int64 total_bytes = RequestedSize(src_ptr);
  int64 gpu_part_size, cpu_part_size;
  if (fabs(out_fraction) < 1e-6) {
    gpu_part_size = total_bytes;
    cpu_part_size = 0;
  } else if (fabs(out_fraction - 1.0f) < 1e-6) {
    gpu_part_size = 0;
    cpu_part_size = total_bytes;
  } else {
    cpu_part_size = int(total_bytes * out_fraction) / 4 * 4;
    gpu_part_size = total_bytes - cpu_part_size;
  }

  if (cpu_part_size < kCopyThreshold) {
  #ifdef _DEBUGV2
    LOG(INFO) << tensor_name << " memory size is below 2MB, ignore it!";
  #endif
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    swap_params.data_ready = SwapStatus::IN;
    return;
  }
#ifdef _DEBUG
  LOG(INFO) << "Start to swap out: " << tensor_name;
#endif


  Device* device = swap_params.device;
  DeviceContext* device_context = swap_params.device_context;
  const DeviceBase::GpuDeviceInfo* dev_info = nullptr;
  gpu::Stream* send_stream = nullptr;
  Status s = PrepareCopy(device, device_context, &dev_info, &send_stream);
  if (!s.ok()) {
    LOG(FATAL) << "PrepareCopy failed.";
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    swap_params.data_ready = SwapStatus::IN;
    return;
  }

#ifdef _DEBUGV2
  LOG(INFO) << "PrepareCopy success: " << tensor_name;
#endif

  swap_params.swapped_gpu_buffer = std::make_pair(src_ptr, gpu_part_size);

  static Allocator* cuda_host_allocator = ProcessState::singleton()->GetCUDAHostAllocator(0);
  void* cpu_part_dst_ptr = cuda_host_allocator->AllocateRaw(0, cpu_part_size);
  swap_params.swapped_cpu_buffer = std::make_pair(cpu_part_dst_ptr, cpu_part_size);

  if (cpu_part_dst_ptr == nullptr) {
    LOG(FATAL) << "Allocate host memory failed.";
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    swap_params.data_ready = SwapStatus::IN;
    return;
  }


  auto device_to_host_stream =
      static_cast<const GPUDeviceContext*>(device_context)->device_to_host_stream();
  if (device_to_host_stream == nullptr) {
    LOG(FATAL) << "No device_to_host_stream is available.";
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    swap_params.data_ready = SwapStatus::IN;
    return;
  }
  // Wait for the sender's main stream to make sure the data are available.
  // px: As the kernel in compute() function in ExecutorState::Process() is async,
  // we need to wait the stream to complete current computation
  // TODO(px): this ThenWaitFor is a coarse-grain sync, as the tensor's producer kernel may finish and queue a lot of other kernel
  // and still we need to wait for these useless kernel to finish so we can execute our swapping func properly.
  // Maybe we can add a event to indicate the completion of tensor's producer kernel and wait only for that particular event.
  // No need to record this time as the ThenWaitFor will not block the host
  device_to_host_stream->ThenWaitFor(send_stream);


  DeviceMemoryBase gpu_src_ptr((void*)((uintptr_t)src_ptr + gpu_part_size), cpu_part_size);
  device_to_host_stream->ThenMemcpy(cpu_part_dst_ptr, gpu_src_ptr, cpu_part_size);


  // Use of the input may outlive stack scope, so keep a ref.
  // (px): keep the ref may cause the GPU memory being exhausted.
  // And there is no need to keep this ref, it's enough that knowing the swapping address.
  // tensor_buffer->Ref();
  dev_info->event_mgr->ThenExecute(
      device_to_host_stream,
      [this, device_to_host_stream, &swap_params]() {
        if (!device_to_host_stream->ok()) {
          LOG(FATAL) << "GPU->CPU Memcpy failed";
          std::lock_guard<std::mutex> l(*(swap_params.cv_mu.second));
          swap_params.data_ready = SwapStatus::IN;
          // tensor_buffer->Unref();
          return;
        }
        auto &cv_mu = swap_params.cv_mu;
        // std::unique_lock<std::mutex> lk(*(cv_mu.second));
        std::lock_guard<std::mutex> lk(*(cv_mu.second));
        swap_params.data_ready = SwapStatus::OUT;
      #ifdef _DEBUGV2
        LOG(INFO) << swap_params.tensor_name << " swap out done.";
      #endif
      });
}


void GPUBFCAllocator::SwapIn(const string& tensor_name) {
  std::lock_guard<std::mutex> l(lock_);
  /* if (invalid_swap_.count(tensor_name) != 0) {
  #ifdef _DEBUGV2
    LOG(INFO) << "Ignore the invalid swap in: " << tensor_name;
  #endif
    return;
  } */

  CHECK(tensor_swap_params_map_.count(tensor_name));
  auto &swap_params = tensor_swap_params_map_[tensor_name];
  auto &cv_mu = swap_params.cv_mu;
  {
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    int ready = swap_params.data_ready;
    if (ready != SwapStatus::SWAPPING_OUT and ready != SwapStatus::OUT) {
      return;
    }
    swap_params.data_ready = SwapStatus::SWAPPING_IN;
  }

#ifdef _DEBUG
  LOG(INFO) << "Start to swap in " << tensor_name;
#endif

  void* gpu_part_src_ptr = swap_params.swapped_gpu_buffer.first;
  void* cpu_part_src_ptr = swap_params.swapped_cpu_buffer.first;
  int64 gpu_part_size = swap_params.swapped_gpu_buffer.second;
  int64 cpu_part_size = swap_params.swapped_cpu_buffer.second;

  Device* device = swap_params.device;
  DeviceContext* device_context = swap_params.device_context;
  const DeviceBase::GpuDeviceInfo* dev_info = nullptr;
  gpu::Stream* recv_stream = nullptr;

  // Get comp. stream and d2h stream to wait for to make sure right trigger time and ready data
  Status s = PrepareCopy(device, device_context, &dev_info, &recv_stream);
  auto device_to_host_stream =
    static_cast<const GPUDeviceContext*>(device_context)->device_to_host_stream();
  if (!s.ok()) {
    LOG(FATAL) << "PrepareCopy failed";
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    swap_params.data_ready = SwapStatus::OUT;
    return;
  }

  static Allocator* cuda_host_allocator = ProcessState::singleton()->GetCUDAHostAllocator(0);

  // TODO(px): deprecated: no need partial swapping
  if (gpu_part_size > 0) {
  #ifdef _DEBUGV2
    LOG(INFO) << "[SwapIn] Start to try to merge copy.";
  #endif
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(gpu_part_src_ptr);
    CHECK(h != kInvalidChunkHandle);
    BFCAllocator::Chunk* c = ChunkFromHandle(h);
    BFCAllocator::Chunk* c_next = nullptr;
    if (c->next != kInvalidChunkHandle) {
      c_next = ChunkFromHandle(c->next);
    }
    mutex_lock ll(BFCAllocator::lock_);
    if (c_next && c_next->size >= cpu_part_size && ! c_next->in_use()) {
      // try to avoid the intra-device memory copy

      RemoveFreeChunkFromBin(c->next);
      c_next->allocation_id = next_allocation_id_++;
      void* gpu_part2_ptr = c_next->ptr;

      auto host_to_device_stream =
        static_cast<const GPUDeviceContext*>(device_context)->host_to_device_stream();
      if (host_to_device_stream == nullptr) {
        LOG(FATAL) << "No host_to_device_stream is available.";
        std::lock_guard<std::mutex> l(*(cv_mu.second));
        swap_params.data_ready = SwapStatus::OUT;
        return;
      }

      // TODO: we don't need to wait for the recv_stream as the data in host is ready for sure
      // (px): As this insertion is in stack, need to block the h2d stream
      // wait for comp. and d2h stream to make sure right trigger time and ready data
      host_to_device_stream->ThenWaitFor(recv_stream, device_to_host_stream);

      DeviceMemoryBase gpu_dst_ptr(gpu_part2_ptr, cpu_part_size);
      host_to_device_stream->ThenMemcpy(&gpu_dst_ptr, cpu_part_src_ptr, cpu_part_size);

      dev_info->event_mgr->ThenExecute(
        host_to_device_stream,
        [this, host_to_device_stream, gpu_part_src_ptr, gpu_part2_ptr, cpu_part_src_ptr, &swap_params]() {
          if (!host_to_device_stream->ok()) {
            LOG(FATAL) << "CPU->GPU Memcpy failed";
            return;
          }
          auto& cv_mu = swap_params.cv_mu;
          std::lock_guard<std::mutex> l(*(cv_mu.second));
          swap_params.data_ready = SwapStatus::IN;
          MergeBuffers(gpu_part_src_ptr, gpu_part2_ptr);
          swap_params.tensor_buffer->set_data(gpu_part_src_ptr);
          cuda_host_allocator->DeallocateRaw(cpu_part_src_ptr);
          cv_mu.first->notify_all();
        });

      return;
    }
  }

  void* dst_ptr = AllocateRaw(0, gpu_part_size + cpu_part_size);
  swap_params.in_gpu_src = dst_ptr;

  // TODO(px): to be removed
  if (gpu_part_size > 0) {
  #ifdef _DEBUGV2
    LOG(INFO) << "[SwapIn] Start to device_to_device copy.";
  #endif
    auto device_to_device_stream =
      static_cast<const GPUDeviceContext*>(device_context)->device_to_device_stream();
    if (device_to_device_stream == nullptr) {
      LOG(FATAL) << "No device_to_device_stream is available.";
      std::lock_guard<std::mutex> l(*(cv_mu.second));
      swap_params.data_ready = SwapStatus::OUT;
      return;
    }

    DeviceMemoryBase gpu_src_ptr(gpu_part_src_ptr, gpu_part_size);
    DeviceMemoryBase gpu_dst_ptr(dst_ptr, gpu_part_size);

    device_to_device_stream->ThenMemcpy(&gpu_dst_ptr, gpu_src_ptr, gpu_part_size);

    dev_info->event_mgr->ThenExecute(
      device_to_device_stream,
      [this, gpu_part_src_ptr]() {
        DeallocateRaw(gpu_part_src_ptr);
      });
  }


  auto host_to_device_stream=
      static_cast<const GPUDeviceContext*>(device_context)->host_to_device_stream();

  if (host_to_device_stream == nullptr) {
    LOG(FATAL) << "No host_to_device_stream is available.";
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    swap_params.data_ready = SwapStatus::OUT;
    return;
  }

  // Wait for the recv-stream to come to the appropriate trigger point
  host_to_device_stream->ThenWaitFor(recv_stream);

  DeviceMemoryBase gpu_dst_ptr((void*)((uintptr_t)dst_ptr + gpu_part_size), cpu_part_size);
  host_to_device_stream->ThenMemcpy(&gpu_dst_ptr, cpu_part_src_ptr, cpu_part_size);

  {
    // when enqueue the memcpy to the stream, we need to make the comp.stream wait for
    // h2d stream to make sure the data is ready.
    // TODO(px): make this more fined that wait for single tensor
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    swap_params.need_in_addr = true;
  }

  // Use of the input may outlive stack scope, so keep a ref.
  dev_info->event_mgr->ThenExecute(
      host_to_device_stream,
      [host_to_device_stream, dst_ptr, cpu_part_src_ptr, &swap_params]() {
        if (!host_to_device_stream->ok()) {
          LOG(FATAL) << "GPU->CPU Memcpy failed";
          std::lock_guard<std::mutex> l(*(swap_params.cv_mu.second));
          swap_params.data_ready = SwapStatus::OUT;
          return;
        }
        auto &cv_mu = swap_params.cv_mu;
        std::lock_guard<std::mutex> l(*(cv_mu.second));
      #ifdef _DEBUGV2
        LOG(INFO) << swap_params.tensor_name << " swap in done.";
      #endif
        swap_params.data_ready = SwapStatus::IN;
        cuda_host_allocator->DeallocateRaw(cpu_part_src_ptr);
      });
}

}  // namespace tensorflow
