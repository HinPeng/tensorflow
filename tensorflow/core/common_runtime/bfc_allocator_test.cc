#include "tensorflow/core/common_runtime/bfc_allocator.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {

TEST(BFCAllocatorTest, TraceSim) {
  std::vector<SubAllocator::Visitor> alloc_visitors;
  std::vector<SubAllocator::Visitor> free_visitors;

  SubAllocator* sub_allocator = new BasicCPUAllocator(port::kNUMANoAffinity, alloc_visitors, free_visitors);

  int64 mem_limit_in_mb;
  Status status = ReadInt64FromEnvVar("TF_CPU_BFC_MEM_LIMIT_IN_MB",
                                      1LL << 14 /*16GB max by default*/,
                                      &mem_limit_in_mb);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }

  int64 mem_limit = mem_limit_in_mb * (1LL << 20);
  DCHECK(sub_allocator);
  BFCAllocator allocator(sub_allocator, mem_limit, false/*allow_growth*/,
                         "bfc_cpu_allocator" /*name*/);

  // prepare the allocation trace
  // allocations are already ordered by time in this file
  string trace_file_name;
  status = ReadStringFromEnvVar("TF_BFC_ALLOCATOR_TEST_TRACE_FILE",
                                "",
                                &trace_file_name);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }

  
  std::vector<std::pair<std::string, int64>> allocs;
  std::unordered_map<std::string, void*> ptrs;
  std::ifstream trace_file(trace_file_name, trace_file.in);
  if (!trace_file.is_open()) {
    LOG(ERROR) << "Trace file " << trace_file_name << " can not be opened!";
  }

  std::string line, tensor_name;
  std::stringstream ss;
  uint64 alloc_time;
  int64 alloc_bytes;
  while (getline(trace_file, line)) {
    ss.str(line);
    ss >> alloc_time >> tensor_name >> alloc_bytes;
    allocs.push_back(std::make_pair(tensor_name, alloc_bytes));
    ss.clear();
  }
  LOG(INFO) << "Read Trace file success, total allocations: " << allocs.size() / 2;

  absl::optional<AllocatorStats> stats;
  uint64 max_addr = 0x000000000000;   // should not care the max base_addr, but the max(base_addr+alloc_bytes)
  uint64 min_addr = 0xffffffffffff;
  int64 alloc_bytes_max_addr;
  // void* max_addr, min_addr;
  for (auto it: allocs) {
    if (it.second > 0) {
      void* raw = allocator.AllocateRaw(1, it.second);
      if (ptrs.find(it.first) != ptrs.end()) {
        LOG(ERROR) << "Tensor [" << it.first << "] with duplicated allocate";
        return;
      }
      ptrs[it.first] = raw;
      uint64 raw_ = reinterpret_cast<uint64>(raw);
      if (raw_ < min_addr) min_addr = raw_;
      if ((raw_+static_cast<uint64>(it.second)) > max_addr) {
        max_addr = raw_+static_cast<uint64>(it.second);
        alloc_bytes_max_addr = it.second;
      }
      // LOG(INFO) << it.first << " : "  << raw;
    } else {
      if (ptrs.find(it.first) == ptrs.end()) {
        LOG(ERROR) << "Tensor [" << it.first << "] has not been allocated yet!";
        return;
      }
      allocator.DeallocateRaw(ptrs[it.first]);
    }
  }

  // LOG(INFO) << "Min addr: " << min_addr << ", Max addr: " << max_addr << ", diff: " << max_addr - min_addr;
  // printf("Min addr: 0x%llx, Max addr: 0x%llx, diff: %llu\n", min_addr, max_addr, max_addr-min_addr);
  printf("Allocation range: [0x%llx, 0x%llx], diff: %llu, Max addr allocation bytes: %llu\n", min_addr, max_addr, max_addr-min_addr, alloc_bytes_max_addr);
  // printf("Min addr: 0x%llx\nMax addr: 0x%llx\nAllocation bytes at max addr: %llu\nMaxInUse: %llu\n", min_addr, max_addr, alloc_bytes_max_addr, max_addr-min_addr);
  stats = allocator.GetStats();
  LOG(INFO) << stats->DebugString();
}

}
} // namespace tensorflow