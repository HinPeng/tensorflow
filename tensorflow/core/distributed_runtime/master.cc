/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Master implements the service MasterService.
//
// A Master maintains the state of live graph computation
// sessions, each session orchestrates both local and remote devices
// to carry out the graph computation.
//
// A Master knows ahead of time local devices available as
// client devices.
//
// A Master discovers remote devices on-demand and keeps track of
// statistics of those remote devices.
//
// Each session analyzes the graph, places nodes across available
// devices, and ultimately drives the graph computation by initiating
// RunGraph on the workers.

#include "tensorflow/core/distributed_runtime/master.h"

#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/distributed_runtime/remote_device.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"

#define _TF_USE_SHARED_SESSION

namespace tensorflow {

namespace {
const char* const kGrpcProtocol = "grpc://";
static bool _log_allocator_stats;
}  // namespace

Master::Master(MasterEnv* env, double session_gc_seconds)
    : env_(env),
      last_1000_steps_(1000),
      step_count_(0),
      session_gc_seconds_(session_gc_seconds),
      recent_request_ids_(10000) {
  // Right now, a master service must be co-located with a device.
  // Otherwise, fetches do not work.
  CHECK(!env->local_devices.empty());
  ReadBoolFromEnvVar("_TF_LOG_ALLOCATOR_STATS", false, &_log_allocator_stats);

  if (session_gc_seconds_ > 0.0) {
    gc_thread_ = env_->env->StartThread(ThreadOptions(), "TF_master_GC",
                                        [this]() { GC(); });
  } else {
    gc_thread_ = nullptr;
  }
}

Master::~Master() {
  if (gc_thread_) {
    mutex_lock l(mu_);
    shutdown_ = true;
    shutdown_cv_.notify_all();
    delete gc_thread_;
  }
}

void Master::GC() {
  Env* env = Env::Default();
  while (true) {
    mutex_lock l(mu_);
    const int kTimeoutMilliseconds = 10 * 1000;  // 10 seconds.
    WaitForMilliseconds(&l, &shutdown_cv_, kTimeoutMilliseconds);
    if (shutdown_) {
      break;
    }
    std::vector<string> handles;
    const int64 num_micros = static_cast<int64>(session_gc_seconds_ * 1000000);
    for (const auto& entry : sessions_) {
      int64 lat = entry.second->last_access_time_usec();
      if (static_cast<int64>(env->NowMicros()) - lat > num_micros) {
        handles.push_back(entry.first);
        auto* sess = entry.second;
        SchedClosure([this, sess]() {
          LOG(WARNING) << "GC session " << sess->handle() << " after "
                       << session_gc_seconds_ << " seconds.  "
                       << "Note that if you are starting multiple replicas "
                       << "on a staggered delay, session_gc_seconds may need "
                       << "to be raised.";
          sess->GarbageCollect();
        });
      }
    }
    for (const auto& handle : handles) sessions_.erase(handle);
  }
}

#ifdef _TF_USE_SHARED_SESSION
SharedMasterSession* Master::FindMasterSession(const string& handle) {
  SharedMasterSession* session = nullptr;
  {
    mutex_lock l(mu_);
    session = gtl::FindPtrOrNull(sessions_, handle);
    if (session != nullptr) {
      session->Ref();
    }
  }
  return session;
}

SharedMasterSession* Master::MayGetSharedMasterSession(
    const CreateSessionRequest* req) {
  if (session_configs_.empty()) return nullptr;

  bool find = true;
  string find_session_handle = "";
  SharedMasterSession* session = nullptr;
  const ClusterDef& curr_cluster_def = req->config().cluster_def();
  // TODO(px): same cluster_def can not identify share or not
  // as same task can own different number of GPUs
  for (const auto& entry : session_configs_) {
    // for (const auto& config : session_configs_) {
    const Config& config = entry.second;
    if (!config.accept_shared_) continue;

    const string& session_handle = entry.first;
    const ClusterDef& cluster_def = config.cluster_def_;
    if (sessions_.at(session_handle)->NumTasks() > 1) {
      // TODO(px): Currently, we don't consider pack more than two jobs in one
      // session
      LOG(WARNING) << "Session (handle: "
                   << sessions_.at(session_handle)->handle()
                   << ") has more than one task: "
                   << sessions_.at(session_handle)->NumTasks();
      continue;
    }
    find = true;
    if (curr_cluster_def.job_size() > 2) {
      LOG(ERROR) << "Error cluster_def: "
                 << curr_cluster_def.ShortDebugString();
      break;
    }
    const JobDef* c_worker_job = nullptr;
    for (auto&& job : curr_cluster_def.job()) {
      if (job.name() == "ps") continue;
      c_worker_job = &job;
    }
    // TODO(px): check c_worker_job is not null
    for (auto&& job : cluster_def.job()) {
      if (job.name() == "ps")
        continue;  // the tasks in 'PS' job can be different
      if (job.tasks_size() != c_worker_job->tasks_size()) {
        find = false;
        break;
      }

      for (auto&& task : job.tasks()) {
        auto c_string = c_worker_job->tasks().at(task.first);
        if (task.second != c_string) {
          find = false;
          break;
        }
      }
    }
    // find a session that can be shared
    if (find) {
      find_session_handle = session_handle;
      break;
    }
  }

  if (!find_session_handle.empty()) {
    session = sessions_.at(find_session_handle);
    session->Ref();
  }
  return session;
}
#else
MasterSession* Master::FindMasterSession(const string& handle) {
  MasterSession* session = nullptr;
  {
    mutex_lock l(mu_);
    session = gtl::FindPtrOrNull(sessions_, handle);
    if (session != nullptr) {
      session->Ref();
    }
  }
  return session;
}
#endif

class DeviceFinder {
 public:
  static Status GetRemoteDevices(
      const protobuf::RepeatedPtrField<string>& device_filters, MasterEnv* env,
      WorkerCacheInterface* worker_cache,
      std::vector<std::unique_ptr<Device>>* out_remote) {
    DeviceFinder finder(device_filters, env, worker_cache);
    finder.Start();
    TF_RETURN_IF_ERROR(finder.Wait());
    finder.GetRemoteDevices(env->local_devices, out_remote);
    return Status::OK();
  }

  static void GetRemoteWorkers(
      const protobuf::RepeatedPtrField<string>& device_filters, MasterEnv* env,
      WorkerCacheInterface* worker_cache, std::vector<string>* workers) {
    DeviceFinder finder(device_filters, env, worker_cache);
    *workers = finder.targets_;
  }

 private:
  explicit DeviceFinder(
      const protobuf::RepeatedPtrField<string>& device_filters, MasterEnv* env,
      WorkerCacheInterface* worker_cache)
      : env_(env), worker_cache_(worker_cache) {
    CHECK(worker_cache) << "Worker cache was null!";
    auto process_filter = [this](const string& filter) {
      DeviceNameUtils::ParsedName parsed;
      if (DeviceNameUtils::ParseFullName(filter, &parsed)) {
        filters_.push_back(parsed);
      } else {
        LOG(FATAL) << "Skipping invalid filter: " << filter;
      }
    };
    for (const string& filter : device_filters) {
      process_filter(filter);
    }
    // Enumerates all known workers' target. A target name is a
    // prefix of a device name. E.g., /job:mnist/replica:0/task:10.
    if (filters_.empty()) {
      // If no filters were specified, we list all known workers in
      // `worker_cache`.
      std::vector<string> workers;
      worker_cache->ListWorkers(&workers);
      std::swap(workers, targets_);
    } else {
      // When applying filters, we must include the local worker, even if it
      // does not match any of the filters.
      CHECK_GT(env_->local_devices.size(), 0) << "No local devices provided.";
      const string& local_device_name = env_->local_devices[0]->name();
      DeviceNameUtils::ParsedName local_parsed_name;
      CHECK(DeviceNameUtils::ParseFullName(local_device_name,
                                           &local_parsed_name));
      bool all_filters_have_job = true;
      std::unordered_set<string> filter_job_names({local_parsed_name.job});
      for (const DeviceNameUtils::ParsedName& filter : filters_) {
        all_filters_have_job = all_filters_have_job && filter.has_job;
        if (filter.has_job) {
          filter_job_names.insert(filter.job);
        }
      }

      std::vector<string> workers;
      if (all_filters_have_job) {
        // If all of the device filters have a job specified, then we only need
        // to list the workers in the jobs named in the filter, because a worker
        // in any other job would not match any filter.
        for (const string& job_name : filter_job_names) {
          VLOG(2) << "Selectively listing workers in job: " << job_name;
          std::vector<string> workers_in_job;
          worker_cache->ListWorkersInJob(job_name, &workers_in_job);
          workers.insert(workers.end(), workers_in_job.begin(),
                         workers_in_job.end());
        }
      } else {
        // If any of the device filters does not have a job specified, then we
        // must list the workers from all jobs.
        VLOG(2) << "Listing workers in all jobs because some device "
                << "filter has no job specified. Filters were:";
        if (device_filters.empty()) {
          VLOG(2) << "- <NO FILTERS>";
        } else {
          for (const string& filter : device_filters) {
            VLOG(2) << "- " << filter;
          }
        }
        worker_cache->ListWorkers(&workers);
      }
      for (const string& name : workers) {
        if (MatchFilters(name) ||
            DeviceNameUtils::IsSameAddressSpace(name, local_device_name)) {
          targets_.push_back(name);
        }
      }
    }
    seen_targets_.assign(targets_.size(), false);
  }

  ~DeviceFinder() {
    for (Device* dev : found_) delete dev;
  }

  void Start() {
    {
      mutex_lock l(mu_);
      num_pending_ = targets_.size();
      if (num_pending_ == 0) {
        pending_zero_.notify_all();
      }
    }
    // Talk to all workers to get the list of available devices.
    using std::placeholders::_1;
    using std::placeholders::_2;
    for (size_t i = 0; i < targets_.size(); ++i) {
      // TODO(mrry): Propagate a timeout here, since `this->WhenFound()` may
      // never be called.
      NewRemoteDevices(env_->env, worker_cache_, targets_[i],
                       std::bind(&ME::WhenFound, this, i, _1, _2));
    }
  }

  // Every `kLoggingPeriodMs`, while the DeviceFinder is still waiting
  // to hear from workers, log a list of the workers who have not
  // responded.
  const int32 kLoggingPeriodMs = 10 * 1000;

  Status Wait() {
    mutex_lock l(mu_);
    // TODO(mrry): Propagate a timeout here, since `num_pending_` may
    // never become zero.
    while (num_pending_ != 0) {
      pending_zero_.wait_for(l, std::chrono::milliseconds(kLoggingPeriodMs));
      if (num_pending_ != 0) {
        for (size_t i = 0; i < targets_.size(); ++i) {
          if (!seen_targets_[i]) {
            LOG(INFO)
                << "CreateSession still waiting for response from worker: "
                << targets_[i];
          }
        }
      }
    }
    return status_;
  }

  // The caller takes the ownership of returned remote devices.
  void GetRemoteDevices(const std::vector<Device*>& local,
                        std::vector<std::unique_ptr<Device>>* remote) {
    std::unordered_set<string> names(local.size());
    for (Device* dev : local) names.insert(dev->name());
    mutex_lock l(mu_);
    for (Device* dev : found_) {
      const string& name = dev->name();
      if (names.insert(name).second && MatchFilters(name)) {
        remote->push_back(std::unique_ptr<Device>(dev));
      } else {
        delete dev;
      }
    }
    found_.clear();
  }

  typedef DeviceFinder ME;
  const MasterEnv* env_;
  WorkerCacheInterface* worker_cache_;
  std::vector<DeviceNameUtils::ParsedName> filters_;

  mutex mu_;
  int num_pending_ GUARDED_BY(mu_);
  condition_variable pending_zero_;
  std::vector<Device*> found_ GUARDED_BY(mu_);
  // List of targets to be contacted by this DeviceFinder. The
  // respective `bool` in `seen_targets_` indicates whether we have
  // heard from this target or not.
  std::vector<string> targets_;
  std::vector<bool> seen_targets_ GUARDED_BY(mu_);
  Status status_;

  void WhenFound(int target_index, const Status& s,
                 std::vector<Device*>* devices) {
    mutex_lock l(mu_);
    seen_targets_[target_index] = true;
    if (!s.ok()) {
      LOG(ERROR) << "CreateSession failed because worker "
                 << targets_[target_index] << " returned error: " << s;
      status_.Update(s);
    } else {
      found_.insert(found_.end(), devices->begin(), devices->end());
      devices->clear();
    }
    --num_pending_;
    if (num_pending_ == 0) {
      pending_zero_.notify_all();
    }
  }

  // Returns true iff the set of devices allowed by 'x' intersects
  // with the set of devices allowed by 'y'.
  bool Intersects(const DeviceNameUtils::ParsedName& x,
                  const DeviceNameUtils::ParsedName& y) {
    return (!x.has_job || !y.has_job || x.job == y.job) &&
           (!x.has_replica || !y.has_replica || x.replica == y.replica) &&
           (!x.has_task || !y.has_task || x.task == y.task) &&
           (!x.has_type || !y.has_type || x.type == y.type) &&
           (!x.has_id || !y.has_id || x.id == y.id);
  }

  // Returns true iff 'name' matches one of the filters_.
  bool MatchFilters(const string& name) {
    if (filters_.empty()) return true;
    DeviceNameUtils::ParsedName x;
    if (DeviceNameUtils::ParseFullName(name, &x)) {
      for (const auto& filter : filters_) {
        if (Intersects(x, filter)) return true;
      }
    }
    return false;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceFinder);
};

namespace {
// utility to extract session_handle from session_handle with task_id
inline const ::tensorflow::string _session_handle(
    const ::tensorflow::string& s) {
  // TODO(px): currently, the task_id is < 10
  return s.substr(0, s.length() - 2);
}

// utility to extract task_id from request
template <class ClientRequest>
inline const int32 GetTaskId(const ClientRequest* req) {
  ::tensorflow::string session_handle = req->session_handle();
  return static_cast<int32>(session_handle.back() - '0');
}

template <class ClientRequest>
inline const int32 GetTaskId(const ClientRequest& req) {
  return GetTaskId(&req);
}

}  // namespace

void Master::CreateSession(const CreateSessionRequest* req,
                           CreateSessionResponse* resp, MyClosure done) {
  SchedClosure([this, req, resp, done]() {
    Status status;
    WorkerCacheFactoryOptions worker_cache_factory_options;
    string grpc_protocol("grpc");
    worker_cache_factory_options.protocol = &grpc_protocol;
    auto call_done = gtl::MakeCleanup([&status, &done] { done(status); });
    status = ValidateExternalGraphDefSyntax(req->graph_def());
    if (!status.ok()) return;

    // The following 4 variables are set differently, depending on whether this
    // session uses a client-provided clusterspec or not.
    WorkerCacheInterface* worker_cache = nullptr;
    // Note: worker_cache_ptr will be null except if this session is using a
    // client-supplied ClusterDef (ClusterSpec propagation).
    std::unique_ptr<WorkerCacheInterface> worker_cache_ptr;
    std::unique_ptr<DeviceSet> device_set;
    // TODO(saeta): Convert to std::make_unique when available.
    std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devices(
        new std::vector<std::unique_ptr<Device>>());

    if (req->config().has_cluster_def()) {
      worker_cache_factory_options.cluster_def = &req->config().cluster_def();

      // Set the server_def's job_name and task_index fields.
      string normalized_string;
      string grpc_protocol(kGrpcProtocol);
      if (req->target().compare(0, grpc_protocol.length(), grpc_protocol) ==
          0) {
        normalized_string =
            req->target().substr(grpc_protocol.length(), string::npos);
      } else {
        normalized_string = req->target();
      }
      for (auto&& job : req->config().cluster_def().job()) {
        for (auto&& task : job.tasks()) {
          if (task.second == normalized_string) {
            if (worker_cache_factory_options.job_name != nullptr) {
              status = errors::InvalidArgument(
                  "Found multiple matching tasks that correspond to "
                  "to the master. Master target: '",
                  req->target(), "'. ClusterDef: ",
                  req->config().cluster_def().ShortDebugString());
              LOG(ERROR) << status;
              return;
            }
            if (env_->local_devices[0]->parsed_name().job == job.name() &&
                env_->local_devices[0]->parsed_name().task == task.first) {
              // TODO(b/37868888): Remove this limitation when resolved
              status = errors::InvalidArgument(
                  "The ClusterSpec names the job and task index to be the same "
                  "names that were provided when the server booted. This is "
                  "currently not allowed. Job: ",
                  job.name(), ", task index: ", task.first);
              return;
            }
            worker_cache_factory_options.job_name = &job.name();
            worker_cache_factory_options.task_index = task.first;
          }
        }
      }

      // Create the worker cache from the computed server_def.
      status = env_->worker_cache_factory(worker_cache_factory_options,
                                          &worker_cache);
      if (!status.ok()) return;
      worker_cache_ptr = std::unique_ptr<WorkerCacheInterface>(worker_cache);
      // Ping all the workers and build the list of devices that the
      // session will use.
      status =
          DeviceFinder::GetRemoteDevices(req->config().device_filters(), env_,
                                         worker_cache, remote_devices.get());
      if (!status.ok()) return;
      device_set.reset(new DeviceSet);
      for (auto&& d : *remote_devices) {
        device_set->AddDevice(d.get());
        DeviceNameUtils::ParsedName name = d->parsed_name();
        if (name.job == *worker_cache_factory_options.job_name &&
            name.task == worker_cache_factory_options.task_index &&
            name.type == "CPU" && name.id == 0) {
          device_set->set_client_device(d.get());
        }
      }
    } else {
      worker_cache = env_->worker_cache;
      // Ping all the workers and build the list of devices that the
      // session will use.
      status =
          DeviceFinder::GetRemoteDevices(req->config().device_filters(), env_,
                                         worker_cache, remote_devices.get());
      if (!status.ok()) return;
      device_set.reset(new DeviceSet);
      for (auto&& d : *remote_devices) {
        device_set->AddDevice(d.get());
      }
      int num_local_devices = 0;
      for (Device* d : env_->local_devices) {
        device_set->AddDevice(d);
        if (num_local_devices == 0) {
          // Uses the first local device as the client device.
          device_set->set_client_device(d);
        }
        num_local_devices++;
      }
    }

#ifdef _TF_USE_SHARED_SESSION
    mutex_lock l(mu_);
    SharedMasterSession* shared_session = nullptr;
    // auto shared_session = MayGetSharedMasterSession(req);
    if (req->config().has_gpu_options()) {
      if (req->config().gpu_options().allow_shared()) {
        shared_session = MayGetSharedMasterSession(req);
      } else {
        VLOG(1) << "Task do not allow sharing GPU!";
      }
    }

    if (shared_session) {
      SessionOptions options;
      options.config = req->config();
      int32 next_task_id = shared_session->NumTasks();

      GraphDef* gdef =
          const_cast<CreateSessionRequest*>(req)->mutable_graph_def();

      status = shared_session->AddTask(next_task_id, options, std::move(*gdef),
                                       worker_cache_factory_options);
      if (!status.ok()) {
        shared_session->Close(next_task_id).IgnoreError();
        shared_session->Unref();
        return;
      }

      string session_handle_w_tid =
          strings::StrCat(shared_session->handle(), "_", next_task_id);
      resp->set_session_handle(session_handle_w_tid);
      return;
    }
#endif

    CHECK(device_set->client_device()) << "No client device found. Missing "
                                       << "CPU:0 device?";

    SessionOptions options;
    options.config = req->config();

    std::vector<string> filtered_worker_list;
    DeviceFinder::GetRemoteWorkers(req->config().device_filters(), env_,
                                   worker_cache, &filtered_worker_list);

#ifdef _TF_USE_SHARED_SESSION
    SharedMasterSession* session = env_->shared_master_session_factory(
        options, env_, std::move(remote_devices), std::move(worker_cache_ptr),
        std::move(device_set), std::move(filtered_worker_list));
#else
    MasterSession* session = env_->master_session_factory(
        options, env_, std::move(remote_devices), std::move(worker_cache_ptr),
        std::move(device_set), std::move(filtered_worker_list));
#endif

    GraphDef* gdef =
        const_cast<CreateSessionRequest*>(req)->mutable_graph_def();
    status = session->Create(std::move(*gdef), worker_cache_factory_options);
    if (!status.ok()) {
      session->Close(0).IgnoreError();
      session->Unref();
      return;
    }
#ifdef _TF_USE_SHARED_SESSION
    string session_handle = strings::StrCat(session->handle(), "_", 0);
#else
    string session_handle = session->handle();
#endif
    resp->set_session_handle(session_handle);
    // Insert into the session map, which takes ownership of the session.
    {
#ifndef _TF_USE_SHARED_SESSION
      mutex_lock l(mu_);
#endif
      CHECK(sessions_.insert({session->handle(), session}).second);
#ifdef _TF_USE_SHARED_SESSION
      CHECK(session_configs_
                .insert({session->handle(),
                         Config(req->config().cluster_def(),
                                req->config().has_gpu_options()
                                    ? req->config().gpu_options().allow_shared()
                                    : false)})
                .second);
#endif
    }
  });
}

void Master::ExtendSession(const ExtendSessionRequest* req,
                           ExtendSessionResponse* resp, MyClosure done) {
#ifdef _TF_USE_SHARED_SESSION
  string session_handle = _session_handle(req->session_handle());
#else
  string session_handle = req->session_handle();
#endif
  auto session = FindMasterSession(session_handle);
  if (session == nullptr) {
    done(errors::Aborted("Session ", session_handle, " is not found."));
    return;
  }

  SchedClosure([session, req, resp, done]() {
    Status status = ValidateExternalGraphDefSyntax(req->graph_def());
    if (status.ok()) {
      status = session->Extend(req, resp);
    }
    session->Unref();
    done(status);
  });
}

void Master::PartialRunSetup(const PartialRunSetupRequest* req,
                             PartialRunSetupResponse* resp, MyClosure done) {
  Status s = recent_request_ids_.TrackUnique(req->request_id(),
                                             "PartialRunSetup (Master)", *req);
  if (!s.ok()) {
    done(s);
    return;
  }
#ifdef _TF_USE_SHARED_SESSION
  string session_handle = _session_handle(req->session_handle());
#else
  string session_handle = req->session_handle();
#endif
  auto session = FindMasterSession(session_handle);
  if (session == nullptr) {
    done(errors::Aborted("Session ", session_handle, " is not found."));
    return;
  }

  SchedClosure([session, req, resp, done]() {
    Status s = session->PartialRunSetup(req, resp);
    session->Unref();
    done(s);
  });
}

void Master::RunStep(CallOptions* opts, const RunStepRequestWrapper* req,
                     MutableRunStepResponseWrapper* resp, MyClosure done) {
  Status s = recent_request_ids_.TrackUnique(req->request_id(),
                                             "RunStep (Master)", req);
  if (!s.ok()) {
    done(s);
    return;
  }
  auto start_time = env_->env->NowMicros();
#ifdef _TF_USE_SHARED_SESSION
  string session_handle = _session_handle(req->session_handle());
#else
  string session_handle = req->session_handle();
#endif
  auto session = FindMasterSession(session_handle);
  if (session == nullptr) {
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  SchedClosure([this, start_time, session, opts, req, resp, _log_allocator_stats, done]() {
    AllocatorStats stats;
    Status status = session->Run(opts, *req, resp);
    session->Unref();
    uint64 done_time = env_->env->NowMicros();
    done(status);
    if (_log_allocator_stats) {
      GPUOptions options;
      Allocator* gpu_allocator = GPUProcessState::singleton()->GetGPUAllocator(options, TfGpuId(0), 0);
      stats = gpu_allocator->GetStats().value();
      LOG(INFO) << stats.DebugString();
    }
    mutex_lock l(mu_);
    last_1000_steps_.AddValue((done_time - start_time) / 1e9);
    ++step_count_;
  });
}

void Master::CloseSession(const CloseSessionRequest* req,
                          CloseSessionResponse* resp, MyClosure done) {
#ifdef _TF_USE_SHARED_SESSION
  SharedMasterSession* session = nullptr;
  const int32 task_id = GetTaskId(req);
#else
  MasterSession* session = nullptr;
#endif
  {
    mu_.lock();
#ifdef _TF_USE_SHARED_SESSION
    string session_handle = _session_handle(req->session_handle());
#else
    string session_handle = req->session_handle();
#endif
    auto iter = sessions_.find(session_handle);
    if (iter == sessions_.end()) {
      mu_.unlock();
      done(errors::Aborted(
          "Session ", session_handle,
          " is not found. Possibly, this master has restarted."));
      return;
    }
    // NOTE(mrry): One reference to the session is transferred from
    // `sessions_[req->session_handle()]` to `session`.
    session = iter->second;
    if (session->NumTasks() == 1) {
      sessions_.erase(iter);
      session_configs_.erase(session_configs_.find(session_handle));
    }
    mu_.unlock();
  }

  // Session Close() blocks on thread shutdown. Therefore, we need to
  // delete it in non-critical thread.
  SchedClosure([session, task_id, done]() {
    Status s = session->Close(task_id);
    session->Unref();
    done(s);
  });
}

void Master::ListDevices(const ListDevicesRequest* req,
                         ListDevicesResponse* resp, MyClosure done) {
  SchedClosure([this, req, resp, done]() {
    if (!req->session_handle().empty()) {
#ifdef _TF_USE_SHARED_SESSION
      string session_handle = _session_handle(req->session_handle());
#else
      string session_handle = req->session_handle();
#endif
      auto session = FindMasterSession(session_handle);
      if (session == nullptr) {
        done(errors::InvalidArgument(
            "Session ", session_handle,
            " is not found. Possibly, this master has restarted."));
        return;
      }
      core::ScopedUnref ref(session);
      Status s = session->ListDevices(resp);
      done(s);
      return;
    }
    std::vector<std::unique_ptr<Device>> remote_devices;
    Status s = DeviceFinder::GetRemoteDevices({}, env_, env_->worker_cache,
                                              &remote_devices);
    if (s.ok()) {
      for (Device* dev : env_->local_devices) {
        *(resp->add_local_device()) = dev->attributes();
      }
      for (auto&& dev : remote_devices) {
        *(resp->add_remote_device()) = dev->attributes();
      }
    }
    done(s);
  });
}

void Master::CleanupWorkers(const ResetRequest& reset) {
  std::vector<string> worker_names;
  DeviceFinder::GetRemoteWorkers(reset.device_filters(), env_,
                                 env_->worker_cache, &worker_names);
  if (!worker_names.empty()) {
    const int num_workers = worker_names.size();
    std::vector<Notification> n(num_workers);
    CleanupAllRequest req;
    (*req.mutable_container()) = reset.container();
    std::vector<CleanupAllResponse> resp(num_workers);
    int c = 0;
    for (int i = 0; i < num_workers; ++i) {
      const string& worker_name = worker_names[i];
      auto worker = env_->worker_cache->GetOrCreateWorker(worker_name);
      if (worker) {
        worker->CleanupAllAsync(
            &req, &resp[i], [this, &n, worker_name, worker, c](Status s) {
              TF_CHECK_OK(s);
              env_->worker_cache->ReleaseWorker(worker_name, worker);
              n[c].Notify();
            });
      } else {
        n[c].Notify();
      }
      ++c;
    }
    for (size_t i = 0; i < n.size(); ++i) {
      n[i].WaitForNotification();
    }
  }
}

void Master::Reset(const ResetRequest* req, ResetResponse* resp,
                   MyClosure done) {
  // Vector to hold the session pointers present in the sessions_
  // (string->Session*) map.
#ifdef _TF_USE_SHARED_SESSION
  std::vector<SharedMasterSession*> sessions_to_close;
#else
  std::vector<MasterSession*> sessions_to_close;
#endif
  {
    mutex_lock l(mu_);
    // NOTE(mrry): Transfer one reference to each session from the
    // `sessions_` map to the `sessions_to_close` vector.
    for (const auto& entry : sessions_) {
      sessions_to_close.push_back(entry.second);
    }
    sessions_.clear();
  }

  CleanupWorkers(*req);

  SchedClosure([sessions_to_close, done]() {
    Status s;
    for (auto* session : sessions_to_close) {
      for (int i = 0; i < session->NumTasks(); ++i) {
        s.Update(session->Close(i));
        session->Unref();
      }
    }
    done(s);
  });
}

void Master::MakeCallable(const MakeCallableRequest* req,
                          MakeCallableResponse* resp, MyClosure done) {
  Status s = recent_request_ids_.TrackUnique(req->request_id(),
                                             "MakeCallable (Master)", *req);
  if (!s.ok()) {
    done(s);
    return;
  }
  auto session = FindMasterSession(req->session_handle());
  if (session == nullptr) {
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  SchedClosure(std::bind(
      [session, req, resp](MyClosure done) {
        Status s = session->MakeCallable(*req, resp);
        session->Unref();
        done(s);
      },
      std::move(done)));
}

void Master::RunCallable(CallOptions* opts, const RunCallableRequest* req,
                         RunCallableResponse* resp, MyClosure done) {
  Status s = recent_request_ids_.TrackUnique(req->request_id(),
                                             "RunCallable (Master)", *req);
  if (!s.ok()) {
    done(s);
    return;
  }
  auto session = FindMasterSession(req->session_handle());
  if (session == nullptr) {
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  SchedClosure(std::bind(
      [session, opts, req, resp](MyClosure done) {
        Status s = session->RunCallable(opts, *req, resp);
        session->Unref();
        done(s);
      },
      std::move(done)));
}

void Master::ReleaseCallable(const ReleaseCallableRequest* req,
                             ReleaseCallableResponse* resp, MyClosure done) {
  auto session = FindMasterSession(req->session_handle());
  if (session == nullptr) {
    done(errors::Aborted("Session ", req->session_handle(), " is not found."));
    return;
  }

  SchedClosure(std::bind(
      [session, req, resp](MyClosure done) {
        Status s = session->ReleaseCallable(*req, resp);
        session->Unref();
        done(s);
      },
      std::move(done)));
}

}  // end namespace tensorflow
