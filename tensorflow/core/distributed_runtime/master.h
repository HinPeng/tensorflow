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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_

#define _TF_USE_SHARED_SESSION

#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#ifdef _TF_USE_SHARED_SESSION
#include "tensorflow/core/distributed_runtime/shared_master_session.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#endif
#include "tensorflow/core/distributed_runtime/recent_request_ids.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

class Master {
 public:
  explicit Master(MasterEnv* env, double session_gc_seconds);
  virtual ~Master();

  // Convenient typedef for a closure passing a Status.
  typedef std::function<void(const Status&)> MyClosure;

  void CreateSession(const CreateSessionRequest* req,
                     CreateSessionResponse* resp, MyClosure done);

  void ExtendSession(const ExtendSessionRequest* req,
                     ExtendSessionResponse* resp, MyClosure done);

  void PartialRunSetup(const PartialRunSetupRequest* req,
                       PartialRunSetupResponse* resp, MyClosure done);

  void RunStep(CallOptions* opts, const RunStepRequestWrapper* req,
               MutableRunStepResponseWrapper* resp, MyClosure done);

  void CloseSession(const CloseSessionRequest* req, CloseSessionResponse* resp,
                    MyClosure done);

  void ListDevices(const ListDevicesRequest* req, ListDevicesResponse* resp,
                   MyClosure done);

  // See tensorflow::Reset() and the comment on ResetRequest.
  void Reset(const ResetRequest* req, ResetResponse* resp, MyClosure done);

  void MakeCallable(const MakeCallableRequest* req, MakeCallableResponse* resp,
                    MyClosure done);
  void RunCallable(CallOptions* opts, const RunCallableRequest* req,
                   RunCallableResponse* resp, MyClosure done);
  void ReleaseCallable(const ReleaseCallableRequest* req,
                       ReleaseCallableResponse* resp, MyClosure done);

 private:
  typedef Master ME;

  // Not owned.
  MasterEnv* env_ = nullptr;

  // Owned.
  mutex mu_;

  // shutdown_ is set to true by the dtor.
  condition_variable shutdown_cv_;
  bool shutdown_ GUARDED_BY(mu_) = false;
  Thread* gc_thread_;

  // Maps session handles to sessions.
#ifdef _TF_USE_SHARED_SESSION
  std::unordered_map<string, SharedMasterSession*> sessions_ GUARDED_BY(mu_);

  // config that identify whether can share session
  struct Config {
    const ClusterDef cluster_def_;
    bool accept_shared_;

    Config(const ClusterDef& cluster_def, bool accept_shared)
        : cluster_def_(cluster_def), accept_shared_(accept_shared) {}
  };

  // use ClusterDef to identify which two Sessions can be shared
  // std::unordered_map<string, const ClusterDef> sessions_configs_
  // GUARDED_BY(mu_); std::vector<Config> session_configs_ GUARDED_BY(mu_);
  std::unordered_map<string, Config> session_configs_ GUARDED_BY(mu_);
#else
  std::unordered_map<string, MasterSession*> sessions_ GUARDED_BY(mu_);
#endif

  // Moving average of step times.
  MovingAverage last_1000_steps_ GUARDED_BY(mu_);

  // Cumulative number of steps executed.
  int64 step_count_ GUARDED_BY(mu_);

  // If a session is not active for this many seconds, it will be
  // closed automatically.
  const double session_gc_seconds_;

  // Used to track ids for incoming requests so we can detect duplicates.
  RecentRequestIds recent_request_ids_;

#ifdef _TF_USE_SHARED_SESSION
  SharedMasterSession* MayGetSharedMasterSession(
      const CreateSessionRequest* req) GUARDED_BY(mu_);
#endif

  // Call CleanupAll on all workers.
  void CleanupWorkers(const ResetRequest& reset);

  // Cleanup unused session.
  void GC();

  // Find master session by session handle, and increments the reference count
  // on the returned MasterSession if not null.
#ifdef _TF_USE_SHARED_SESSION
  SharedMasterSession* FindMasterSession(const string& handle);
#else
  MasterSession* FindMasterSession(const string& handle);
#endif

  TF_DISALLOW_COPY_AND_ASSIGN(Master);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
