#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SHARED_MASTER_SESSION_H
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SHARED_MASTER_SESSION_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/debugger_state_interface.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/common_runtime/stats_publisher_interface.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class Device;
struct MasterEnv;

// A session encapsulates a graph computation (resource allocation,
// placement, execution, etc.).
// Multiple models can run simulataneously, they share (must)
//    worker_cache_ (the optional session-specific worker cluster)
//    devices_
//    filtered_worker_list_ (names of remote worker tasks)
// DO NOT share
//    GraphExecutionState
//    worker_env (different at:)
//        local_devices
//        device_mgr (local devices manager, maybe null)
//        rendezvous_mgr (a set of rendezvous keyed by step ids)
//        collective_executor_mgr ()
//    same at:
//        Env* env
//        session_mgr
//        compute_pool TODO(px): not sure need to be diff?
class SharedMasterSession : public core::RefCounted {
 public:
  typedef std::function<void(const Status&)> DoneCallback;
  // This session encapsulates the graph computation for a graph.
  //
  // The session places nodes on devices in "remote_devs" and executes
  // operations on these devices.
  //
  // The caller takes ownership of all remote devices.
  SharedMasterSession(
      const SessionOptions& options, const MasterEnv* env,
      std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
      std::unique_ptr<WorkerCacheInterface> worker_cache,
      std::unique_ptr<DeviceSet> device_set,
      std::vector<string> filtered_worker_list,
      StatsPublisherFactory stats_publisher_factory);

  // Initialize the SharedMasterSession for "def".  Must be called before
  // Extend(), Run(), or Close().
  Status Create(GraphDef&& def, const WorkerCacheFactoryOptions& options);

  // Add another task to this shared_session
  Status AddTask(const int32 task_id, const SessionOptions& opt, GraphDef&& def,
                 const WorkerCacheFactoryOptions& options);

  // Returns the session handle.
  const string& handle() const { return handle_; }

  // Returns the last access time (the number of micro-seconds since
  // some fixed point in time) of this session.
  uint64 last_access_time_usec() const { return last_access_time_usec_.load(); }

  // Attempt to extend the graph according to the given "req".
  // (See master.proto for details of valid extensions.)
  //
  // PRECONDITION: The current version of this session's graph
  //   is "req->current_graph_version".
  //
  // POSTCONDITION: The current version of this session's graph
  //   is "resp->new_graph_version".
  //
  // Extend() may block the caller thread for a long time.
  Status Extend(const ExtendSessionRequest* req, ExtendSessionResponse* resp);

  // Setup a partial run call.
  Status PartialRunSetup(const PartialRunSetupRequest* req,
                         PartialRunSetupResponse* resp);

  // Run one step.
  Status Run(CallOptions* opts, const RunStepRequestWrapper& req,
             MutableRunStepResponseWrapper* resp);

  Status ListDevices(ListDevicesResponse* resp) const;

  Status MakeCallable(const MakeCallableRequest& req,
                      MakeCallableResponse* resp);

  Status RunCallable(CallOptions* opts, const RunCallableRequest& req,
                     RunCallableResponse* resp);

  Status ReleaseCallable(const ReleaseCallableRequest& req,
                         ReleaseCallableResponse* resp);

  // Close this session and delete "*this". Returns OK if all known
  // states are cleanup successfully.
  //
  // Close() may block the caller thread for a long time.
  Status Close(const int32 task_id);

  // Close this session and release a reference on "*this".
  //
  // Note that, unlike Close(), this method does not block on the
  // completion of all work.
  void GarbageCollect();

  int32 NumTasks() const { return num_tasks_; }

 private:
  // SessionOptions session_opts_;
  // each SessionOptions's address is referenced by an ExecutionState
  // so the resizement of this vector will change the address which will cause
  // Seg fault
  std::vector<SessionOptions*> session_opts_;

  // container name for different tasks, cleaned when a task invokes Close()
  std::vector<string> container_names_;

  // Not owned.
  const MasterEnv* env_;

  // The opaque session handle. (shared by tasks in this shared_session)
  const string handle_;

  std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs_;

  // The optional session-specific worker cluster.
  // TODO(saeta): Convert to std::optional when available.
  const std::unique_ptr<WorkerCacheInterface> worker_cache_;
  // Retrieves either worker_cache_ or the env_->worker_cache as appropriate.
  WorkerCacheInterface* get_worker_cache() const;

  // The device set used by this session.
  std::unique_ptr<DeviceSet> devices_;

  // The (partial device) names of remote worker tasks that this
  // session will contact. (px: this is shared by multiple tasks)
  const std::vector<string> filtered_worker_list_;

  StatsPublisherFactory stats_publisher_factory_;

  std::atomic_ulong last_access_time_usec_;

  std::atomic<int64> partial_run_handle_counter_ = {0};

  uint64 NewStepId(int64 graph_key);

  mutex mu_;
  // std::unique_ptr<GraphExecutionState> execution_state_ GUARDED_BY(mu_);
  std::vector<std::unique_ptr<GraphExecutionState>> execution_states_
      GUARDED_BY(mu_);
  // int64 graph_version_;
  std::vector<int64> graph_versions_;

  // We keep a map from a signature of a run request to the
  // ReffedClientGraph the can execute it.  We keep up to one old copy
  // of each ReffedClientGraph around because if it gets deallocated
  // before a new substitute has been created, Variables can go out of
  // scope and lose their state.
  class ReffedClientGraph;
  typedef std::unordered_map<uint64, ReffedClientGraph*> RCGMap;
  RCGMap run_graphs_ GUARDED_BY(mu_);
  RCGMap partial_run_graphs_ GUARDED_BY(mu_);
  int64 next_callable_handle_ GUARDED_BY(mu_) = 0;
  RCGMap callables_ GUARDED_BY(mu_);

  struct PerStepState {
    bool collect_costs = false;
    bool collect_timeline = false;
    bool collect_rpcs = false;
    bool collect_partition_graphs = false;
    bool report_tensor_allocations_upon_oom = false;
    Microseconds start_micros = Microseconds(0);
    Microseconds end_micros = Microseconds(0);
    std::vector<StepStats> step_stats;  // per partition
    StepStats rpc_stats;                // for RPC layer
    CostGraphDef cost_graph;
  };

  struct RunState {
    std::unordered_map<string, bool> pending_inputs;   // true if fed
    std::unordered_map<string, bool> pending_outputs;  // true if fetched
    ReffedClientGraph* rcg = nullptr;
    uint64 step_id;
    int64 collective_graph_key;
    int64 count = 0;
    PerStepState pss;
    std::unique_ptr<ProfileHandler> ph;
    bool step_started = false;

    RunState(const std::vector<string>& input_names,
             const std::vector<string>& output_names, ReffedClientGraph* rcg,
             const uint64 step_id, const int64 count);

    bool PendingDone() const;

    ~RunState();
  };

  struct Task {
    const int32 task_handle_;  // a unique identifier for a task
    std::mutex mu_;
    bool is_running_ GUARDED_BY(mu_) = true;  // mark this as running
    std::condition_variable task_running_ GUARDED_BY(mu_);

    // parameter for one step run, not own
    CallOptions* call_opts;
    const RunStepRequestWrapper* req;
    MutableRunStepResponseWrapper* resp;

    Task(const int32 task_handle, CallOptions* call_opts,
         const RunStepRequestWrapper* req, MutableRunStepResponseWrapper* resp);

    Task(Task&&);

    ~Task() {}

    const int32 handle() const { return task_handle_; }

    void Renew(CallOptions* call_opts, const RunStepRequestWrapper* req,
               MutableRunStepResponseWrapper* resp);

    TF_DISALLOW_COPY_AND_ASSIGN(Task);
  };

  friend void DebugReadyTasks(const std::vector<Task*>* ready_tasks);

  std::unordered_map<string, std::unique_ptr<RunState>> partial_runs_
      GUARDED_BY(mu_);

  // num tasks that share this session
  int32 num_tasks_;

  int32 num_remain_tasks_;

  // Active RunStep calls.
  condition_variable num_running_is_zero_;
  std::vector<int32> num_runnings_ GUARDED_BY(mu_);

  bool closed_ GUARDED_BY(mu_) = false;
  bool garbage_collected_ GUARDED_BY(mu_) = false;

  std::unordered_map<uint64, int64> subgraph_execution_counts_ GUARDED_BY(mu_);

  // We need to ensure that certain nodes added (e.g., send and recv
  // nodes) are unique across all sub-graphs within this session.
  int64 next_node_id_ GUARDED_BY(mu_) = 0;

  // Used to cancel running steps on Close().
  CancellationManager cancellation_manager_;

  // Task related
  mutex tasks_mu_;

  // for each task, use one Task to describe one step's execution
  std::unordered_map<int32, Task*> tasks_ GUARDED_BY(tasks_mu_);

  std::vector<Task*> ready_tasks_ GUARDED_BY(tasks_mu_);

  bool aligned_scheduling;

  bool stop_polling_ GUARDED_BY(tasks_mu_);
  std::unique_ptr<Notification> polling_stopped_;

  condition_variable tasks_pending_ GUARDED_BY(tasks_mu_);
  thread::ThreadPool tasks_pool_;

  void PollTasks() EXCLUSIVE_LOCKS_REQUIRED(tasks_mu_);

  void ScheduleLoop();

  void StartScheduleLoop();
  void StopScheduleLoop();

  // Private dtor. The client must call Close().
  virtual ~SharedMasterSession();

  // Creates sessions on all workers.
  //
  // If this session is operating using the new ClusterSpec propagation behavior
  // call this method in order to propagate the cluster membership to all
  // workers.
  Status CreateWorkerSessions(const int32 task_id,
                              const WorkerCacheFactoryOptions& server_def);

  bool should_delete_worker_sessions_ = false;
  Status DeleteWorkerSessions();

  Status StartStep(const int32 task_id, const BuildGraphOptions& opts,
                   bool is_partial, ReffedClientGraph** out_rcg,
                   int64* out_count);
  void ClearRunsTable(std::vector<ReffedClientGraph*>* to_unref,
                      RCGMap* rcg_map) EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void FillPerStepState(const int32 task_id,
                        SharedMasterSession::ReffedClientGraph* rcg,
                        const RunOptions& run_options, uint64 step_id,
                        int64 count, PerStepState* out_pss,
                        std::unique_ptr<ProfileHandler>* out_ph);
  Status DoRunWithLocalExecution(CallOptions* opts,
                                 const RunStepRequestWrapper& req,
                                 MutableRunStepResponseWrapper* resp);
  Status DoPartialRun(CallOptions* opts, const RunStepRequestWrapper& req,
                      MutableRunStepResponseWrapper* resp);
  Status DoRunCallable(CallOptions* opts, ReffedClientGraph* rcg,
                       const RunCallableRequest& req,
                       RunCallableResponse* resp);
  Status PostRunCleanup(const int32 task_id,
                        SharedMasterSession::ReffedClientGraph* rcg,
                        uint64 step_id, const RunOptions& run_options,
                        PerStepState* pss,
                        const std::unique_ptr<ProfileHandler>& ph,
                        const Status& run_status,
                        RunMetadata* out_run_metadata);

  void MarkRunCompletion(const int32 task_id);
  void UpdateLastAccessTime();

  void CleanupContainerResource(const int32 task_id);

  Status BuildAndRegisterPartitions(const int32 task_id,
                                    ReffedClientGraph* rcg);

  Status CreateDebuggerState(
      const DebugOptions& debug_options, const RunStepRequestWrapper& req,
      int64 rcg_execution_count,
      std::unique_ptr<DebuggerStateInterface>* debugger_state);

  void RunWithCallback(CallOptions* opts, const RunStepRequestWrapper* req,
                       MutableRunStepResponseWrapper* resp, DoneCallback done);

  TF_DISALLOW_COPY_AND_ASSIGN(SharedMasterSession);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SHARED_MASTER_SESSION_H
