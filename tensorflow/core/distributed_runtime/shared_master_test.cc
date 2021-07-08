#include "tensorflow/core/distributed_runtime/shared_master_session.h"

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "grpcpp/grpcpp.h"

#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_server_lib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

static bool _is_allowed_shared;

class MasterTest : public ::testing::Test {
 protected:
  MasterTest() {
    ReadBoolFromEnvVar("TF_IS_ALLOWED_SHARED_SESSION", false,
                       &_is_allowed_shared);
    string targets;
    SessionOptions options;
    (*options.config.mutable_device_count())["CPU"] = 1;
    (*options.config.mutable_device_count())["GPU"] = 0;
    TF_CHECK_OK(test::TestCluster::MakeTestCluster(options, 1, &cluster_));
    SharedGrpcChannelPtr channel_ptr;
    RPCOptions rpc_opts;
    // TF_CHECK_OK(NewHostPortGrpcChannel(cluster_->targets()[0],
    // &channel_ptr));
    TF_CHECK_OK(NewHostPortGrpcChannel(cluster_->targets()[0], &rpc_opts,
                                       &channel_ptr));
    master_ = grpc::MasterService::NewStub(channel_ptr);
  }

  std::unique_ptr<test::TestCluster> cluster_;
  std::unique_ptr<grpc::MasterService::Stub> master_;

  // Helpers for MasterService.{CreateSession,RunStep,CloseSession}
  // rpc calls.

  Status CreateSession(const GraphDef& def, string* handle,
                       int64* initial_version) {
    ::grpc::ClientContext ctx;
    CreateSessionRequest req;
    *(req.mutable_graph_def()) = def;
    // Invokes placement frequently.
    req.mutable_config()->set_placement_period(1);
    req.mutable_config()->mutable_gpu_options()->set_allow_shared(
        _is_allowed_shared);
    CreateSessionResponse resp;
    const Status s = FromGrpcStatus(master_->CreateSession(&ctx, req, &resp));
    if (s.ok()) {
      *handle = resp.session_handle();
      *initial_version = resp.graph_version();
    }
    return s;
  }

  Status ExtendSession(const string& handle, const GraphDef& def,
                       int64 current_version, int64* new_version) {
    ::grpc::ClientContext ctx;
    ExtendSessionRequest req;
    req.set_session_handle(handle);
    *(req.mutable_graph_def()) = def;
    req.set_current_graph_version(current_version);
    ExtendSessionResponse resp;
    const Status s = FromGrpcStatus(master_->ExtendSession(&ctx, req, &resp));
    if (s.ok()) {
      *new_version = resp.new_graph_version();
    }
    return s;
  }

  Status RunStep(const string& handle,
                 const std::vector<std::pair<string, const Tensor*>>& feed,
                 const std::map<string, Tensor*>& fetch) {
    ::grpc::ClientContext ctx;
    RunStepRequest req;
    req.set_session_handle(handle);
    for (const auto& p : feed) {
      const string& feed_name = p.first;
      const Tensor* feed_tensor = p.second;
      auto f = req.add_feed();
      f->set_name(feed_name);
      feed_tensor->AsProtoTensorContent(f->mutable_tensor());
    }
    for (const auto& p : fetch) {
      const string& fetch_name = p.first;
      req.add_fetch(fetch_name);
    }
    RunStepResponse resp;
    const Status s = FromGrpcStatus(master_->RunStep(&ctx, req, &resp));
    if (s.ok()) {
      for (const auto& fetch_resp : resp.tensor()) {
        auto it = fetch.find(fetch_resp.name());
        CHECK(it != fetch.end());
        CHECK(it->second->FromProto(fetch_resp.tensor()));
      }
    }
    return s;
  }

  Status CloseSession(const string& handle) {
    ::grpc::ClientContext ctx;
    CloseSessionRequest req;
    req.set_session_handle(handle);
    CloseSessionResponse resp;
    return FromGrpcStatus(master_->CloseSession(&ctx, req, &resp));
  }

  Status Reset() {
    ::grpc::ClientContext ctx;
    ResetRequest req;
    ResetResponse resp;
    return FromGrpcStatus(master_->Reset(&ctx, req, &resp));
  }
};

// Success
TEST_F(MasterTest, ConcurrentCreate) {
  int n = 2;
  std::vector<GraphDef> graph_defs(n);
  std::vector<string> handles(n);
  std::vector<int64> initial_versions(n);

  auto create_fn = [this, &graph_defs, &handles, &initial_versions](int i) {
    TF_ASSERT_OK(
        CreateSession(graph_defs[i], &handles[i], &initial_versions[i]));
  };

  {
    thread::ThreadPool thread_pool(Env::Default(), "create_pool", int(n / 2));
    for (int i = 0; i < n; ++i) {
      auto create_i_fn = [this, i, &create_fn]() { create_fn(i); };
      thread_pool.Schedule(create_i_fn);
    }
  }

  string trucated_handle1, trucated_handle2;
  for (int i = 0; i < n - 1; ++i) {
    trucated_handle1 = handles[i].substr(0, handles[i].length() - 2);
    trucated_handle2 = handles[i + 1].substr(0, handles[i + 1].length() - 2);
    EXPECT_EQ(trucated_handle1, trucated_handle2);
  }

  EXPECT_TRUE(CloseSession(handles[0]).ok());
}

TEST_F(MasterTest, ConcurrentRun) {
  Graph graph_0(OpRegistry::Global());
  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  test::graph::Constant(&graph_0, a_tensor, "A");
  GraphDef def_0;
  test::graph::ToGraphDef(&graph_0, &def_0);

  Graph graph_1(OpRegistry::Global());
  Tensor b_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&b_tensor, {1, 0, 0, 1});
  test::graph::Constant(&graph_1, b_tensor, "B");
  GraphDef def_1;
  test::graph::ToGraphDef(&graph_1, &def_1);

  string handle1, handle2;
  int64 initial_version1, initial_version2;
  TF_ASSERT_OK(CreateSession(def_0, &handle1, &initial_version1));
  TF_ASSERT_OK(CreateSession(def_1, &handle2, &initial_version2));
  string trucated_handle1 = handle1.substr(0, handle1.length() - 2);
  string trucated_handle2 = handle2.substr(0, handle2.length() - 2);
  EXPECT_EQ(trucated_handle1, trucated_handle2);

  auto get_a_fn = [this, handle1, &a_tensor]() {
    Tensor A(DT_FLOAT, TensorShape({2, 2}));
    TF_ASSERT_OK(RunStep(handle1, {}, {{"A:0", &A}}));
    test::ExpectTensorEqual<float>(A, a_tensor);
  };

  auto get_b_fn = [this, handle2, &b_tensor]() {
    Tensor B(DT_FLOAT, TensorShape({2, 2}));
    TF_ASSERT_OK(RunStep(handle2, {}, {{"B:0", &B}}));
    test::ExpectTensorEqual<float>(B, b_tensor);
  };

  {
    thread::ThreadPool thread_pool(Env::Default(), "compute_pool", 2);
    thread_pool.Schedule(get_a_fn);
    thread_pool.Schedule(get_b_fn);
  }

  EXPECT_TRUE(CloseSession(handle1).ok());
}

TEST_F(MasterTest, CreateClose) {
  GraphDef def;  // Empty.
  string handle;
  int64 initial_version;
  TF_ASSERT_OK(CreateSession(def, &handle, &initial_version));
  EXPECT_TRUE(errors::IsAborted(CloseSession("randombits")));
  EXPECT_TRUE(CloseSession(handle).ok());
}

TEST_F(MasterTest, ListDevices) {
  ::grpc::ClientContext ctx;
  ListDevicesRequest req;
  ListDevicesResponse resp;
  const Status s = FromGrpcStatus(master_->ListDevices(&ctx, req, &resp));
  TF_EXPECT_OK(s);
  EXPECT_EQ(1, resp.local_device_size());
  EXPECT_EQ("CPU", resp.local_device(0).device_type());
}

TEST_F(MasterTest, Reset) {
  GraphDef def;  // Empty.
  string s1, s2;
  int64 initial_version1, initial_version2;
  TF_ASSERT_OK(CreateSession(def, &s1, &initial_version1));
  TF_ASSERT_OK(CreateSession(def, &s2, &initial_version2));
  EXPECT_TRUE(Reset().ok());
  EXPECT_TRUE(errors::IsAborted(CloseSession(s1)));
  EXPECT_TRUE(errors::IsAborted(CloseSession(s2)));
}

TEST_F(MasterTest, Extend) {
  GraphDef def_0;  // Empty.
  string handle;
  int64 initial_version;
  TF_ASSERT_OK(CreateSession(def_0, &handle, &initial_version));

  Tensor A_expected(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&A_expected, {3.0, 2.0, -1.0, 0.0});

  Tensor x_expected(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_expected, {2.0, 2.0});

  Graph graph_1(OpRegistry::Global());
  test::graph::Constant(&graph_1, A_expected, "A");
  GraphDef def_1;
  test::graph::ToGraphDef(&graph_1, &def_1);
  int64 version_1;
  TF_ASSERT_OK(ExtendSession(handle, def_1, initial_version, &version_1));
  EXPECT_GT(version_1, initial_version);
  Tensor A(DT_FLOAT, TensorShape({2, 2}));
  TF_ASSERT_OK(RunStep(handle, {}, {{"A:0", &A}}));
  test::ExpectTensorEqual<float>(A, A_expected);

  Graph graph_2(OpRegistry::Global());
  test::graph::Constant(&graph_2, x_expected, "x");
  GraphDef def_2;
  test::graph::ToGraphDef(&graph_2, &def_2);
  int64 version_2;
  EXPECT_TRUE(errors::IsAborted(
      ExtendSession("randombits", def_2, version_1, &version_2)));
  TF_ASSERT_OK(ExtendSession(handle, def_2, version_1, &version_2));
  EXPECT_GT(version_2, version_1);

  Tensor x(DT_FLOAT, TensorShape({2, 1}));
  TF_ASSERT_OK(RunStep(handle, {}, {{"A:0", &A}, {"x:0", &x}}));
  test::ExpectTensorEqual<float>(A, A_expected);
  test::ExpectTensorEqual<float>(x, x_expected);

  TF_ASSERT_OK(CloseSession(handle));
}

TEST_F(MasterTest, ExtendUpdateStatefulFails) {
  GraphDef def_0;  // Empty.
  string handle;
  int64 initial_version;
  TF_ASSERT_OK(CreateSession(def_0, &handle, &initial_version));

  Graph graph_1(OpRegistry::Global());
  test::graph::Var(&graph_1, DT_FLOAT, TensorShape({512}));
  GraphDef def_1;
  test::graph::ToGraphDef(&graph_1, &def_1);

  int64 version_1, version_2;
  TF_ASSERT_OK(ExtendSession(handle, def_1, initial_version, &version_1));
  EXPECT_GT(version_1, initial_version);
  EXPECT_TRUE(errors::IsInvalidArgument(
      ExtendSession(handle, def_1, version_1, &version_2)));
  TF_ASSERT_OK(CloseSession(handle));
}

TEST_F(MasterTest, ExtendTwiceFails) {
  GraphDef def_0;  // Empty.
  string handle;
  int64 initial_version;
  TF_ASSERT_OK(CreateSession(def_0, &handle, &initial_version));

  Graph graph_1(OpRegistry::Global());
  test::graph::Var(&graph_1, DT_FLOAT, TensorShape({512}));
  GraphDef def_1;
  test::graph::ToGraphDef(&graph_1, &def_1);

  int64 version_1;
  TF_ASSERT_OK(ExtendSession(handle, def_1, initial_version, &version_1));
  EXPECT_GT(version_1, initial_version);
  EXPECT_TRUE(errors::IsAborted(
      ExtendSession(handle, def_1, initial_version, &version_1)));
  TF_ASSERT_OK(CloseSession(handle));
}

TEST_F(MasterTest, ConcurrentExtendOnlyOneSucceeds) {
  GraphDef def_0;  // Empty.
  string handle;
  int64 initial_version;
  TF_ASSERT_OK(CreateSession(def_0, &handle, &initial_version));

  Graph graph_1(OpRegistry::Global());
  test::graph::Var(&graph_1, DT_FLOAT, TensorShape({512}));
  GraphDef def_1;
  test::graph::ToGraphDef(&graph_1, &def_1);

  Notification n;
  mutex mu;
  int succeeded = 0;
  int failed = 0;
  auto extend_fn = [this, handle, def_1, initial_version, &n, &mu, &succeeded,
                    &failed]() {
    n.WaitForNotification();
    int64 new_version;
    Status s = ExtendSession(handle, def_1, initial_version, &new_version);
    EXPECT_TRUE(s.ok() || errors::IsAborted(s));
    {
      mutex_lock l(mu);
      if (s.ok()) {
        ++succeeded;
      } else {
        ++failed;
      }
    }
  };

  // Run 100 concurrent Extend calls and expect only one to succeed.
  {
    thread::ThreadPool thread_pool(Env::Default(), "extend_pool", 100);
    for (int i = 0; i < 100; ++i) {
      thread_pool.Schedule(extend_fn);
    }
    n.Notify();
  }

  EXPECT_EQ(failed, 99);
  EXPECT_EQ(succeeded, 1);
  TF_ASSERT_OK(CloseSession(handle));
}

TEST_F(MasterTest, ConcurrentExtendAndRun) {
  Graph graph_0(OpRegistry::Global());
  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  test::graph::Constant(&graph_0, a_tensor, "A");
  GraphDef def_0;
  test::graph::ToGraphDef(&graph_0, &def_0);

  string handle;
  int64 initial_version;
  TF_ASSERT_OK(CreateSession(def_0, &handle, &initial_version));

  Graph graph_1(OpRegistry::Global());
  Tensor b_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&b_tensor, {1, 0, 0, 1});
  test::graph::Constant(&graph_1, b_tensor, "B");
  GraphDef def_1;
  test::graph::ToGraphDef(&graph_1, &def_1);

  Notification extend_done;
  Notification extend_can_start;

  auto get_a_fn = [this, handle, &extend_done]() {
    Tensor A(DT_FLOAT, TensorShape({2, 2}));
    while (!extend_done.HasBeenNotified()) {
      TF_ASSERT_OK(RunStep(handle, {}, {{"A:0", &A}}));
    }
    // Run at least once after the Extend has completed.
    TF_ASSERT_OK(RunStep(handle, {}, {{"A:0", &A}}));
  };

  auto get_a_and_b_fn = [this, handle, &extend_done, &extend_can_start]() {
    Tensor A(DT_FLOAT, TensorShape({2, 2}));
    Tensor B(DT_FLOAT, TensorShape({2, 2}));

    // Run at least once before the Extend has completed.
    EXPECT_TRUE(
        errors::IsNotFound(RunStep(handle, {}, {{"A:0", &A}, {"B:0", &B}})));
    extend_can_start.Notify();

    // Concurrent with the Extend, we will either fail (as above), or
    // succeed (as below).
    while (!extend_done.HasBeenNotified()) {
      Status s = RunStep(handle, {}, {{"A:0", &A}, {"B:0", &B}});
      EXPECT_TRUE(errors::IsNotFound(s) || s.ok());
    }

    // Run at least once after the Extend has completed.
    TF_ASSERT_OK(RunStep(handle, {}, {{"A:0", &A}, {"B:0", &B}}));
  };

  auto extend_fn = [this, handle, def_1, initial_version, &extend_done,
                    &extend_can_start]() {
    extend_can_start.WaitForNotification();
    int64 version_1;
    TF_ASSERT_OK(ExtendSession(handle, def_1, initial_version, &version_1));
    extend_done.Notify();
  };

  {
    thread::ThreadPool thread_pool(Env::Default(), "extend_pool", 3);
    thread_pool.Schedule(get_a_fn);
    thread_pool.Schedule(get_a_and_b_fn);
    thread_pool.Schedule(extend_fn);
  }

  TF_ASSERT_OK(CloseSession(handle));
}

TEST_F(MasterTest, EigenProblem) {
  // A = [3 2; -1 0]; x = rand(2, 1);
  // for i=1:100; x = A * x; end
  // We'll try to compute the largest eigenvalue for A.
  Graph graph(OpRegistry::Global());
  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  // Store rows [3, 2] and [-1, 0] in row major format.
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  Node* a_node = test::graph::Constant(&graph, a_tensor);

  // x is from the feed.
  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_tensor, {0, 0});
  Node* x_node = test::graph::Constant(&graph, x_tensor);

  // y = A * x
  Node* y_node = test::graph::Matmul(&graph, a_node, x_node, false, false);

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  string handle;
  int64 initial_version;
  TF_CHECK_OK(CreateSession(def, &handle, &initial_version));

  // Temps supporting the computation of the convergence condition.
  const Eigen::array<Eigen::DenseIndex, 1> sum_along_dim(0);
  const Eigen::array<Eigen::DenseIndex, 2> matrix_transpose({1, 0});
  Tensor x(DT_FLOAT, TensorShape({2, 1}));
  Tensor y(DT_FLOAT, TensorShape({2, 1}));
  Eigen::Tensor<float, 1, Eigen::RowMajor> y_square_sum;
  Eigen::Tensor<float, 2, Eigen::RowMajor> y_normalized(2, 1);
  y_normalized.setRandom();
  Eigen::Tensor<float, 1, Eigen::RowMajor> error_square_sum;
  float lambda;

  // The computation loop.
  bool converged = false;
  while (!converged) {
    // Run one step of the graph.
    auto x_matrix = x.matrix<float>();
    x_matrix = y_normalized;
    TF_EXPECT_OK(
        RunStep(handle, {{x_node->name(), &x}}, {{y_node->name() + ":0", &y}}));
    auto y_matrix = y.matrix<float>();

    // Client code computes the convergence condition.
    {
      lambda = y_matrix(0, 0) / x_matrix(0, 0);
      y_square_sum = y.matrix<float>().square().sum(sum_along_dim);
      const float norm = static_cast<float>(sqrt(y_square_sum(0)));
      y_normalized = y_matrix * (1 / norm);
      error_square_sum = (x_matrix - y_normalized).square().sum(sum_along_dim);
      VLOG(1) << "x = [" << x_matrix.shuffle(matrix_transpose) << "] y = ["
              << y_matrix.shuffle(matrix_transpose) << "] lambda = " << lambda;
      converged = sqrt(error_square_sum(0)) < 1e-10;
    }
  }
  EXPECT_NEAR(lambda, 2.0, 0.01);
  TF_EXPECT_OK(CloseSession(handle));
}

// TEST(SharedMasterTest, ConcurrentRun1) {
//   ReadBoolFromEnvVar("TF_IS_ALLOWED_SHARED_SESSION", false,
//   &_is_allowed_shared); string targets; SessionOptions options;
//   (*options.config.mutable_device_count())["CPU"] = 1;
//   (*options.config.mutable_device_count())["GPU"] = 0;

//   // cluster config
//   int num_cluster = 1;

//   std::vector<std::unique_ptr<SubProcess>> subprocesses_;
//   std::vector<string> targets_(num_cluster);
//   std::vector<DeviceAttributes> devices_;
//   const string binary_path =
//   "/home/hp/px/tensorflow-1.15.2/bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_testlib_server";

//   std::vector<int> port(num_cluster);
//   for (int i = 0; i < num_cluster; ++i) {
//     port[i] = testing::PickUnusedPortOrDie();
//     targets_[i] = strings::StrCat("localhost:", port[i]);
//   }

//   const string tf_jobs = strings::StrCat("--tf_jobs=localhost|",
//                                          absl::StrJoin(targets_, ";"));

//   int num_cpus = 1;
//   int num_gpus = 0;
//   for (int i = 0; i < num_cluster; ++i) {
//     if (!options.env->FileExists(binary_path).ok()) {
//       LOG(ERROR) << "Could not find grpc_testlib_server";
//       exit(1);
//     }
//     const std::vector<string> argv(
//         {binary_path, /* see grpc_testlib_server.cc for flags */
//          tf_jobs, "--tf_job=localhost", strings::StrCat("--tf_task=", i),
//          strings::StrCat("--num_cpus=", num_cpus),
//          strings::StrCat("--num_gpus=", num_gpus)});
//     subprocesses_.emplace_back(CreateSubProcess(argv));
//     bool success = subprocesses_[i]->Start();
//     if (!success) {
//       LOG(ERROR) << "Could not start subprocess";
//       exit(1);
//     }
//   }

//   options.target = strings::StrCat("grpc://", targets_[0]);
//   options.accept_shared = _is_allowed_shared;
//   std::unique_ptr<GrpcSession> session_1, session_2;
//   TF_ASSERT_OK(GrpcSession::Create(options, &session_1));

//   SessionOptions options_2(options);
//   options_2.accept_shared = _is_allowed_shared;
//   TF_ASSERT_OK(GrpcSession::Create(options_2, &session_2));

//   // shared session create test
//   GraphDef def_1, def_2;
//   session_1->Create(def_1);
//   session_2->Create(def_2);

//   // CHECK
//   if (options.accept_shared && options_2.accept_shared) {
//     string trucated_handle1 = session_1->handle().substr(0,
//     session_1->handle().length()-2); string trucated_handle2 =
//     session_2->handle().substr(0, session_2->handle().length()-2);
//     EXPECT_EQ(trucated_handle1, trucated_handle2);
//   }

//   // Graph definition
//   Graph graph_1(OpRegistry::Global());
//   Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
//   test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
//   test::graph::Constant(&graph_1, a_tensor, "A");
//   test::graph::ToGraphDef(&graph_1, &def_1);

//   Graph graph_2(OpRegistry::Global());
//   Tensor b_tensor(DT_FLOAT, TensorShape({2, 2}));
//   test::FillValues<float>(&b_tensor, {1, 0, 0, 1});
//   test::graph::Constant(&graph_2, b_tensor, "B");
//   test::graph::ToGraphDef(&graph_2, &def_2);

//   // Extend
//   session_1->Extend(def_1);
//   session_2->Extend(def_2);

//   // Run
//   std::vector<std::pair<string, Tensor>> inputs_1, inputs_2;
//   std::vector<string> output_tensor_names_1, output_tensor_names_2;
//   std::vector<string> target_node_names_1, target_node_names_2;
//   std::vector<Tensor> outputs_1, outputs_2;

//   Tensor A(DT_FLOAT, TensorShape({2, 2}));
//   output_tensor_names_1.emplace_back("A:0");
//   outputs_1.emplace_back(A);
//   Tensor B(DT_FLOAT, TensorShape({2, 2}));
//   output_tensor_names_2.emplace_back("B:0");
//   outputs_2.emplace_back(B);

//   session_1->Run(inputs_1, output_tensor_names_1, target_node_names_1,
//   &outputs_1); session_2->Run(inputs_2, output_tensor_names_2,
//   target_node_names_2, &outputs_2);

//   for (auto& subprocess: subprocesses_) {
//     subprocess->Kill(9);
//   }
// }

// namespace {

// Status FillServerDef(const string& job_spec, const string& job_name,
//                      int num_cpus, int num_gpus, int task_index,
//                      ServerDef* options) {
//   options->set_protocol("grpc");
//   options->set_job_name(job_name);
//   options->set_task_index(task_index);

//   uint32 my_tasks_per_replica = 0;
//   for (const string& job_str : str_util::Split(job_spec, ',')) {
//     JobDef* job_def = options->mutable_cluster()->add_job();
//     // Split each entry in the flag into 2 pieces, separated by "|".
//     const std::vector<string> job_pieces = str_util::Split(job_str, '|');
//     CHECK_EQ(2, job_pieces.size()) << job_str;
//     job_def->set_name(job_pieces[0]);
//     // Does a bit more validation of the tasks_per_replica.
//     const StringPiece spec = job_pieces[1];
//     // job_str is of form <job_name>|<host_ports>.
//     const std::vector<string> host_ports = str_util::Split(spec, ';');
//     uint32 tasks_per_replica = host_ports.size();
//     for (size_t i = 0; i < host_ports.size(); ++i) {
//       (*job_def->mutable_tasks())[i] = host_ports[i];
//     }
//     if (job_def->name() == options->job_name()) {
//       my_tasks_per_replica = tasks_per_replica;
//     }
//     LOG(INFO) << "Peer " << job_def->name() << " " << tasks_per_replica << "
//     {"
//               << absl::StrJoin(host_ports, ", ") << "}";
//   }
//   if (my_tasks_per_replica == 0) {
//     return errors::InvalidArgument("Invalid job specification");
//   }
//   ConfigProto* config = options->mutable_default_session_config();
//   (*config->mutable_device_count())["CPU"] = num_cpus;
//   (*config->mutable_device_count())["GPU"] = num_gpus;
//   return Status::OK();
// }

// }  // namespace

// class SharedMasterTest : public ::testing::Test {
//  protected:
//   SharedMasterTest() {
//     int port = testing::PickUnusedPortOrDie();
//     string target = strings::StrCat("localhost:", port);

//     int num_cpus = 1;
//     int num_gpus = 0;
//     int task_index = 0;

//     string job_spec = strings::StrCat("localhost|", target);
//     string job_name = "localhost";

//     Status s = FillServerDef(job_spec, job_name, num_cpus, num_gpus,
//     task_index, &def); if (!s.ok()) {
//       LOG(ERROR) << "Could not parse job spec: " << s.error_message() <<
//       "\n";
//     }

//     s = GrpcMasterServer::Create(def, nullptr, &server_);

//     if (!s.ok()) {
//       LOG(ERROR) << "Could not create server: " << s.error_message();
//     }

//     server_thread_.reset(
//         Env::Default()->StartThread(ThreadOptions(), "TF_GRPC_MASTER_SERVER",
//                                 [this] {
//                                   server_->Start();
//                                   server_->Join();}));
//   }

//   ~SharedMasterTest() {
//     server_->worker_service()->Shutdown();
//     delete server_.get();
//     server_.reset();
//     server_thread_.reset();
//   }

//   void CreateSession(const GraphDef& def, string* handle,
//                      int64* initial_version, bool accept_shared=false) {
//     CreateSessionRequest req;
//     req.set_accept_shared(accept_shared);
//     *(req.mutable_graph_def()) = def;
//     // Invokes placement frequently.
//     req.mutable_config()->set_placement_period(1);
//     // GraphOptions* graph_options =
//     req.mutable_config()->mutable_graph_options();
//     // RewriterConfig* rewriter_config =
//     graph_options->mutable_rewrite_options();

//     CreateSessionResponse resp;

//     std::mutex mu;
//     std::condition_variable cv;
//     bool done = false;
//     auto cb = [&](const Status& s) {
//       if (!s.ok()) {
//         LOG(ERROR) << s.error_message();
//         return;
//       }
//       std::unique_lock<std::mutex> l(mu);
//       done = true;
//       cv.notify_all();
//     };
//     server_->master_impl()->CreateSession(&req, &resp, cb);

//     std::unique_lock<std::mutex> l(mu);
//     cv.wait(l, [&] { return done; });

//     *handle = resp.session_handle();
//     *initial_version = resp.graph_version();
//   }

//   void ExtendSession(const string& handle, const GraphDef& def,
//                      int64 current_version, int64* new_version) {
//     ExtendSessionRequest req;
//     req.set_session_handle(handle);
//     *(req.mutable_graph_def()) = def;
//     req.set_current_graph_version(current_version);
//     ExtendSessionResponse resp;

//     std::mutex mu;
//     std::condition_variable cv;
//     bool done = false;
//     auto cb = [&](const Status& s) {
//       if (!s.ok()) {
//         LOG(ERROR) << s.error_message();
//         return;
//       }
//       std::unique_lock<std::mutex> l(mu);
//       done = true;
//       cv.notify_all();
//     };
//     server_->master_impl()->ExtendSession(&req, &resp, cb);

//     std::unique_lock<std::mutex> l(mu);
//     cv.wait(l, [&] { return done; });

//     *new_version = resp.new_graph_version();
//   }

//   void RunStep(const string& handle,
//                const std::vector<std::pair<string, const Tensor*>>& feed,
//                const std::map<string, Tensor*>& fetch) {
//     RunStepRequest req;
//     req.set_session_handle(handle);
//     for (const auto& p : feed) {
//       const string& feed_name = p.first;
//       const Tensor* feed_tensor = p.second;
//       auto f = req.add_feed();
//       f->set_name(feed_name);
//       feed_tensor->AsProtoTensorContent(f->mutable_tensor());
//     }
//     for (const auto& p : fetch) {
//       const string& fetch_name = p.first;
//       req.add_fetch(fetch_name);
//     }
//     RunStepResponse resp;
//     CallOptions* call_opts = new CallOptions;
//     if (req.options().timeout_in_ms() > 0) {
//       call_opts->SetTimeout(req.options().timeout_in_ms());
//     } else {
//       call_opts->SetTimeout(def.default_session_config().operation_timeout_in_ms());
//     }

//     std::mutex mu;
//     std::condition_variable cv;
//     bool done = false;
//     auto cb = [&](const Status& s) {
//       if (!s.ok()) {
//         LOG(ERROR) << s.error_message();
//         return;
//       }
//       std::unique_lock<std::mutex> l(mu);
//       done = true;
//       cv.notify_all();
//     };
//     RunStepRequestWrapper* wrapped_request =
//         new ProtoRunStepRequest(&req);
//     MutableRunStepResponseWrapper* wrapped_response =
//         new NonOwnedProtoRunStepResponse(&resp);
//     server_->master_impl()->RunStep(call_opts, wrapped_request,
//     wrapped_response, cb);

//     std::unique_lock<std::mutex> l(mu);
//     cv.wait(l, [&] { return done; });

//     for (const auto& fetch_resp : resp.tensor()) {
//       auto it = fetch.find(fetch_resp.name());
//       CHECK(it != fetch.end());
//       CHECK(it->second->FromProto(fetch_resp.tensor()));
//     }

//     delete call_opts;
//     delete wrapped_request;
//     delete wrapped_response;
//   }

//   ServerDef def;
//   int num_cluster = 1;
//   std::unique_ptr<GrpcMasterServer> server_;
//   std::unique_ptr<Thread> server_thread_;
// };

// TEST_F(SharedMasterTest, Concurrent) {
//   ReadBoolFromEnvVar("TF_IS_ALLOWED_SHARED_SESSION", false,
//   &_is_allowed_shared); LOG(INFO) << "Set allow shared session to: " <<
//   _is_allowed_shared; GraphDef def_0; string handle_0; int64
//   initial_version_0; CreateSession(def_0, &handle_0, &initial_version_0,
//   _is_allowed_shared); LOG(INFO) << "Session handle: " << handle_0 << ",
//   initial_version: " << initial_version_0;

//   GraphDef def_1;
//   string handle_1;
//   int64 initial_version_1;
//   CreateSession(def_1, &handle_1, &initial_version_1, _is_allowed_shared);
//   LOG(INFO) << "Session handle: " << handle_1 << ", initial_version: " <<
//   initial_version_1;

//   Graph graph_0(OpRegistry::Global());
//   Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
//   test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
//   test::graph::Constant(&graph_0, a_tensor, "A");
//   test::graph::ToGraphDef(&graph_0, &def_0);
//   int64 version_01;
//   ExtendSession(handle_0, def_0, initial_version_0, &version_01);
//   EXPECT_GT(version_01, initial_version_0);
//   LOG(INFO) << "Extend " << handle_0 << ", new graph version: " <<
//   version_01;
//   // CreateSession(def_0, &handle_0, &initial_version_0, _is_allowed_shared);
//   // LOG(INFO) << "Session handle: " << handle_0 << ", initial_version: " <<
//   initial_version_0;

//   Graph graph_1(OpRegistry::Global());
//   Tensor b_tensor(DT_FLOAT, TensorShape({2, 2}));
//   test::FillValues<float>(&b_tensor, {1, 0, 0, 1});
//   test::graph::Constant(&graph_1, b_tensor, "B");
//   test::graph::ToGraphDef(&graph_1, &def_1);
//   int64 version_11;
//   ExtendSession(handle_1, def_1, initial_version_1, &version_11);
//   EXPECT_GT(version_11, initial_version_1);
//   LOG(INFO) << "Extend " << handle_1 << ", new graph version: " <<
//   version_11;
//   // CreateSession(def_1, &handle_1, &initial_version_1, _is_allowed_shared);
//   // LOG(INFO) << "Session handle: " << handle_1 << ", initial_version: " <<
//   initial_version_1;

//   Tensor A(DT_FLOAT, TensorShape({2, 2}));
//   RunStep(handle_0, {}, {{"A:0", &A}});
//   test::ExpectTensorEqual<float>(A, a_tensor);

//   Tensor B(DT_FLOAT, TensorShape({2, 2}));
//   RunStep(handle_1, {}, {{"B:0", &B}});
//   test::ExpectTensorEqual<float>(B, b_tensor);
// }

}  // namespace tensorflow