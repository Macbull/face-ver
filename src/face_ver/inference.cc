/* Copyright 2016 Google Inc. All Rights Reserved.

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

// A gRPC server that classifies images into digit 0-9.
// Given each request with an image pixels encoded as floats, the server
// responds with 10 float values as probabilities for digit 0-9 respectively.
// The classification is done by running image data through a simple softmax
// regression network trained and exported by mnist_export.py.
// The intention of this example to demonstrate usage of Tensorflow
// APIs in an end-to-end scenario.

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpc/include/grpc++/security/server_credentials.h"
#include "grpc/include/grpc++/server.h"
#include "grpc/include/grpc++/server_builder.h"
#include "grpc/include/grpc++/server_context.h"
#include "grpc/include/grpc++/support/status.h"
#include "grpc/include/grpc++/support/status_code_enum.h"
#include "grpc/include/grpc/grpc.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/command_line_flags.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow_serving/example/inference.grpc.pb.h"
#include "tensorflow_serving/example/inference.pb.h"
#include "tensorflow_serving/session_bundle/manifest.pb.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"
#include "tensorflow_serving/session_bundle/signature.h"

using grpc::InsecureServerCredentials;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;
using tensorflow::serving::ClassificationSignature;
using tensorflow::serving::VerifyRequest;
using tensorflow::serving::VerifyResponse;
using tensorflow::serving::VerifyService;
using tensorflow::serving::SessionBundle;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::TensorShape;

TF_DEFINE_int32(port, 0, "Port server listening on.");

namespace {
const int kImageSize1 = 112;
const int kImageSize2 = 98;
const int kNumChannels = 1;

const int kImageDataSize = 2 * kImageSize1 * kImageSize2 * kNumChannels;
const int kNumLabels = 1;

// Creates a gRPC Status from a TensorFlow Status.
Status ToGRPCStatus(const tensorflow::Status& status) {
  return Status(static_cast<grpc::StatusCode>(status.code()),
                status.error_message());
}

class VerifyServiceImpl final : public VerifyService::Service {
 public:
  explicit VerifyServiceImpl(std::unique_ptr<SessionBundle> bundle)
      : bundle_(std::move(bundle)) {
    signature_status_ = tensorflow::serving::GetClassificationSignature(
        bundle_->meta_graph_def, &signature_);
  }

  Status Verify(ServerContext* context, const VerifyRequest* request,
                  VerifyResponse* response) override {
    // Verify protobuf input.
    if (request->image_data_size() != kImageDataSize) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    tensorflow::strings::StrCat("expected image_data of size ",
                                                kImageDataSize, ", got ",
                                                request->image_data_size()));
    }

    if (request->image_data_p_size() != kImageDataSize) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    tensorflow::strings::StrCat("expected image_data of size ",
                                                kImageDataSize, ", got ",
                                                request->image_data_p_size()));
    }

    // Transform protobuf input to inference input tensor and create
    // output tensor placeholder.
    // See minist_export.py for details.
    Tensor input(tensorflow::DT_FLOAT, {2, 1, kImageSize1, kImageSize2, kNumChannels});
    std::copy_n(request->image_data().begin(), kImageDataSize,
                input.flat<float>().data());
    std::vector<Tensor> outputs;

    // Run inference.
    if (!signature_status_.ok()) {
      return ToGRPCStatus(signature_status_);
    }
    // WARNING(break-tutorial-inline-code): The following code snippet is
    // in-lined in tutorials, please update tutorial documents accordingly
    // whenever code changes.
    const tensorflow::Status status = bundle_->session->Run(
        {{signature_.input().tensor_name(), input}},
        {signature_.scores().tensor_name()}, {}, &outputs);
    if (!status.ok()) {
      return ToGRPCStatus(status);
    }

    // Transform inference output tensor to protobuf output.
    // See minist_export.py for details.
    if (outputs.size() != 1) {
      return Status(StatusCode::INTERNAL,
                    tensorflow::strings::StrCat(
                        "expected one model output, got ", outputs.size()));
    }
    const Tensor& score_tensor = outputs[0];
    const TensorShape expected_shape({1, kNumLabels});
    if (!score_tensor.shape().IsSameSize(expected_shape)) {
      return Status(
          StatusCode::INTERNAL,
          tensorflow::strings::StrCat("expected output of size ",
                                      expected_shape.DebugString(), ", got ",
                                      score_tensor.shape().DebugString()));
    }
    const auto score_flat = outputs[0].flat<float>();
    for (int i = 0; i < score_flat.size(); ++i) {
      response->add_value(score_flat(i));
    }

    return Status::OK;
  }

 private:
  std::unique_ptr<SessionBundle> bundle_;
  tensorflow::Status signature_status_;
  ClassificationSignature signature_;
};

void RunServer(int port, std::unique_ptr<SessionBundle> bundle) {
  // "0.0.0.0" is the way to listen on localhost in gRPC.
  const string server_address = "0.0.0.0:" + std::to_string(port);
  VerifyServiceImpl service(std::move(bundle));
  ServerBuilder builder;
  std::shared_ptr<grpc::ServerCredentials> creds = InsecureServerCredentials();
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Running...";
  server->Wait();
}

}  // namespace

int main(int argc, char** argv) {
  tensorflow::Status s = tensorflow::ParseCommandLineFlags(&argc, argv);
  if (!s.ok()) {
    LOG(FATAL) << "Error parsing command line flags: " << s.ToString();
  }
  if (argc != 2) {
    LOG(FATAL) << "Usage: mnist_inference --port=9000 /path/to/export";
  }
  const string bundle_path(argv[1]);

  tensorflow::port::InitMain(argv[0], &argc, &argv);

  // WARNING(break-tutorial-inline-code): The following code snippet is
  // in-lined in tutorials, please update tutorial documents accordingly
  // whenever code changes.
  tensorflow::SessionOptions session_options;
  std::unique_ptr<SessionBundle> bundle(new SessionBundle);
  const tensorflow::Status status =
      tensorflow::serving::LoadSessionBundleFromPath(session_options,
                                                     bundle_path, bundle.get());
  if (!status.ok()) {
    LOG(ERROR) << "Fail to load tensorflow export: " << status.error_message();
    return -1;
  }

  RunServer(FLAGS_port, std::move(bundle));

  return 0;
}
