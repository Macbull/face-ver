# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-parse_headers",
        "no_layering_check",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("@serving//tensorflow_serving:serving.bzl", "serving_proto_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

cc_library(
    name = "grpc_cc",
    srcs = [
        "inference.grpc.pb.cc",
        "inference.pb.cc",
    ],
    hdrs = [
        "inference.grpc.pb.h",
        "inference.pb.h",
    ],
    deps = [
        "//grpc:grpc++",
    ],
)

py_library(
    name = "input_data",
    srcs = ["input_data.py"],
)

py_library(
    name = "cifar10",
    srcs = ["cifar10.py"],
)

cc_binary(
    name = "inference",
    srcs = [
        "inference.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        "@tf//tensorflow/core:core_cpu",
        "@tf//tensorflow/core:framework",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:tensorflow",
        ":grpc_cc",
        "//grpc",
        "//tensorflow_serving/session_bundle",
        "//tensorflow_serving/session_bundle:manifest_proto",
        "//tensorflow_serving/session_bundle:signature",
    ],
)

py_binary(
    name = "cifar10_train",
    srcs = [
        "cifar10_train.py",
    ],
    deps = [
        "@tf//tensorflow:tensorflow_py",
        ":input_data",
        ":cifar10",
        "@serving//tensorflow_serving/session_bundle:exporter",
    ],
)


py_binary(
    name = "client",
    srcs = [
        "client.py",
        "inference_pb2.py",
    ],
    deps = [
        "@tf//tensorflow:tensorflow_py",
        ":input_data",
    ],
)
