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

py_library(
    name = "input_data",
    srcs = ["input_data.py"],
)

py_library(
    name = "cifar10",
    srcs = ["cifar10.py"],
)

py_binary(
    name = "mnist_export",
    srcs = [
        "mnist_export.py",
    ],
    deps = [
        "@tf//tensorflow:tensorflow_py",
        ":mnist_input_data",
        "@serving//tensorflow_serving/session_bundle:exporter",
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
