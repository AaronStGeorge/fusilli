// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <cstdint>
#include <memory>

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

// Test: 2D matrix multiplication with bias addition
// A[M,K] x B[K,N] = C[M,N], then C + bias[1,N] = output[M,N]
TEST(MatmulIntegrationTest, SimpleMatmulWithBias) {
  // Initialize HIP.
  ASSERT_EQ(hipInit(0), hipSuccess);
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  hipStream_t stream = nullptr;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

  // Set plugin paths.
  auto pluginPath = std::filesystem::canonical(getCurrentExecutableDirectory() /
                                               FUSILLI_PLUGIN_PATH);
  const std::array<const char *, 1> paths = {pluginPath.c_str()};
  ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(),
                                           HIPDNN_PLUGIN_LOADING_ABSOLUTE),
            HIPDNN_STATUS_SUCCESS);

  // Create handle.
  hipdnnHandle_t handle;
  ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);

  // Dimensions: A[4,8] x B[8,5] = C[4,5], bias[1,5]
  const int64_t M = 4;
  const int64_t K = 8;
  const int64_t N = 5;

  // UIDs.
  const int64_t aUID = 0;
  const int64_t bUID = 1;
  const int64_t biasUID = 2;
  const int64_t outUID = 3;

  // Initialize tensors.
  PinnedTensor<float> aTensor({M, K});
  PinnedTensor<float> bTensor({K, N});
  PinnedTensor<float> biasTensor({1, N});
  PinnedTensor<float> outTensor({M, N});
  aTensor.fillWithValue(1.0f);
  bTensor.fillWithValue(1.0f);
  biasTensor.fillWithValue(2.0f);
  outTensor.fillWithValue(-100.0f);

  // Expected output: each element = K + bias = 8 + 2 = 10.
  PinnedTensor<float> expectedOutput({M, N});
  expectedOutput.fillWithValue(static_cast<float>(K) + 2.0f);

  // Create graph.
  auto graph = std::make_shared<graph::Graph>();
  graph->set_name("simple_matmul_with_bias_test");
  graph->set_io_data_type(DataType_t::FLOAT)
      .set_intermediate_data_type(DataType_t::FLOAT)
      .set_compute_data_type(DataType_t::FLOAT);

  // Create tensor attributes for inputs.
  auto aAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("A", DataType_t::FLOAT, aTensor));
  aAttr->set_uid(aUID);
  auto bAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("B", DataType_t::FLOAT, bTensor));
  bAttr->set_uid(bUID);
  auto biasAttr = std::make_shared<graph::TensorAttributes>(
      graph::makeTensorAttributes("bias", DataType_t::FLOAT, biasTensor));
  biasAttr->set_uid(biasUID);

  // Create matmul attributes.
  graph::MatmulAttributes matmulAttr;
  matmulAttr.set_name("matmul");

  // Create matmul node (intermediate output).
  auto matmulOutAttr = graph->matmul(aAttr, bAttr, matmulAttr);
  matmulOutAttr->set_dim(outTensor.dims())
      .set_stride(outTensor.strides())
      .set_output(false);

  // Create pointwise ADD for bias.
  graph::PointwiseAttributes biasAddAttr;
  biasAddAttr.set_name("bias_add").set_mode(PointwiseMode_t::ADD);

  // Add bias to matmul output (final output).
  auto outAttr = graph->pointwise(matmulOutAttr, biasAttr, biasAddAttr);
  outAttr->set_uid(outUID);
  outAttr->set_dim(outTensor.dims())
      .set_stride(outTensor.strides())
      .set_output(true);

  // Build + validate + build plans for graph.
  auto result = graph->validate();
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  result = graph->build_operation_graph(handle);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  result = graph->create_execution_plans();
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  result = graph->check_support();
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  result = graph->build_plans();
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

  // Create variant pack.
  std::unordered_map<int64_t, void *> variantPack;
  variantPack[aUID] = aTensor.memory().deviceData();
  variantPack[bUID] = bTensor.memory().deviceData();
  variantPack[biasUID] = biasTensor.memory().deviceData();
  variantPack[outUID] = outTensor.memory().deviceData();

  // Execute graph.
  result = graph->execute(handle, variantPack, nullptr);
  ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  outTensor.memory().markDeviceModified();

  // Check results.
  CpuFpReferenceValidation<float> validator(1e-6f, 1e-6f);
  EXPECT_TRUE(validator.allClose(expectedOutput, outTensor));

  // Clean up.
  ASSERT_EQ(hipStreamDestroy(stream), HIPDNN_STATUS_SUCCESS);
  ASSERT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
