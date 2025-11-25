// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>

#include <algorithm>
#include <filesystem>
#include <string>

using hipdnn_sdk::utilities::getCurrentExecutableDirectory;

static std::vector<std::string> getLoadedPlugins(hipdnnHandle_t handle) {
  size_t numPlugins = 0;
  size_t maxPathLength = 0;
  auto status = hipdnnGetLoadedEnginePluginPaths_ext(handle, &numPlugins,
                                                     nullptr, &maxPathLength);

  if (status != HIPDNN_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to get loaded plugin paths");
  }

  if (numPlugins == 0) {
    return {};
  }

  std::vector<std::vector<char>> pathBuffers(numPlugins,
                                             std::vector<char>(maxPathLength));
  std::vector<char *> pluginPathsC(numPlugins);
  for (size_t i = 0; i < numPlugins; ++i) {
    pluginPathsC[i] = pathBuffers[i].data();
  }

  status = hipdnnGetLoadedEnginePluginPaths_ext(
      handle, &numPlugins, pluginPathsC.data(), &maxPathLength);
  if (status != HIPDNN_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to get loaded plugin paths");
  }

  std::vector<std::string> pluginPaths;
  pluginPaths.reserve(numPlugins);
  for (size_t i = 0; i < numPlugins; ++i) {
    pluginPaths.emplace_back(pluginPathsC[i]);
  }
  return pluginPaths;
}

TEST(IntegrationTests, PluginLoad) {
  // Resolve plugin path relative to executable directory.
  // FUSILLI_PLUGIN_PATH is e.g. "../lib/hipdnn_plugins/engines/fusilli_plugin"
  auto pluginPath = std::filesystem::weakly_canonical(
      getCurrentExecutableDirectory() / FUSILLI_PLUGIN_PATH);
  const std::string pluginPathStr = pluginPath.string();
  const std::array<const char *, 1> paths = {pluginPathStr.c_str()};
  hipdnnStatus_t status = hipdnnSetEnginePluginPaths_ext(
      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE);
  EXPECT_EQ(status, HIPDNN_STATUS_SUCCESS);

  // Stand up enough of hipDNN to load plugins.
  hipdnnHandle_t handle = nullptr;
  status = hipdnnCreate(&handle);
  ASSERT_EQ(status, HIPDNN_STATUS_SUCCESS);
  ASSERT_NE(handle, nullptr);

  // If fusilli plugin fails to define a required method it will fail to load.
  auto loadedPlugins = getLoadedPlugins(handle);
  EXPECT_EQ(loadedPlugins.size(), 1);

  // Check that fusilli plugin did load.
  auto expectedPath =
      pluginPath / std::format("lib{}.so", FUSILLI_PLUGIN_NAME);
  EXPECT_TRUE(std::ranges::any_of(
      loadedPlugins, [&expectedPath](const std::string &loadedPluginPath) {
        return std::filesystem::canonical(loadedPluginPath) ==
               std::filesystem::canonical(expectedPath);
      }));

  EXPECT_EQ(hipdnnDestroy(handle), HIPDNN_STATUS_SUCCESS);
}
