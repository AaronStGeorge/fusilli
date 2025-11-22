// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains utilities for finding required external programs at
// runtime.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_EXTERNAL_TOOLS_H
#define FUSILLI_SUPPORT_EXTERNAL_TOOLS_H

#include <cstdlib>
#include <string>

namespace fusilli {

inline std::string getIreeCompilePath() {
  // Check environment variable.
  const char *envPath = std::getenv("FUSILLI_EXTERNAL_IREE_COMPILE");
  if (envPath && envPath[0] != '\0') {
    return std::string(envPath);
  }

  // Let the shell search for it.
  return "iree-compile";
}

inline std::string getRocmAgentEnumeratorPath() {
  // Check environment variable.
  const char *envPath = std::getenv("FUSILLI_EXTERNAL_ROCM_AGENT_ENUMERATOR");
  if (envPath && envPath[0] != '\0') {
    return std::string(envPath);
  }

  // Let shell search for it.
  return std::string("rocm_agent_enumerator");
}

} // namespace fusilli

#undef IREE_COMPILE_PATH
#undef ROCM_AGENT_ENUMERATOR_PATH

#endif // FUSILLI_SUPPORT_EXTERNAL_TOOLS_H
