# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.23)

project(fusilli-plugin
  VERSION 0.1.0
  DESCRIPTION "Fusilli-Plugin: A Fusilli/IREE powered hipDNN plugin for graph JIT compilation."
  LANGUAGES C CXX)

# Include GNUInstallDirs for standard install directories
include(GNUInstallDirs)
include(FetchContent)

# Set C++ standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Dependencies
set(HIP_PLATFORM "amd")
find_package(hip CONFIG REQUIRED)

################################################################################
# IREE Runtime Dependency
# Build from source, same pattern as fusilli/CMakeLists.txt
################################################################################

set(IREE_GIT_TAG "iree-3.10.0rc20251201")
set(IREE_SOURCE_DIR "" CACHE FILEPATH "Path to IREE source")

# Set IREE build flags
set(IREE_VISIBILITY_HIDDEN OFF)
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
set(IREE_HAL_DRIVER_DEFAULTS OFF)
set(IREE_HAL_DRIVER_HIP ON)
set(IREE_HIP_TEST_TARGET_CHIP "" CACHE STRING "Enable ROCm testing on specific architecture (e.g. gfx942)")

message(STATUS "TACO")

if(IREE_SOURCE_DIR)
  message(STATUS "Using existing IREE sources: ${IREE_SOURCE_DIR}")
  add_subdirectory(${IREE_SOURCE_DIR} fusilli_plugin_iree SYSTEM EXCLUDE_FROM_ALL)
else()
  message(STATUS "Fetching IREE sources from tag ${IREE_GIT_TAG}")
  set(IREE_SUBMODULES "third_party/benchmark third_party/flatcc third_party/hip-build-deps")
  FetchContent_Declare(
    fusilli_plugin_iree
    GIT_REPOSITORY https://github.com/iree-org/iree.git
    GIT_TAG "${IREE_GIT_TAG}"
    GIT_SUBMODULES ${IREE_SUBMODULES}
    GIT_SHALLOW TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
  )
  FetchContent_GetProperties(fusilli_plugin_iree)
  if(NOT fusilli_plugin_iree_POPULATED)
    FetchContent_MakeAvailable(fusilli_plugin_iree)
  endif()
endif()

# Other dependencies
find_package(hipdnn_sdk CONFIG REQUIRED)
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(Fusilli CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

# Global constants
set(FUSILLI_PLUGIN_NAME fusilli_plugin)
set(FUSILLI_PLUGIN_ENGINE_ID 1001)
# Relative path from build/install prefix to plugin location.
# HIPDNN_PLUGIN_ENGINE_SUBDIR is provided by hipdnn_sdk.
set(FUSILLI_PLUGIN_RELATIVE_PATH "${CMAKE_INSTALL_LIBDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}")

# Useful compile warnings
list(PREPEND FUSILLI_PLUGIN_WARNING_COMPILE_OPTIONS
    -Werror                                  # Treat all warnings as errors
    -Wall                                    # Enable most common warnings
    -Wextra                                  # Enable additional warnings not covered by -Wall
    -Wpedantic                               # Enforce strict ISO C++ compliance
    -Wshadow                                 # Warn about variable shadowing
    -Wnon-virtual-dtor                       # Warn if a class with virtual functions has a non-virtual destructor
    -Wold-style-cast                         # Warn about C-style casts
    -Wcast-align                             # Warn about potential performance issues with misaligned casts
    -Woverloaded-virtual                     # Warn if a base class function is hidden by a derived class function with the same name
    -Wconversion                             # Warn about implicit type conversions that may alter a value
    -Wsign-conversion                        # Warn about implicit conversions between signed and unsigned types
    -Wnull-dereference                       # Warn about dereferencing null pointers
    -Wdouble-promotion                       # Warn when a float is implicitly promoted to a double
    -Wformat=2                               # Enable stricter format string checks
    -Winit-self                              # Warn about variables initialized with itself
    -Wunreachable-code                       # Warn about unreachable code
    -Wno-error=unused-command-line-argument  # Disable error for unused command line arguments
    -Wno-gnu-statement-expression            # Disable gnu error for statement expression, this will need to change on Windows.
    -Wswitch-default                         # Warn if a switch statement does not have a default case
)

# Includes
include_directories(include)

# Plugin definition
add_library(${FUSILLI_PLUGIN_NAME} SHARED
  src/fusilli_plugin.cpp
)
target_compile_options(${FUSILLI_PLUGIN_NAME} PRIVATE ${FUSILLI_PLUGIN_WARNING_COMPILE_OPTIONS})
target_link_libraries(${FUSILLI_PLUGIN_NAME} PRIVATE hipdnn_sdk hip::host fusilli::fusilli)
target_compile_definitions(${FUSILLI_PLUGIN_NAME} PRIVATE
  FUSILLI_PLUGIN_NAME="${FUSILLI_PLUGIN_NAME}"
  FUSILLI_PLUGIN_ENGINE_ID=${FUSILLI_PLUGIN_ENGINE_ID}
)
set_target_properties(${FUSILLI_PLUGIN_NAME} PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

# Installation
install(TARGETS ${FUSILLI_PLUGIN_NAME}
  LIBRARY DESTINATION "${FUSILLI_PLUGIN_RELATIVE_PATH}"
)

# Tests
enable_testing()
add_subdirectory(test)
