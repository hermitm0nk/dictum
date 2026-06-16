cmake_minimum_required(VERSION 3.26)
project(dictum-native LANGUAGES C CXX)

include(ExternalProject)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(DICTUM_BUILD_LLAMA   "Build llama.cpp (LLM server with Vulkan)"  ON)
option(DICTUM_BUILD_CRISPASR "Build CrispASR (ASR engine, CPU only)"    ON)

set(DICTUM_NPROC "$ENV{DICTUM_NPROC}")
if(NOT DICTUM_NPROC OR DICTUM_NPROC EQUAL 0)
  include(ProcessorCount)
  ProcessorCount(DICTUM_NPROC)
  if(DICTUM_NPROC EQUAL 0)
    set(DICTUM_NPROC 4)
  endif()
endif()

set(DICTUM_NATIVE_SOURCE_CACHE "$ENV{DICTUM_NATIVE_SOURCE_CACHE}" CACHE PATH
  "Directory used to cache native dependency git checkouts between builds")
if(NOT DICTUM_NATIVE_SOURCE_CACHE)
  if(DEFINED ENV{XDG_CACHE_HOME} AND NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
    set(DICTUM_NATIVE_SOURCE_CACHE "$ENV{XDG_CACHE_HOME}/dictum/native-src")
  elseif(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
    set(DICTUM_NATIVE_SOURCE_CACHE "$ENV{HOME}/.cache/dictum/native-src")
  else()
    set(DICTUM_NATIVE_SOURCE_CACHE "${CMAKE_BINARY_DIR}/_deps/src")
  endif()
endif()
if(DICTUM_BUILD_LLAMA OR DICTUM_BUILD_CRISPASR)
  file(MAKE_DIRECTORY "${DICTUM_NATIVE_SOURCE_CACHE}")
endif()

set(NATIVE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/dictum/_native")
set(LLAMA_CPP_TAG "b9647")
set(CRISPASR_TAG "v0.7.2")

# ---------------------------------------------------------------------------
# 1. llama.cpp  –  Vulkan backend, no CUDA, no curl, no UI
# ---------------------------------------------------------------------------
if(DICTUM_BUILD_LLAMA)
  ExternalProject_Add(llama_cpp
    GIT_REPOSITORY    https://github.com/ggml-org/llama.cpp.git
    GIT_SHALLOW       ON
    GIT_TAG           "${LLAMA_CPP_TAG}"
    GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
    SOURCE_DIR        "${DICTUM_NATIVE_SOURCE_CACHE}/llama.cpp-${LLAMA_CPP_TAG}"
    BINARY_DIR        "${CMAKE_BINARY_DIR}/_deps/build/llama_cpp"

    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      -DGGML_VULKAN=ON
      -DGGML_CUDA=OFF
      -DLLAMA_CURL=OFF
      -DLLAMA_BUILD_UI=OFF
      -DLLAMA_BUILD_EXAMPLES=OFF
      -DLLAMA_BUILD_TESTS=OFF
      -DBUILD_SHARED_LIBS=ON
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
      "-DCMAKE_INSTALL_RPATH=\$ORIGIN/../lib/llama"
      "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"

    BUILD_COMMAND ${CMAKE_COMMAND} --build . -j ${DICTUM_NPROC} --target llama-server
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE OFF
    INSTALL_DIR "${CMAKE_BINARY_DIR}/llama-install"
  )
  ExternalProject_Get_Property(llama_cpp BINARY_DIR)
  set(LLAMA_CPP_BUILD_DIR "${BINARY_DIR}")
endif()

# ---------------------------------------------------------------------------
# 2. CrispASR  –  CPU only (Vulkan shader build is broken upstream)
# ---------------------------------------------------------------------------
if(DICTUM_BUILD_CRISPASR)
  ExternalProject_Add(crispasr
    GIT_REPOSITORY    https://github.com/CrispStrobe/CrispASR.git
    GIT_SHALLOW       ON
    GIT_TAG           "${CRISPASR_TAG}"
    GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
    SOURCE_DIR        "${DICTUM_NATIVE_SOURCE_CACHE}/CrispASR-${CRISPASR_TAG}"
    BINARY_DIR        "${CMAKE_BINARY_DIR}/_deps/build/crispasr"

    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      -DGGML_VULKAN=OFF
      -DCRISPASR_BUILD_EXAMPLES=ON
      -DCRISPASR_BUILD_TESTS=OFF
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
      "-DCMAKE_INSTALL_RPATH=\$ORIGIN/../lib/crispasr"
      "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"

    BUILD_COMMAND ${CMAKE_COMMAND} --build . -j ${DICTUM_NPROC} --target crispasr-cli
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE OFF
    INSTALL_DIR "${CMAKE_BINARY_DIR}/crispasr-install"
  )
  ExternalProject_Get_Property(crispasr BINARY_DIR)
  set(CRISPASR_BUILD_DIR "${BINARY_DIR}")
endif()

# ---------------------------------------------------------------------------
# 3. Install native artifacts into wheel layout  (dictum/_native/)
# ---------------------------------------------------------------------------
# llama.cpp: binary + shared libs
if(DICTUM_BUILD_LLAMA)
  install(PROGRAMS    "${LLAMA_CPP_BUILD_DIR}/bin/llama-server"
    DESTINATION      "${NATIVE_INSTALL_DIR}/bin"
  )
  install(DIRECTORY   "${LLAMA_CPP_BUILD_DIR}/bin/"
    DESTINATION      "${NATIVE_INSTALL_DIR}/lib/llama"
    FILES_MATCHING
      PATTERN "lib*.so*"
      PATTERN "lib*.dylib*"
  )
endif()

# CrispASR: binary (renamed to parakeet-main) + shared libs
if(DICTUM_BUILD_CRISPASR)
  install(PROGRAMS    "${CRISPASR_BUILD_DIR}/bin/crispasr"
    DESTINATION      "${NATIVE_INSTALL_DIR}/bin"
    RENAME           parakeet-main
  )
  install(DIRECTORY   "${CRISPASR_BUILD_DIR}/src/"
    DESTINATION      "${NATIVE_INSTALL_DIR}/lib/crispasr"
    FILES_MATCHING
      PATTERN "lib*.so*"
      PATTERN "lib*.dylib*"
  )
  install(DIRECTORY   "${CRISPASR_BUILD_DIR}/ggml/src/"
    DESTINATION      "${NATIVE_INSTALL_DIR}/lib/crispasr"
    FILES_MATCHING
      PATTERN "lib*.so*"
      PATTERN "lib*.dylib*"
  )
endif()
