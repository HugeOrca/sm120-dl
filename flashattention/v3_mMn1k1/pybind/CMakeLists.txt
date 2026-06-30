cmake_minimum_required(VERSION 3.24)

if(DEFINED ENV{TORCH132_ENV})
  set(_DEFAULT_TORCH132_ENV "$ENV{TORCH132_ENV}")
elseif(DEFINED ENV{CONDA_PREFIX})
  set(_DEFAULT_TORCH132_ENV "$ENV{CONDA_PREFIX}")
else()
  set(_DEFAULT_TORCH132_ENV "")
endif()
set(TORCH132_ENV "${_DEFAULT_TORCH132_ENV}" CACHE PATH "Conda environment used for Python, Torch, and CUDA runtime libraries")
if(NOT TORCH132_ENV)
  message(FATAL_ERROR "TORCH132_ENV is not set. Run `conda activate torch132`, export TORCH132_ENV, or pass -DTORCH132_ENV=/path/to/env.")
endif()

if(DEFINED ENV{CUDA_HOME})
  set(_DEFAULT_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc")
else()
  set(_DEFAULT_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()
set(CMAKE_CUDA_COMPILER "${_DEFAULT_CUDA_COMPILER}" CACHE FILEPATH "CUDA compiler with sm_120a support")

project(flashatten_v3_sm120_pybind LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY None)
set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "Only sm_120a is supported" FORCE)
set(TORCH_CUDA_ARCH_LIST "12.0a" CACHE STRING "Torch CUDA arch list for sm_120a" FORCE)
set(ENV{TORCH_CUDA_ARCH_LIST} "12.0a")
set(FLASH_FWD_MAX_SMEM_KB "100" CACHE STRING "Maximum dynamic shared memory per generated FlashAttention config")
set(FLASH_FWD_ONLY_CONFIGS "" CACHE STRING "Comma-separated generated FlashAttention configs to build")

set(Python3_EXECUTABLE "${TORCH132_ENV}/bin/python" CACHE FILEPATH "Python executable from torch132" FORCE)
set(Python_EXECUTABLE "${TORCH132_ENV}/bin/python" CACHE FILEPATH "Python executable from torch132" FORCE)
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
  OUTPUT_VARIABLE TORCH_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import os, sysconfig; print(os.path.join(sysconfig.get_paths()['purelib'], 'nvidia', 'cu13', 'lib'))"
  OUTPUT_VARIABLE TORCH132_CUDA_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)

list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}" "${PYBIND11_CMAKE_DIR}")
find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_library(TORCH132_CUDART_LIBRARY
  NAMES cudart libcudart.so.13
  PATHS "${TORCH132_CUDA_LIB_DIR}"
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(TORCH_LIBRARY NAMES torch PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(TORCH_CPU_LIBRARY NAMES torch_cpu PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(TORCH_CUDA_LIBRARY NAMES torch_cuda PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(TORCH_PYTHON_LIBRARY NAMES torch_python PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(C10_LIBRARY NAMES c10 PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(C10_CUDA_LIBRARY NAMES c10_cuda PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(FLASH_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_DIR}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${FLASH_SOURCE_DIR}/flash_configs.py"
)
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "FLASH_FWD_MAX_SMEM_KB=${FLASH_FWD_MAX_SMEM_KB}"
    "FLASH_FWD_ONLY_CONFIGS=${FLASH_FWD_ONLY_CONFIGS}"
    "${Python3_EXECUTABLE}" "${FLASH_SOURCE_DIR}/flash_configs.py"
  WORKING_DIRECTORY "${GENERATED_DIR}"
  RESULT_VARIABLE GEN_RESULT
)
if(NOT GEN_RESULT EQUAL 0)
  message(FATAL_ERROR "flashattention/v3_mMn1k1/flash_configs.py failed with code ${GEN_RESULT}")
endif()
include("${GENERATED_DIR}/configs.cmake")

string(REPLACE "|" ";" default_cfg_list "${FLASH_FWD_DEFAULT_CONFIG}")
list(GET default_cfg_list 6 DEFAULT_CONFIG_NAME)
set(GENERATED_KERNELS "")
set(GENERATED_CONFIG_DECLS "")
set(GENERATED_CONFIG_TABLE "")

foreach(cfg IN LISTS FLASH_FWD_CONFIGS)
  string(REPLACE "|" ";" cfg_list "${cfg}")
  list(GET cfg_list 0 HEAD_DIM)
  list(GET cfg_list 1 BM)
  list(GET cfg_list 2 BN)
  list(GET cfg_list 3 NWARPS)
  list(GET cfg_list 4 STAGE)
  list(GET cfg_list 5 SMEM_BYTES)
  list(GET cfg_list 6 CFG_NAME)

  string(MAKE_C_IDENTIFIER "${CFG_NAME}" CFG_SYMBOL)
  message(STATUS "Instantiating flash_fwd config: ${CFG_NAME}")

  set(GENERATED_KERNEL "${GENERATED_DIR}/flash_fwd_${CFG_NAME}.cu")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/flash_fwd_config.cu.in" "${GENERATED_KERNEL}" @ONLY)
  list(APPEND GENERATED_KERNELS "${GENERATED_KERNEL}")

  string(APPEND GENERATED_CONFIG_DECLS
"torch::Tensor run_flash_fwd_${CFG_SYMBOL}(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool causal);

")
  string(APPEND GENERATED_CONFIG_TABLE
"    {
        \"${CFG_NAME}\",
        ${HEAD_DIM},
        ${BM},
        ${BN},
        ${NWARPS},
        ${STAGE},
        ${SMEM_BYTES},
        &run_flash_fwd_${CFG_SYMBOL},
    },
")
endforeach()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/flash_fwd_config_registry.hpp.in"
               "${GENERATED_DIR}/flash_fwd_config_registry.hpp" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/flash_fwd_config_registry.cu.in"
               "${GENERATED_DIR}/flash_fwd_config_registry.cu" @ONLY)

pybind11_add_module(flashatten_v3_sm120 MODULE
  flashatten_v3_pybind.cu
  ${GENERATED_KERNELS}
  "${GENERATED_DIR}/flash_fwd_config_registry.cu"
)

set_target_properties(flashatten_v3_sm120 PROPERTIES
  CUDA_ARCHITECTURES 120a
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  BUILD_RPATH "${TORCH_LIB_DIR};${TORCH132_CUDA_LIB_DIR};${TORCH132_ENV}/lib"
  INSTALL_RPATH "${TORCH_LIB_DIR};${TORCH132_CUDA_LIB_DIR};${TORCH132_ENV}/lib"
)

target_link_options(flashatten_v3_sm120 PRIVATE
  -Wl,--disable-new-dtags
)

target_include_directories(flashatten_v3_sm120 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${REPO_ROOT}
  ${GENERATED_DIR}
  ${REPO_ROOT}/3rd/cutlass/include
  ${REPO_ROOT}/3rd/cutlass/tools/util/include
  ${TORCH_INCLUDE_DIRS}
)

target_compile_options(flashatten_v3_sm120 PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

target_compile_options(flashatten_v3_sm120 PRIVATE ${TORCH_CXX_FLAGS})

target_link_libraries(flashatten_v3_sm120 PRIVATE
  "${TORCH_PYTHON_LIBRARY}"
  "${TORCH_LIBRARY}"
  "${TORCH_CPU_LIBRARY}"
  "${TORCH_CUDA_LIBRARY}"
  "${C10_LIBRARY}"
  "${C10_CUDA_LIBRARY}"
  "${TORCH132_CUDART_LIBRARY}"
)
