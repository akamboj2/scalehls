#!/bin/bash

MLIR_OPT=/home/abhi/courses/ece527-SOC-Design/final_project/scalehls/build/bin/mlir-opt
MLIR_CPU_RUNNER=/home/abhi/courses/ece527-SOC-Design/final_project/scalehls/build/bin/mlir-cpu-runner

MLIR_RUNNER_UTILS=/home/abhi/courses/ece527-SOC-Design/final_project/scalehls/build/lib/libmlir_runner_utils.so
# MLIR_C_RUNNER_UTILS := ../../llvm/build/lib/libmlir_c_runner_utils.so
# MLIR_ASYNC_RUNTIME := ../../llvm/build/lib/libmlir_async_runtime.so
MLIR_CUDA_RUNTIME=/home/abhi/courses/ece527-SOC-Design/final_project/scalehls/build/lib/libmlir_cuda_runtime.so

${MLIR_OPT} test_gemm_add_nvvm.mlir -gpu-kernel-outlining | \
${MLIR_OPT} -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,gpu-to-cubin))' | \
${MLIR_OPT} -gpu-to-llvm | \
${MLIR_CPU_RUNNER} -entry-point-result=void -shared-libs=${MLIR_RUNNER_UTILS} -shared-libs=${MLIR_CUDA_RUNTIME}