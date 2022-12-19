module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, gpu.container_module, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @test_gemm(%arg0: memref<8192x8192xf32>, %arg1: memref<8192x8192xf32>, %arg2: memref<8192x8192xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c8192 = arith.constant 8192 : index
    %0 = arith.subi %c8192, %c0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c8192_1 = arith.constant 8192 : index
    %1 = arith.subi %c8192_1, %c0_0 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    gpu.launch_func  @test_gemm_kernel::@test_gemm_kernel blocks in (%c1_3, %c1_3, %c1_3) threads in (%0, %1, %c1_3) args(%c0 : index, %c0_0 : index, %arg1 : memref<8192x8192xf32>, %arg2 : memref<8192x8192xf32>, %arg0 : memref<8192x8192xf32>)
    return
  }
  gpu.module @test_gemm_kernel {
    gpu.func @test_gemm_kernel(%arg0: index, %arg1: index, %arg2: memref<8192x8192xf32>, %arg3: memref<8192x8192xf32>, %arg4: memref<8192x8192xf32>) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %12 = arith.addi %arg0, %3 : index
      %13 = arith.addi %arg1, %4 : index
      %14 = memref.load %arg2[%12, %13] : memref<8192x8192xf32>
      %15 = memref.load %arg3[%12, %13] : memref<8192x8192xf32>
      %16 = arith.addf %14, %15 : f32
      memref.store %16, %arg4[%12, %13] : memref<8192x8192xf32>
      gpu.return
    }
  }
}

