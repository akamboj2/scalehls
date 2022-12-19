module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @test_gemm(%arg0: memref<8192x8192xf32>, %arg1: memref<8192x8192xf32>, %arg2: memref<8192x8192xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c0_0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %0 = arith.subi %c256, %c0_0 : index
    %c1 = arith.constant 1 : index
    %c0_1 = arith.constant 0 : index
    %c256_2 = arith.constant 256 : index
    %1 = arith.subi %c256_2, %c0_1 : index
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %c32_5 = arith.constant 32 : index
    %2 = arith.subi %c32_5, %c0_4 : index
    %c1_6 = arith.constant 1 : index
    %c0_7 = arith.constant 0 : index
    %c32_8 = arith.constant 32 : index
    %3 = arith.subi %c32_8, %c0_7 : index
    %c1_9 = arith.constant 1 : index
    %c1_10 = arith.constant 1 : index
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %0, %arg10 = %1, %arg11 = %c1_10) threads(%arg6, %arg7, %arg8) in (%arg12 = %2, %arg13 = %3, %arg14 = %c1_10) {
      %4 = arith.addi %c0_0, %arg3 : index
      %5 = arith.addi %c0_1, %arg4 : index
      %6 = arith.addi %c0_4, %arg6 : index
      %7 = arith.addi %c0_7, %arg7 : index
      affine.for %arg15 = 0 to 256 {
        %alloc = memref.alloc() : memref<1x32xf32>
        affine.for %arg16 = 0 to 32 {
          %18 = arith.muli %4, %c32 : index
          %19 = arith.addi %6, %18 : index
          %20 = arith.muli %arg15, %c32 : index
          %21 = arith.addi %arg16, %20 : index
          %22 = memref.load %arg1[%19, %21] : memref<8192x8192xf32>
          memref.store %22, %alloc[%c0, %arg16] : memref<1x32xf32>
        } {parallel, point}
        %alloc_11 = memref.alloc() : memref<32x1xf32>
        affine.for %arg16 = 0 to 32 {
          %18 = arith.muli %arg15, %c32 : index
          %19 = arith.addi %arg16, %18 : index
          %20 = arith.muli %5, %c32 : index
          %21 = arith.addi %7, %20 : index
          %22 = memref.load %arg2[%19, %21] : memref<8192x8192xf32>
          memref.store %22, %alloc_11[%arg16, %c0] : memref<32x1xf32>
        } {parallel, point}
        %alloc_12 = memref.alloc() : memref<1x1xf32>
        %8 = arith.muli %4, %c32 : index
        %9 = arith.addi %6, %8 : index
        %10 = arith.muli %5, %c32 : index
        %11 = arith.addi %7, %10 : index
        %12 = memref.load %arg0[%9, %11] : memref<8192x8192xf32>
        memref.store %12, %alloc_12[%c0, %c0] : memref<1x1xf32>
        affine.for %arg16 = 0 to 32 {
          %18 = memref.load %alloc[%c0, %arg16] : memref<1x32xf32>
          %19 = memref.load %alloc_11[%arg16, %c0] : memref<32x1xf32>
          %20 = arith.mulf %18, %19 : f32
          %21 = memref.load %alloc_12[%c0, %c0] : memref<1x1xf32>
          %22 = arith.addf %21, %20 : f32
          memref.store %22, %alloc_12[%c0, %c0] : memref<1x1xf32>
        } {point}
        %13 = memref.load %alloc_12[%c0, %c0] : memref<1x1xf32>
        %14 = arith.muli %4, %c32 : index
        %15 = arith.addi %6, %14 : index
        %16 = arith.muli %5, %c32 : index
        %17 = arith.addi %7, %16 : index
        memref.store %13, %arg0[%15, %17] : memref<8192x8192xf32>
      }
      gpu.terminator
    }
    return
  }
}

