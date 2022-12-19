module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1_3, %arg10 = %c1_3, %arg11 = %c1_3) threads(%arg6, %arg7, %arg8) in (%arg12 = %0, %arg13 = %1, %arg14 = %c1_3) {
      %2 = arith.addi %c0, %arg6 : index
      %3 = arith.addi %c0_0, %arg7 : index
      %4 = memref.load %arg1[%2, %3] : memref<8192x8192xf32>
      %5 = memref.load %arg2[%2, %3] : memref<8192x8192xf32>
      %6 = arith.addf %4, %5 : f32
      memref.store %6, %arg0[%2, %3] : memref<8192x8192xf32>
      gpu.terminator
    }
    return
  }
}

