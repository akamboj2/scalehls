#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @convolution1(%arg0: memref<1x32x32xf32>, %arg1: memref<6x1x5x5xf32>, %arg2: memref<6xf32>, %arg3: memref<6x28x28xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg4 = 0 to 6 {
      affine.for %arg5 = 0 to 28 {
        affine.for %arg6 = 0 to 28 {
          %0 = affine.for %arg7 = #map(%arg5) to #map1(%arg5) iter_args(%arg8 = %cst) -> (f32) {
            %3 = affine.for %arg9 = #map(%arg6) to #map1(%arg6) iter_args(%arg10 = %arg8) -> (f32) {
              %4 = affine.load %arg1[%arg4, 0, %arg7 - %arg5, %arg9 - %arg6] : memref<6x1x5x5xf32>
              %5 = affine.load %arg0[0, %arg7, %arg9] : memref<1x32x32xf32>
              %6 = arith.mulf %4, %5 : f32
              %7 = arith.addf %arg10, %6 : f32
              affine.yield %7 : f32
            }
            affine.yield %3 : f32
          }
          %1 = affine.load %arg2[%arg4] : memref<6xf32>
          %2 = arith.addf %0, %1 : f32
          affine.store %2, %arg3[%arg4, %arg5, %arg6] : memref<6x28x28xf32>
        }
      }
    }
    return
  }
}
