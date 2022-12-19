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
    llvm.func @test_gemm_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr<f32>, %arg3: !llvm.ptr<f32>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr<f32>, %arg10: !llvm.ptr<f32>, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: !llvm.ptr<f32>, %arg17: !llvm.ptr<f32>, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %1 = llvm.insertvalue %arg2, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %2 = llvm.insertvalue %arg3, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %3 = llvm.insertvalue %arg4, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %4 = llvm.insertvalue %arg5, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %6 = llvm.insertvalue %arg6, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %9 = llvm.insertvalue %arg9, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %10 = llvm.insertvalue %arg10, %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %11 = llvm.insertvalue %arg11, %10[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %12 = llvm.insertvalue %arg12, %11[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %13 = llvm.insertvalue %arg14, %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %14 = llvm.insertvalue %arg13, %13[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %15 = llvm.insertvalue %arg15, %14[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %16 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
      %17 = llvm.insertvalue %arg16, %16[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %18 = llvm.insertvalue %arg17, %17[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %19 = llvm.insertvalue %arg18, %18[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %20 = llvm.insertvalue %arg19, %19[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %21 = llvm.insertvalue %arg21, %20[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %22 = llvm.insertvalue %arg20, %21[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %23 = llvm.insertvalue %arg22, %22[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %24 = nvvm.read.ptx.sreg.tid.x : i32
      %25 = llvm.sext %24 : i32 to i64
      %26 = nvvm.read.ptx.sreg.tid.y : i32
      %27 = llvm.sext %26 : i32 to i64
      llvm.br ^bb1
    ^bb1:  // pred: ^bb0
      %28 = llvm.add %arg0, %25  : i64
      %29 = llvm.add %arg1, %27  : i64
      %30 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %31 = llvm.mlir.constant(8192 : index) : i64
      %32 = llvm.mul %28, %31  : i64
      %33 = llvm.add %32, %29  : i64
      %34 = llvm.getelementptr %30[%33] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %35 = llvm.load %34 : !llvm.ptr<f32>
      %36 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %37 = llvm.mlir.constant(8192 : index) : i64
      %38 = llvm.mul %28, %37  : i64
      %39 = llvm.add %38, %29  : i64
      %40 = llvm.getelementptr %36[%39] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %41 = llvm.load %40 : !llvm.ptr<f32>
      %42 = llvm.fadd %35, %41  : f32
      %43 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %44 = llvm.mlir.constant(8192 : index) : i64
      %45 = llvm.mul %28, %44  : i64
      %46 = llvm.add %45, %29  : i64
      %47 = llvm.getelementptr %43[%46] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %42, %47 : !llvm.ptr<f32>
      llvm.return
    }
  }
}

