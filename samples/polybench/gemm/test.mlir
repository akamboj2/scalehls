module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @test_gemm(%arg0: memref<8192x8192xf32>, %arg1: memref<8192x8192xf32>, %arg2: memref<8192x8192xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.mlir.constant(8192 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(256 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(32 : index) : i64
    %5 = builtin.unrealized_conversion_cast %arg1 : memref<8192x8192xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = builtin.unrealized_conversion_cast %arg2 : memref<8192x8192xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = builtin.unrealized_conversion_cast %arg0 : memref<8192x8192xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = builtin.unrealized_conversion_cast %2 : i64 to index
    %9 = builtin.unrealized_conversion_cast %2 : i64 to index
    %10 = builtin.unrealized_conversion_cast %1 : i64 to index
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %10, %arg10 = %10, %arg11 = %10) threads(%arg6, %arg7, %arg8) in (%arg12 = %8, %arg13 = %9, %arg14 = %10) {
      %11 = builtin.unrealized_conversion_cast %arg6 : index to i64
      %12 = builtin.unrealized_conversion_cast %arg7 : index to i64
      %13 = llvm.add %11, %3  : i64
      %14 = llvm.add %12, %3  : i64
      %15 = builtin.unrealized_conversion_cast %3 : i64 to index
      cf.br ^bb1(%15 : index)
    ^bb1(%16: index):  // 2 preds: ^bb0, ^bb5
      %17 = builtin.unrealized_conversion_cast %16 : index to i64
      %18 = llvm.icmp "slt" %17, %4 : i64
      llvm.cond_br %18, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %19 = builtin.unrealized_conversion_cast %3 : i64 to index
      cf.br ^bb3(%19 : index)
    ^bb3(%20: index):  // 2 preds: ^bb2, ^bb4
      %21 = builtin.unrealized_conversion_cast %20 : index to i64
      %22 = llvm.icmp "slt" %21, %4 : i64
      llvm.cond_br %22, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %23 = llvm.mul %13, %4  : i64
      %24 = llvm.add %17, %23  : i64
      %25 = llvm.mul %14, %4  : i64
      %26 = llvm.add %21, %25  : i64
      %27 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %28 = llvm.mul %24, %0  : i64
      %29 = llvm.add %28, %26  : i64
      %30 = llvm.getelementptr %27[%29] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %31 = llvm.load %30 : !llvm.ptr<f32>
      %32 = llvm.mul %13, %4  : i64
      %33 = llvm.add %17, %32  : i64
      %34 = llvm.mul %14, %4  : i64
      %35 = llvm.add %21, %34  : i64
      %36 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %37 = llvm.mul %33, %0  : i64
      %38 = llvm.add %37, %35  : i64
      %39 = llvm.getelementptr %36[%38] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      %40 = llvm.load %39 : !llvm.ptr<f32>
      %41 = llvm.fadd %31, %40  : f32
      %42 = llvm.mul %13, %4  : i64
      %43 = llvm.add %17, %42  : i64
      %44 = llvm.mul %14, %4  : i64
      %45 = llvm.add %21, %44  : i64
      %46 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
      %47 = llvm.mul %43, %0  : i64
      %48 = llvm.add %47, %45  : i64
      %49 = llvm.getelementptr %46[%48] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %41, %49 : !llvm.ptr<f32>
      %50 = llvm.add %21, %1  : i64
      %51 = builtin.unrealized_conversion_cast %50 : i64 to index
      cf.br ^bb3(%51 : index)
    ^bb5:  // pred: ^bb3
      %52 = llvm.add %17, %1  : i64
      %53 = builtin.unrealized_conversion_cast %52 : i64 to index
      cf.br ^bb1(%53 : index)
    ^bb6:  // pred: ^bb1
      gpu.terminator
    }
    return
  }
}

