module {
  func.func @matmul(%arg0: memref<8192x8192xf32>, %arg1: memref<8192x8192xf32>, %arg2: memref<8192x8192xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %c1_4 = arith.constant 1 : index
    gpu.launch blocks(%arg3, %arg4, %arg5) in (%arg9 = %c1_4, %arg10 = %c1_4, %arg11 = %c1_4) threads(%arg6, %arg7, %arg8) in (%arg12 = %0, %arg13 = %1, %arg14 = %c1_4) {
      %2 = arith.addi %c0_0, %arg6 : index
      %3 = arith.addi %c0_1, %arg7 : index
      affine.for %arg15 = 0 to 32 {
        affine.for %arg16 = 0 to 32 {
          affine.for %arg17 = 0 to 256 {
            %alloc = memref.alloc() : memref<32x1xf32>
            affine.for %arg18 = 0 to 32 {
              %19 = arith.muli %arg17, %c32 : index
              %20 = arith.addi %arg18, %19 : index
              %21 = arith.muli %3, %c32 : index
              %22 = arith.addi %arg16, %21 : index
              %23 = memref.load %arg0[%20, %22] : memref<8192x8192xf32>
              memref.store %23, %alloc[%arg18, %c0] : memref<32x1xf32>
            } {parallel, point}
            %alloc_5 = memref.alloc() : memref<1x1xf32>
            %4 = arith.muli %3, %c32 : index
            %5 = arith.addi %arg16, %4 : index
            %6 = arith.muli %2, %c32 : index
            %7 = arith.addi %arg15, %6 : index
            %8 = memref.load %arg1[%5, %7] : memref<8192x8192xf32>
            memref.store %8, %alloc_5[%c0, %c0] : memref<1x1xf32>
            %alloc_6 = memref.alloc() : memref<1x1xf32>
            %9 = arith.muli %3, %c32 : index
            %10 = arith.addi %arg16, %9 : index
            %11 = arith.muli %2, %c32 : index
            %12 = arith.addi %arg15, %11 : index
            %13 = memref.load %arg2[%10, %12] : memref<8192x8192xf32>
            memref.store %13, %alloc_6[%c0, %c0] : memref<1x1xf32>
            affine.for %arg18 = 0 to 32 {
              %19 = memref.load %alloc[%arg18, %c0] : memref<32x1xf32>
              %20 = memref.load %alloc_5[%c0, %c0] : memref<1x1xf32>
              %21 = arith.mulf %19, %20 : f32
              %22 = memref.load %alloc_6[%c0, %c0] : memref<1x1xf32>
              %23 = arith.addf %22, %21 : f32
              memref.store %23, %alloc_6[%c0, %c0] : memref<1x1xf32>
            } {point}
            %14 = memref.load %alloc_6[%c0, %c0] : memref<1x1xf32>
            %15 = arith.muli %3, %c32 : index
            %16 = arith.addi %arg16, %15 : index
            %17 = arith.muli %2, %c32 : index
            %18 = arith.addi %arg15, %17 : index
            memref.store %14, %arg2[%16, %18] : memref<8192x8192xf32>
          }
        }
      }
      gpu.terminator
    }
    return
  }
}

