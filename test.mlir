func.func @matmul(%arg0: memref<?x400xf32>, %arg1: memref<?x300xf32>,
                  %arg2: memref<?x300xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 = 0 to 200 {
      affine.for %arg4 = 0 to 300 {
        affine.for %arg5 = 0 to 400 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<?x400xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<?x300xf32>
          %2 = arith.mulf %0, %1 : f32
          %3 = memref.load %arg2[%arg5, %arg4] : memref<?x300xf32>
          %4 = arith.addf %3, %2 : f32
          memref.store %4, %arg2[%arg5, %arg4] : memref<?x300xf32>
        }
      }
    }
    return
  }


// func.func @matmul(%arg0: memref<?x400xf32>, %arg1: memref<?x300xf32>,
//                   %arg2: memref<?x300xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
//     affine.for %arg3 = 0 to 200 {
//       affine.for %arg4 = 0 to 300 {
//         affine.for %arg5 = 0 to 400 {
//           %0 = affine.load %arg0[%arg3, %arg5] : memref<?x400xf32>
//           %1 = affine.load %arg1[%arg5, %arg4] : memref<?x300xf32>
//           %2 = arith.mulf %0, %1 : f32
//           %3 = affine.load %arg2[%arg5, %arg4] : memref<?x300xf32>
//           %4 = arith.addf %3, %2 : f32
//           affine.store %4, %arg2[%arg5, %arg4] : memref<?x300xf32>
//         }
//       }
//     }
//     return
//   }