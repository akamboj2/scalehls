#include <iostream>
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPU.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"
#include "llvm/Support/Debug.h"

//For AffineToMemRef
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Dialect/GPU/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::scf;
using namespace scalehls;

//To Run use: cd build
//            ninja check-scalehls 
//            scalehls-opt test.mlir --affine-to-gpu
//            scalehls-opt test.mlir --scalecuda-pipeline

//for this pass also need to modify /home/abhi/courses/ece527-SOC-Design/final_project/scalehls/include/scalehls/Transforms/Passes.td and Passes.h

class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
                                                *resultOperands);
    return success();
  }
};

class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
public:
   using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

   LogicalResult matchAndRewrite(AffineStoreOp op,
                                 PatternRewriter &rewriter) const override {
     // Expand affine map from 'affineStoreOp'.
     SmallVector<Value, 8> indices(op.getMapOperands());
     auto maybeExpandedMap =
         expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
     if (!maybeExpandedMap)
       return failure();

     // Build memref.store valueToStore, memref[expandedMap.results].
     rewriter.replaceOpWithNewOp<memref::StoreOp>(
         op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
     return success();
   }
};

namespace {
struct AffineToMemref : public AffineToMemrefBase<AffineToMemref> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<AffineLoadLowering, AffineStoreLowering>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
}


namespace {
struct AffineLoopPermute : public AffineLoopPermuteBase<AffineLoopPermute> {
  void runOnOperation() override {
    std::vector<AffineForOp> inputNest;
    Operation * op = getOperation();
    op->walk([&](Operation *inst){
      if (auto forOp = dyn_cast<AffineForOp>(inst)) 
      {
        inputNest.insert(inputNest.begin(), forOp);
    	}
    });

    std::vector<unsigned> permMap;
    unsigned idx = 0;
    for(unsigned i = 0; i < inputNest.size(); i++)
    {
      permMap.push_back(0);
      if(isLoopMemoryParallel(inputNest[i]))
      {
        inputNest[i]->removeAttr("point");
        permMap[i] = idx;
        idx++;
      }
    }
    for(unsigned j = 0; j < inputNest.size(); j++)
    {
      if(!isLoopMemoryParallel(inputNest[j]))
      {
        permMap[j] = idx;
        idx++;
      }
    }
    ArrayRef<unsigned> permMapAR(permMap);
    MutableArrayRef<AffineForOp> inputNestMAR(inputNest);
    if(isValidLoopInterchangePermutation(inputNestMAR, permMapAR))
    {
      permuteLoops(inputNestMAR, permMapAR);
      // printf("input nest");
    }
    else
    {
      op->emitOpError("Invalid Loop Interchange Permutation\n");
    }

    // printf("finished foo-pass\n");
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createAffineToMemrefPass() {
  return std::make_unique<AffineToMemref>();
}

std::unique_ptr<Pass> scalehls::createAffineLoopPermutePass() {
  return std::make_unique<AffineLoopPermute>();
}



namespace{
  struct ScaleCUDAPipelineOptions : public PassPipelineOptions<ScaleCUDAPipelineOptions> {
    // The structure of these options is the same as those for pass options.
    // Option<int> exampleOption{*this, "flag-name", llvm::cl::desc("...")};
    // ListOption<int> exampleListOption{*this, "list-flag-name",
    //                                   llvm::cl::desc("...")};
  };
}

//NOTE: also added a call to this in: /home/abhi/courses/ece527-SOC-Design/final_project/scalehls/lib/Transforms/Passes.cpp 
// and a declaration in: /home/abhi/courses/ece527-SOC-Design/final_project/scalehls/include/scalehls/Transforms/Passes.h
void mlir::scalehls::registerScaleCUDAPipeline() {
    mlir::PassPipelineRegistration<ScaleCUDAPipelineOptions>(
    "scalecuda-pipeline", "Optimize Affine on the GPU dialect", [](OpPassManager &pm, const ScaleCUDAPipelineOptions &opts) {
    
    // Affine loop tiling.
    // pm.addPass(scalehls::createFuncPreprocessPass(opts.hlsTopFunc));
      pm.addPass(bufferization::createBufferLoopHoistingPass());
      // pm.addPass(scalehls::createMaterializeReductionPass()); //note the following two were in attempt to make conv work
      // scalehls::addSimplifyAffineLoopPasses(pm); 
      pm.addPass(scalehls::createAffineLoopPerfectionPass());
      pm.addPass(scalehls::createAffineLoopOrderOptPass());
      pm.addPass(scalehls::createAffineLoopTilePass(32));
      pm.addPass(mlir::createSimplifyAffineStructuresPass());
      pm.addPass(mlir::createCanonicalizerPass());

      pm.addPass(scalehls::createAffineLoopPermutePass());

      // // // Local buffer allocation.
      scalehls::addCreateSubviewPasses(pm);
      pm.addPass(scalehls::createCreateLocalBufferPass(false));
      pm.addPass(scalehls::createLowerCopyToAffinePass());
      pm.addPass(memref::createFoldMemRefAliasOpsPass());
      pm.addPass(mlir::createSimplifyAffineStructuresPass());
      pm.addPass(mlir::createCanonicalizerPass());


      pm.addPass(scalehls::createAffineToMemrefPass());
      pm.addNestedPass<func::FuncOp>(mlir::createAffineForToGPUPass(0,2));

      pm.addPass(mlir::createLowerAffinePass());
      pm.addPass(mlir::createConvertSCFToCFPass());
      pm.addPass(mlir::cf::createConvertControlFlowToLLVMPass());
      pm.addPass(mlir::createReconcileUnrealizedCastsPass()); //AK: https://discourse.llvm.org/t/how-to-convert-affine-dialect-llvm-ir/4710
      pm.addPass(mlir::createConvertIndexToLLVMPass());
      pm.addPass(mlir::createReconcileUnrealizedCastsPass()); //AK: https://discourse.llvm.org/t/how-to-convert-affine-dialect-llvm-ir/4710
      pm.addPass(mlir::createArithToLLVMConversionPass());
      pm.addPass(mlir::createReconcileUnrealizedCastsPass()); //AK: https://discourse.llvm.org/t/how-to-convert-affine-dialect-llvm-ir/4710
      pm.addPass(mlir::createMemRefToLLVMConversionPass());
      pm.addPass(mlir::createCanonicalizerPass());
      
      //for gpu exectuion
      pm.addPass(mlir::createGpuKernelOutliningPass());
      pm.addNestedPass<gpu::GPUModuleOp>(mlir::createStripDebugInfoPass());
      pm.addNestedPass<gpu::GPUModuleOp>(mlir::createLowerGpuOpsToNVVMOpsPass());
      pm.addNestedPass<gpu::GPUModuleOp>(mlir::createGpuSerializeToCubinPass("nvptx64-nvidia-cuda","sm_35","+ptx60"));       // (StringRef triple, StringRef chip, StringRef features)
      pm.addPass(mlir::createGpuAsyncRegionPass());
      pm.addPass(mlir::createGpuToLLVMConversionPass());    
    });
}

/*
Get Any kernels running on a GPU:
- compare it functionally to make sure it's working
- get timing from GPU

Local Buffer: 
- we want the local buffer to be allocated in shared memory
- see how it is percieved in the next steps
- is there a way to make it shared memory if that's not what it is already.
- Establish pipeline: GPU-NVVM-GPUBinary

Get Convolution to Working

Install CUBLAS 
- linear algebra inputs, do a matmul or something
- maybe try with Pytorch working 


what is working:
Running unoptimized MLIR on GPU on (dir is scalehls): /home/abhi/courses/ece527-SOC-Design/final_project/buddy-mlir/llvm/build/bin/mlir-cpu-runner  test_unoptimized.mlir -entry-point-result=void -shared-libs=/home/abhi/courses/ece527-SOC-Design/final_project/buddy-mlir/llvm/build/lib/libmlir_runner_utils.so -shared-libs=/home/abhi/courses/ece527-SOC-Design/final_project/buddy-mlir/llvm/build/lib/libmlir_cuda_runtime.so
Running their version of matrix multiplication in buddy mlir: make gpu-mma-run in buddymlir/examples/mlirgpu directory

*/