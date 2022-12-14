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

using namespace mlir;
using namespace mlir::scf;
using namespace scalehls;

//To Run use: scalehls-opt test.mlir --affine-to-gpu

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

    printf("finished foo-pass\n");
  }
};
} // namespace

// std::unique_ptr<InterfacePass<FunctionOpInterface>> mlir::createAffineLoopPermutePass() {
//   return std::make_unique<AffineLoopPermute>();
// }

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

//NOTE: also added a call to this in: llvm-project/mlir/include/mlir/InitAllPasses.h
// and a declaration in: llvm-project/mlir/include/mlir/Conversion/SCFToGPU/SCFToGPUPass.h
void mlir::scalehls::registerScaleCUDAPipeline() {
    mlir::PassPipelineRegistration<ScaleCUDAPipelineOptions>(
    "scalecuda-pipeline", "Optimize Affine on the GPU dialect", [](OpPassManager &pm, const ScaleCUDAPipelineOptions &opts) {
      pm.addPass(scalehls::createAffineLoopPermutePass());
      pm.addNestedPass<func::FuncOp>(mlir::createAffineForToGPUPass());
    });
}

