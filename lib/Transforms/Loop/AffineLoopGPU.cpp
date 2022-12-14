#include "mlir/IR/Operation.h"


namespace {
struct AffineLoopPermute : public impl::AffineLoopPermuteBase<AffineLoopPermute> {
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

std::unique_ptr<InterfacePass<FunctionOpInterface>> mlir::createAffineLoopPermutePass() {
  return std::make_unique<AffineLoopPermute>();
}

/*
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
void mlir::registerScaleCUDAPipeline() {
    mlir::PassPipelineRegistration<ScaleCUDAPipelineOptions>(
    "scalecuda-pipeline", "Optimize Affine on the GPU dialect", [](OpPassManager &pm, const ScaleCUDAPipelineOptions &opts) {
      pm.addPass(mlir::createFooPassPass());
      pm.addPass(mlir::createAffineForToGPUPass());
    });
}

*/