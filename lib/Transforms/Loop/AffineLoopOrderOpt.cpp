//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Transforms/LoopUtils.h"
#include "scalehls/Analysis/Utils.h"
#include "scalehls/Transforms/Passes.h"

using namespace mlir;
using namespace scalehls;

bool scalehls::applyAffineLoopOrderOpt(AffineLoopBand &band, bool reverse) {
  if (!isPerfectlyNested(band))
    return false;

  auto &loopBlock = *band.back().getBody();
  auto bandDepth = band.size();

  // Collect all load and store operations for each memory in the loop block,
  // and calculate the number of common surrouding loops for later uses.
  MemAccessesMap loadStoresMap;
  getMemAccessesMap(loopBlock, loadStoresMap);
  auto commonLoopDepth = getNumCommonSurroundingLoops(
      *loopBlock.begin(), *std::next(loopBlock.begin()));

  // A map of dependency distances indexed by the loop in the band.
  SmallVector<AffineForOp, 8> targetLoops;
  llvm::SmallDenseMap<Operation *, unsigned, 8> distanceMap;

  //  Only the loops in the loop band will be checked.
  unsigned startDepth = commonLoopDepth - bandDepth + 1;
  for (unsigned depth = startDepth; depth < commonLoopDepth + 1; ++depth) {
    auto loop = band[depth - startDepth];
    unsigned minDistance = UINT_MAX;

    // Traverse all memories in the loop block and find all dependencies
    // associated to each memory.
    for (auto pair : loadStoresMap) {
      auto loadStores = pair.second;

      int64_t dstIndex = 1;
      for (auto dstOp : loadStores) {
        for (auto srcOp : llvm::drop_begin(loadStores, dstIndex)) {
          MemRefAccess dstAccess(dstOp);
          MemRefAccess srcAccess(srcOp);

          FlatAffineConstraints depConstrs;
          SmallVector<DependenceComponent, 2> depComps;

          DependenceResult result = checkMemrefAccessDependence(
              srcAccess, dstAccess, depth, &depConstrs, &depComps);

          if (hasDependence(result)) {
            auto depComp = depComps[depth - 1];
            assert(loop == depComp.op && "unexpected dependency");

            // Only positive distance will be recorded.
            if (depComp.ub.getValue() > 0) {
              unsigned distance = std::max(depComp.lb.getValue(), (int64_t)1);
              minDistance = std::min(minDistance, distance);
            }
          }
        }
        dstIndex++;
      }
    }

    // Collect all candidate loops into an ordered vector. Loop with the
    // smallest distance will appear in the front.
    if (minDistance < UINT_MAX) {
      distanceMap[loop] = minDistance;

      for (auto it = targetLoops.begin(); it <= targetLoops.end(); ++it)
        if (it == targetLoops.end()) {
          targetLoops.push_back(loop);
          break;
        } else if (minDistance < distanceMap[*it]) {
          targetLoops.insert(it, loop);
          break;
        }
    }
  }

  distanceMap.clear();

  // Permute the target loops one by one.
  // TODO: a more comprehensive permution strategy search.
  for (auto loop : targetLoops) {
    unsigned targetLoopLoc =
        std::find(band.begin(), band.end(), loop) - band.begin();

    if (!reverse)
      // Permute the target loop to an as outer as possible location.
      for (unsigned dstLoc = 0; dstLoc < targetLoopLoc; ++dstLoc) {
        SmallVector<unsigned, 4> permMap;

        // Construct permutation map.
        for (unsigned loc = 0; loc < bandDepth; ++loc) {
          if (loc < dstLoc)
            permMap.push_back(loc);
          else if (loc < targetLoopLoc)
            permMap.push_back(loc + 1);
          else if (loc == targetLoopLoc)
            permMap.push_back(dstLoc);
          else
            permMap.push_back(loc);
        }

        // Check the validation of the current permutation.
        if (isValidLoopInterchangePermutation(band, permMap)) {
          auto newRoot = band[permuteLoops(band, permMap)];
          band.clear();
          getLoopBandFromRoot(newRoot, band);
          break;
        }
      }
    else
      // Permute the target loop to an as inner as possible location.
      for (unsigned dstLoc = targetLoopLoc + 1; dstLoc < bandDepth; ++dstLoc) {
        SmallVector<unsigned, 4> permMap;

        // Construct permutation map.
        for (unsigned loc = 0; loc < bandDepth; ++loc) {
          if (loc < targetLoopLoc)
            permMap.push_back(loc);
          else if (loc == targetLoopLoc)
            permMap.push_back(dstLoc);
          else if (loc <= dstLoc)
            permMap.push_back(loc - 1);
          else
            permMap.push_back(loc);
        }

        // Check the validation of the current permutation.
        if (isValidLoopInterchangePermutation(band, permMap)) {
          auto newRoot = band[permuteLoops(band, permMap)];
          band.clear();
          getLoopBandFromRoot(newRoot, band);
          break;
        }
      }
  }
  return true;
}

namespace {
struct AffineLoopOrderOpt : public AffineLoopOrderOptBase<AffineLoopOrderOpt> {
  void runOnOperation() override {
    auto func = getOperation();

    // Collect all target loop bands.
    AffineLoopBands targetBands;
    getLoopBands(func.front(), targetBands);

    // Apply loop order optimization to each loop band.
    for (auto band : targetBands)
      applyAffineLoopOrderOpt(band);
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createAffineLoopOrderOptPass() {
  return std::make_unique<AffineLoopOrderOpt>();
}