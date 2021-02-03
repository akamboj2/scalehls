//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "scalehls/Analysis/Utils.h"
#include "scalehls/Transforms/Passes.h"

using namespace mlir;
using namespace scalehls;

/// Fully unroll all loops insides of a block.
bool scalehls::applyFullyLoopUnrolling(Block &block) {
  // Try 8 iterations before exiting.
  for (auto i = 0; i < 8; ++i) {
    bool hasFullyUnrolled = true;
    block.walk([&](AffineForOp loop) {
      if (failed(loopUnrollFull(loop)))
        hasFullyUnrolled = false;
    });

    if (hasFullyUnrolled)
      break;

    if (i == 7)
      return false;
  }
  return true;
}

/// Apply loop pipelining to the input loop, all inner loops are automatically
/// fully unrolled.
bool scalehls::applyLoopPipelining(AffineForOp targetLoop, int64_t targetII,
                                   OpBuilder &builder) {
  // All inner loops of the pipelined loop are automatically unrolled.
  if (!applyFullyLoopUnrolling(*targetLoop.getBody()))
    return false;

  targetLoop->setAttr("pipeline", builder.getBoolAttr(true));
  targetLoop->setAttr("target_ii", builder.getI64IntegerAttr(targetII));

  // All outer loops that perfect nest the pipelined loop can be flattened.
  SmallVector<AffineForOp, 4> flattenedLoops;
  flattenedLoops.push_back(targetLoop);
  while (true) {
    auto currentLoop = flattenedLoops.back();
    if (auto outerLoop = currentLoop->getParentOfType<AffineForOp>()) {
      // Only if the current loop is the only child loop of the outer loop, the
      // outer loop can be flattened into the current loop.
      auto &body = *outerLoop.getBody();
      if (&body.front() == currentLoop && body.getOperations().size() == 2) {
        flattenedLoops.push_back(outerLoop);
        outerLoop->setAttr("flatten", builder.getBoolAttr(true));
      } else
        break;
    } else
      break;
  }

  return true;
}

namespace {
struct LoopPipelining : public LoopPipeliningBase<LoopPipelining> {
  void runOnOperation() override {
    auto func = getOperation();
    auto builder = OpBuilder(func);

    // Collect all innermost loops.
    SmallVector<AffineForOp, 4> innermostLoops;
    func.walk([&](AffineForOp loop) {
      if (getChildLoopNum(loop) == 0)
        innermostLoops.push_back(loop);
    });

    // Apply loop pipelining to coresponding level of each innermost loop.
    for (auto loop : innermostLoops) {
      auto currentLoop = loop;
      unsigned loopLevel = 0;
      while (true) {
        auto parentLoop = currentLoop->getParentOfType<AffineForOp>();

        // If meet the outermost loop, pipeline the current loop.
        if (!parentLoop || pipelineLevel == loopLevel) {
          applyLoopPipelining(currentLoop, targetII, builder);
          break;
        }

        // Move to the next loop level.
        currentLoop = parentLoop;
        ++loopLevel;
      }
    }

    // Canonicalize the IR after loop pipelining.
    OwningRewritePatternList patterns;
    for (auto *op : func.getContext()->getRegisteredOperations())
      op->getCanonicalizationPatterns(patterns, func.getContext());

    applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createLoopPipeliningPass() {
  return std::make_unique<LoopPipelining>();
}