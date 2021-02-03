//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "scalehls/Analysis/Utils.h"
#include "scalehls/Dialect/HLSCpp/HLSCpp.h"
#include "scalehls/Transforms/Passes.h"

using namespace mlir;
using namespace scalehls;
using namespace hlscpp;

static bool applyArrayPartition(FuncOp func, OpBuilder &builder) {
  // Check whether the input function is pipelined.
  bool funcPipeline = false;
  if (auto attr = func->getAttrOfType<BoolAttr>("pipeline"))
    if (attr.getValue())
      funcPipeline = true;

  // Only memory accesses in pipelined loops or function will be executed in
  // parallel and required to partition.
  SmallVector<Block *, 4> pipelinedBlocks;
  if (funcPipeline)
    pipelinedBlocks.push_back(&func.front());
  else
    func.walk([&](AffineForOp loop) {
      if (auto attr = loop->getAttrOfType<BoolAttr>("pipeline"))
        if (attr.getValue())
          pipelinedBlocks.push_back(loop.getBody());
    });

  // Storing the partition information of each memref.
  using PartitionInfo = std::pair<PartitionKind, int64_t>;
  DenseMap<Value, SmallVector<PartitionInfo, 4>> partitionsMap;

  // Traverse all pipelined loops.
  for (auto block : pipelinedBlocks) {
    MemAccessesMap accessesMap;
    getMemAccessesMap(*block, accessesMap);

    for (auto pair : accessesMap) {
      auto memref = pair.first;
      auto memrefType = memref.getType().cast<MemRefType>();
      auto loadStores = pair.second;
      auto &partitions = partitionsMap[memref];

      // If the current partitionsMap is empty, initialize it with no partition
      // and factor of 1.
      if (partitions.empty()) {
        for (int64_t dim = 0; dim < memrefType.getRank(); ++dim)
          partitions.push_back(PartitionInfo(PartitionKind::NONE, 1));
      }

      // TODO: the issue is the same dimension of different memref accesses
      // represent different value. Therefore, the two memref access map need to
      // be somehow merged to keep things correct.
      // Find the best partition solution for each dimensions of the memref.
      for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
        // Collect all array access indices of the current dimension.
        SmallVector<AffineExpr, 4> indices;
        for (auto accessOp : loadStores) {
          // Get memory access map.
          AffineValueMap accessMap;
          MemRefAccess(accessOp).getAccessMap(&accessMap);

          // Get index expression.
          auto index = accessMap.getResult(dim);

          // Only add unique index.
          if (std::find(indices.begin(), indices.end(), index) == indices.end())
            indices.push_back(index);
        }
        auto accessNum = indices.size();

        // Find the max array access distance in the current block.
        unsigned maxDistance = 0;
        bool requireMux = false;

        for (unsigned i = 0; i < accessNum; ++i) {
          for (unsigned j = i + 1; j < accessNum; ++j) {
            // TODO: this expression can't be simplified in some cases.
            AffineExpr expr;
            auto index = indices[i];

            if (index.getKind() == AffineExprKind::Add) {
              auto addExpr = index.dyn_cast<AffineBinaryOpExpr>();
              expr = indices[j] - addExpr.getLHS() - addExpr.getRHS();
            } else
              expr = indices[j] - index;

            if (auto constDistance = expr.dyn_cast<AffineConstantExpr>()) {
              unsigned distance = std::abs(constDistance.getValue());
              maxDistance = std::max(maxDistance, distance);
            } else
              requireMux = true;
          }
        }

        // Determine array partition strategy.
        // TODO: take storage type into consideration.
        // TODO: the partition strategy requires more case study.
        maxDistance++;
        if (maxDistance == 1) {
          // This means all accesses have the same index, and this dimension
          // should not be partitioned.
          continue;

        } else if (accessNum >= maxDistance) {
          // This means some elements are accessed more than once or exactly
          // once, and successive elements are accessed. In most cases, apply
          // "cyclic" partition should be the best solution.
          unsigned factor = maxDistance;
          if (factor > partitions[dim].second) {
            // The rationale here is if the accessing partition index cannot be
            // determined and partition factor is more than 3, a multiplexer
            // will be generated and the memory access operation will be wrapped
            // into a function call, which will cause dependency problems and
            // make the latency and II even worse.
            if (requireMux)
              for (auto i = 3; i > 0; --i) {
                if (factor % i == 0) {
                  partitions[dim] = PartitionInfo(PartitionKind::CYCLIC, i);
                  break;
                }
              }
            else
              partitions[dim] = PartitionInfo(PartitionKind::CYCLIC, factor);
          }
        } else {
          // This means discrete elements are accessed. Typically, "block"
          // partition will be most benefit for this occasion.
          unsigned factor = accessNum;
          if (factor > partitions[dim].second) {
            if (requireMux)
              for (auto i = 3; i > 0; --i) {
                if (factor % i == 0) {
                  partitions[dim] = PartitionInfo(PartitionKind::BLOCK, i);
                  break;
                }
              }
            else
              partitions[dim] = PartitionInfo(PartitionKind::BLOCK, factor);
          }
        }
      }
    }
  }

  // Constuct and set new type to each partitioned MemRefType.
  for (auto pair : partitionsMap) {
    auto memref = pair.first;
    auto memrefType = memref.getType().cast<MemRefType>();
    auto partitions = pair.second;

    // Walk through each dimension of the current memory.
    SmallVector<AffineExpr, 4> partitionIndices;
    SmallVector<AffineExpr, 4> addressIndices;

    for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
      auto partition = partitions[dim];
      auto kind = partition.first;
      auto factor = partition.second;

      if (kind == PartitionKind::CYCLIC) {
        partitionIndices.push_back(builder.getAffineDimExpr(dim) % factor);
        addressIndices.push_back(
            builder.getAffineDimExpr(dim).floorDiv(factor));

      } else if (kind == PartitionKind::BLOCK) {
        auto blockFactor = (memrefType.getShape()[dim] + factor - 1) / factor;
        partitionIndices.push_back(
            builder.getAffineDimExpr(dim).floorDiv(blockFactor));
        addressIndices.push_back(builder.getAffineDimExpr(dim) % blockFactor);

      } else {
        partitionIndices.push_back(builder.getAffineConstantExpr(0));
        addressIndices.push_back(builder.getAffineDimExpr(dim));
      }
    }

    // Construct new layout map.
    partitionIndices.append(addressIndices.begin(), addressIndices.end());
    auto layoutMap = AffineMap::get(memrefType.getRank(), 0, partitionIndices,
                                    builder.getContext());

    // Construct new memref type.
    auto newType =
        MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                        layoutMap, memrefType.getMemorySpace());

    // Set new type.
    memref.setType(newType);
  }

  // Align function type with entry block argument types.
  auto resultTypes = func.front().getTerminator()->getOperandTypes();
  auto inputTypes = func.front().getArgumentTypes();
  func.setType(builder.getFunctionType(inputTypes, resultTypes));

  // TODO: how to handle the case when different sub-functions have different
  // array partition strategy selected?
  return true;
}

namespace {
struct ArrayPartition : public ArrayPartitionBase<ArrayPartition> {
  void runOnOperation() override {
    auto func = getOperation();
    auto builder = OpBuilder(func);

    applyArrayPartition(func, builder);
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createArrayPartitionPass() {
  return std::make_unique<ArrayPartition>();
}