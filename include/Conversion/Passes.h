//===------------------------------------------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef SCALEHLS_CONVERSION_PASSES_H
#define SCALEHLS_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir {
namespace scalehls {

std::unique_ptr<mlir::Pass> createConvertToHLSCppPass();
std::unique_ptr<mlir::Pass> createHLSKernelToAffinePass();

void registerConversionPasses();

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

} // namespace scalehls
} // namespace mlir

#endif // SCALEHLS_CONVERSION_PASSES_H