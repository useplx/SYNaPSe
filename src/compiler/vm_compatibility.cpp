#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace Syn::Compiler {
    namespace {
        class NormalizeSwitchIndex final
            : public mlir::PassWrapper<NormalizeSwitchIndex, mlir::OperationPass<mlir::ModuleOp>> {
            public:
                MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeSwitchIndex)

                [[nodiscard]] llvm::StringRef getArgument() const final {
                    return "syn-normalize-cf-switch-index";
                }

                [[nodiscard]] llvm::StringRef getDescription() const final {
                    return "normalizes HAL executable selection for the IREE VM ABI";
                }

                void runOnOperation() final {
                    llvm::SmallVector<mlir::cf::SwitchOp> switches;
                    getOperation().walk(
                        [&](mlir::cf::SwitchOp operation) { switches.push_back(operation); });

                    for (mlir::cf::SwitchOp operation : switches) {
                        mlir::Operation *cast = operation.getFlag().getDefiningOp();
                        if (!cast || cast->getName().getStringRef() != "arith.index_cast" ||
                            cast->getNumOperands() != 1 ||
                            !cast->getOperand(0).getType().isIndex() ||
                            !operation.getFlag().getType().isInteger(64)) {
                            continue;
                        }

                        const std::optional<mlir::DenseIntElementsAttr> case_values =
                            operation.getCaseValues();
                        if (!case_values)
                            continue;

                        llvm::SmallVector<std::int32_t> cases;
                        cases.reserve(case_values->getNumElements());
                        for (const llvm::APInt &value : case_values->getValues<llvm::APInt>()) {
                            if (!value.isSignedIntN(32)) {
                                operation.emitOpError("switch case does not fit the VM i32 ABI");
                                signalPassFailure();
                                return;
                            }
                            cases.push_back(static_cast<std::int32_t>(value.getSExtValue()));
                        }

                        mlir::OpBuilder builder(operation);
                        mlir::Value flag = mlir::arith::IndexCastOp::create(
                            builder, operation.getLoc(), builder.getI32Type(), cast->getOperand(0));
                        llvm::SmallVector<mlir::Block *> destinations(
                            operation.getCaseDestinations().begin(),
                            operation.getCaseDestinations().end());
                        llvm::SmallVector<mlir::ValueRange> operands;
                        for (mlir::OperandRange range : operation.getCaseOperands())
                            operands.emplace_back(range);
                        mlir::cf::SwitchOp::create(
                            builder, operation.getLoc(), flag, operation.getDefaultDestination(),
                            operation.getDefaultOperands(), builder.getI32VectorAttr(cases),
                            destinations, operands);
                        operation.erase();
                        if (cast->use_empty())
                            cast->erase();
                    }
                }
        };
    }

    void register_vm_compatibility_passes() {
        static mlir::PassRegistration<NormalizeSwitchIndex> registration;
        (void)registration;
    }
}
