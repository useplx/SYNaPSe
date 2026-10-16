#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace Syn::Compiler {
    namespace {
        class PropagateDispatchAffinity final
            : public mlir::PassWrapper<PropagateDispatchAffinity,
                                       mlir::OperationPass<mlir::ModuleOp>> {
            public:
                MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropagateDispatchAffinity)

                [[nodiscard]] llvm::StringRef getArgument() const final {
                    return "syn-propagate-dispatch-affinity";
                }

                [[nodiscard]] llvm::StringRef getDescription() const final {
                    return "propagates resolved PLeX placement to IREE dispatch regions";
                }

                void runOnOperation() final {
                    bool failed = false;
                    getOperation().walk([&](mlir::Operation *operation) {
                        if (failed ||
                            operation->getName().getStringRef() != "flow.dispatch.workgroups") {
                            return;
                        }

                        mlir::Attribute affinity;
                        operation->walk([&](mlir::Operation *nested) {
                            if (nested == operation || failed)
                                return;
                            mlir::Attribute candidate = nested->getAttr("stream.affinity");
                            if (!candidate)
                                return;
                            if (!affinity) {
                                affinity = candidate;
                                return;
                            }
                            if (affinity != candidate) {
                                operation->emitOpError(
                                    "contains operations assigned to different devices; "
                                    "a cross-device optimization boundary is missing");
                                failed = true;
                            }
                        });
                        if (affinity && !failed)
                            operation->setAttr("stream.affinity", affinity);
                    });
                    if (failed)
                        signalPassFailure();
                }
        };
    }

    void register_dispatch_affinity_passes() {
        static mlir::PassRegistration<PropagateDispatchAffinity> registration;
        (void)registration;
    }
}
