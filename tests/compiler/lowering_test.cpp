#include "mln/error.hpp"
#include "syn/lowering.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
    class Compatibility final : public Mln::Sdk::Lowering::Compatibility {
        public:
            Compatibility(std::string_view source, std::string_view target, int priority = 0)
                : source_(source), target_(target), priority_(priority) {}

            std::string_view source() const noexcept override { return source_; }
            std::string_view target() const noexcept override { return target_; }
            int priority() const noexcept override { return priority_; }
            bool applies(const Mln::Sdk::Backend &backend) const override {
                return backend.device == "qnn-htp";
            }
            bool lower(mlir::ModuleOp, const Context &context) const override {
                ++calls;
                operation_count = context.operations.size();
                return true;
            }

            mutable int calls = 0;
            mutable std::size_t operation_count = 0;

        private:
            std::string_view source_;
            std::string_view target_;
            int priority_;
    };
}

int main() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    mlir::ModuleOp module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    Mln::Sdk::Backend backend{.device = "qnn-htp"};
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToStart(module.getBody());
    auto function = mlir::func::FuncOp::create(builder, module.getLoc(), "npu",
                                               builder.getFunctionType({}, {}));
    function->setAttr("myelin.backend", builder.getStringAttr("qnn-htp:0"));
    auto cpu = mlir::func::FuncOp::create(builder, module.getLoc(), "cpu",
                                          builder.getFunctionType({}, {}));
    cpu->setAttr("myelin.backend", builder.getStringAttr("local:0"));

    Compatibility to_tosa("stablehlo", "tosa");
    Compatibility to_qnn("tosa", "qnn-graph");
    std::vector<Mln::Sdk::Lowering::Compatibility *> chain{&to_tosa, &to_qnn};
    const std::string result = Syn::Compiler::Lowering::apply(module, backend, chain);
    if (result != "qnn-graph" || to_tosa.calls != 1 || to_qnn.calls != 1 ||
        to_tosa.operation_count != 1 || to_qnn.operation_count != 1)
        throw std::runtime_error("lowering chain was not applied in representation order");

    Compatibility direct_a("stablehlo", "qnn-a");
    Compatibility direct_b("stablehlo", "qnn-b");
    chain = {&direct_a, &direct_b};
    try {
        (void)Syn::Compiler::Lowering::apply(module, backend, chain);
    } catch (const Mln::Diagnostic &) {
        return 0;
    }
    throw std::runtime_error("ambiguous lowering chain was accepted");
}
