#pragma once

#include "mln/sdk/functional/dynamicupdateslice.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class DynamicUpdateSlice : public Operation::For<Mln::Sdk::Functional::DynamicUpdateSlice> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::DynamicUpdateSlice &node,
                                   Operations &ctx, mlir::OpBuilder &builder,
                                   mlir::Location loc) const override;
    };
}
