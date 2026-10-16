#pragma once

#include "mln/sdk/functional/dynamicslice.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class DynamicSlice : public Operation::For<Mln::Sdk::Functional::DynamicSlice> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::DynamicSlice &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
