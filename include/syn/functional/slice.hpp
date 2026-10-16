#pragma once

#include "mln/sdk/functional/slice.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Slice : public Operation::For<Mln::Sdk::Functional::Slice> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Slice &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
