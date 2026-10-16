#pragma once

#include "mln/sdk/functional/reshape.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Reshape : public Operation::For<Mln::Sdk::Functional::Reshape> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Reshape &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
