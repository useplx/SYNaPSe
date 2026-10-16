#pragma once

#include "mln/sdk/functional/equal.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Equal : public Operation::For<Mln::Sdk::Functional::Equal> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Equal &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
