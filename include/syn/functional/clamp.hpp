#pragma once

#include "mln/sdk/functional/clamp.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Clamp : public Operation::For<Mln::Sdk::Functional::Clamp> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Clamp &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
