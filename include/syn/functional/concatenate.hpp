#pragma once

#include "mln/sdk/functional/concatenate.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {

    class Concatenate : public Operation::For<Mln::Sdk::Functional::Concatenate> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Concatenate &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
