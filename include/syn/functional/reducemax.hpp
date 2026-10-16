#pragma once

#include "mln/sdk/functional/reducemax.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class ReduceMax : public Operation::For<Mln::Sdk::Functional::ReduceMax> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::ReduceMax &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
