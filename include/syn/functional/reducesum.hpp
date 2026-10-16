#pragma once

#include "mln/sdk/functional/reducesum.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class ReduceSum : public Operation::For<Mln::Sdk::Functional::ReduceSum> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::ReduceSum &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
