#pragma once

#include "mln/sdk/functional/causal_conv1d.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class CausalConv1D : public Operation::For<Mln::Sdk::Functional::CausalConv1D> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::CausalConv1D &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
