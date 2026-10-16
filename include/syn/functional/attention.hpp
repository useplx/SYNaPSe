#pragma once

#include "mln/sdk/functional/attention.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Attention : public Operation::For<Mln::Sdk::Functional::Attention> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Attention &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
