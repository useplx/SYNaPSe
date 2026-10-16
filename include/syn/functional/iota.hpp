#pragma once

#include "mln/sdk/functional/iota.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Iota : public Operation::For<Mln::Sdk::Functional::Iota> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Iota &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
