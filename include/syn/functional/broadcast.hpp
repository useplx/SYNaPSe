#pragma once

#include "mln/sdk/functional/broadcast.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Broadcast : public Operation::For<Mln::Sdk::Functional::Broadcast> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Broadcast &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
