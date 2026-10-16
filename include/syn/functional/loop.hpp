#pragma once

#include "mln/sdk/functional/loop.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Loop : public Operation::For<Mln::Sdk::Functional::Loop> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Loop &loop, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
