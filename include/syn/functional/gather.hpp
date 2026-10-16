#pragma once

#include "mln/sdk/functional/gather.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Gather : public Operation::For<Mln::Sdk::Functional::Gather> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Gather &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
