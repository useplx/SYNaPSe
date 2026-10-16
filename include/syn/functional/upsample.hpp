#pragma once

#include "mln/sdk/functional/upsample.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Upsample : public Operation::For<Mln::Sdk::Functional::Upsample> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Upsample &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
