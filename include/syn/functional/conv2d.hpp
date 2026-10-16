#pragma once

#include "mln/sdk/functional/conv2d.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {

    class Conv2D : public Operation::For<Mln::Sdk::Functional::Conv2D> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Conv2D &node, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };
}
