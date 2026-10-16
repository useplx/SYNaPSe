#pragma once

#include "mln/sdk/functional/map.hpp"
#include "syn/operation.hpp"

namespace Syn::Compiler::Functional {
    class Map : public Operation::For<Mln::Sdk::Functional::Map> {
        public:
            mlir::Value emit_typed(const Mln::Sdk::Functional::Map &map, Operations &ctx,
                                   mlir::OpBuilder &builder, mlir::Location loc) const override;
    };

    mlir::Value emit_gathered_map(const Mln::Sdk::Functional::Map &map, mlir::Value indices,
                                  int64_t axis, Operations &ctx, mlir::OpBuilder &builder,
                                  mlir::Location loc);

    mlir::Value emit_map_scalar(const Mln::Sdk::Functional::Map &map, mlir::ValueRange arguments,
                                mlir::OpBuilder &builder, mlir::Location location,
                                mlir::Type computation_float_type = {});

    const Mln::Sdk::Functional::Map *row_addressable_map(Mln::Sdk::Tensor decoded);
}
