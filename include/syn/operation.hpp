#pragma once

#include "mln/sdk/kind.hpp"
#include "mln/sdk/operation.hpp"
#include "mln/sdk/operation/compatibility.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

#include <cstdint>
#include <span>
#include <vector>

namespace Syn::Compiler {
    class Operations;
    struct Police;

    namespace Operation {
        class Emitter {
            protected:
                explicit Emitter(Mln::Sdk::Functional::Kind operation_kind)
                    : kind(operation_kind) {}

            public:
                const Mln::Sdk::Functional::Kind kind;

                virtual mlir::Value emit(const Mln::Sdk::Operation &node, Operations &ctx,
                                         mlir::OpBuilder &builder,
                                         mlir::Location location) const = 0;
                virtual std::vector<mlir::Value> emit_results(const Mln::Sdk::Operation &node,
                                                              Operations &ctx,
                                                              mlir::OpBuilder &builder,
                                                              mlir::Location location) const {
                    return {emit(node, ctx, builder, location)};
                }
                virtual ~Emitter() = default;
        };

        template <typename Node> class MultiFor : public Emitter {
            protected:
                MultiFor() : Emitter(Node::kind_value) {}

                virtual std::vector<mlir::Value> emit_typed(const Node &node, Operations &ctx,
                                                            mlir::OpBuilder &builder,
                                                            mlir::Location location) const = 0;

            private:
                mlir::Value emit(const Mln::Sdk::Operation &node, Operations &ctx,
                                 mlir::OpBuilder &builder, mlir::Location location) const final {
                    auto results =
                        emit_typed(static_cast<const Node &>(node), ctx, builder, location);
                    return results.empty() ? mlir::Value{} : results.front();
                }

                std::vector<mlir::Value> emit_results(const Mln::Sdk::Operation &node,
                                                      Operations &ctx, mlir::OpBuilder &builder,
                                                      mlir::Location location) const final {
                    return emit_typed(static_cast<const Node &>(node), ctx, builder, location);
                }
        };

        template <typename Node> class For : public Emitter {
            protected:
                For() : Emitter(Node::kind_value) {}

                virtual mlir::Value emit_typed(const Node &node, Operations &ctx,
                                               mlir::OpBuilder &builder,
                                               mlir::Location location) const = 0;

            private:
                mlir::Value emit(const Mln::Sdk::Operation &node, Operations &ctx,
                                 mlir::OpBuilder &builder, mlir::Location location) const final {
                    return emit_typed(static_cast<const Node &>(node), ctx, builder, location);
                }
        };
    }
}
