#include "syn/functional/gated_delta_rule.hpp"

#include "syn/operations.hpp"

#include "mln/sdk/functional.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace Syn::Compiler::Functional {
    Mln::Sdk::Tensor expand_gated_delta_chunk(Mln::Sdk::Operations &operations,
                                              const Mln::Sdk::Functional::GatedDeltaRule &node,
                                              Mln::Sdk::Tensor query, Mln::Sdk::Tensor key);

    namespace {
        namespace F = Mln::Sdk::Functional;

        Mln::Sdk::Tensor scalar(Mln::Sdk::Operations &operations, double value,
                                const std::vector<int64_t> &shape) {
            Mln::Sdk::Tensor constant = operations.insert<F::Const>(Mln::Sdk::Weight{"", {}, value},
                                                                    false, Mln::DType::Float32);
            return operations.insert<F::Broadcast>(std::move(constant), shape,
                                                   std::vector<int64_t>{});
        }

        Mln::Sdk::Tensor slice(Mln::Sdk::Operations &operations, const Mln::Sdk::Tensor &value,
                               int64_t begin, int64_t end) {
            return operations.insert<F::Slice>(value, std::vector<int64_t>{begin},
                                               std::vector<int64_t>{end}, std::vector<int64_t>{1});
        }

        Mln::Sdk::Tensor normalize(Mln::Sdk::Operations &operations, const Mln::Sdk::Tensor &value,
                                   int64_t tokens, int64_t heads, int64_t channels,
                                   double epsilon) {
            Mln::Sdk::Tensor copy = value;
            Mln::Sdk::Tensor squared = operations.insert<F::Multiply>(value, std::move(copy));
            Mln::Sdk::Tensor sum =
                operations.insert<F::ReduceSum>(std::move(squared), std::vector<int64_t>{2});
            sum = operations.insert<F::Add>(std::move(sum),
                                            scalar(operations, epsilon, {tokens, heads}));
            Mln::Sdk::Tensor inverse = operations.insert<F::Rsqrt>(std::move(sum));
            inverse = operations.insert<F::Broadcast>(std::move(inverse),
                                                      std::vector<int64_t>{tokens, heads, channels},
                                                      std::vector<int64_t>{0, 1});
            return operations.insert<F::Multiply>(value, std::move(inverse));
        }
    }

    mlir::Value GatedDeltaRule::emit_typed(const Mln::Sdk::Functional::GatedDeltaRule &node,
                                           Operations &ctx, mlir::OpBuilder &builder,
                                           mlir::Location loc) const {

        ctx.resolve(node.query, builder, loc);
        ctx.resolve(node.key, builder, loc);
        ctx.resolve(node.value, builder, loc);
        ctx.resolve(node.decay, builder, loc);
        ctx.resolve(node.beta, builder, loc);
        ctx.resolve(node.initial_state, builder, loc);

        Mln::Sdk::Operations &operations = ctx.expansion_container();
        operations.set_policy_path(node.policy_path);

        const int64_t tokens = node.query.shape[0];
        const int64_t heads = node.query.shape[1];
        const int64_t channels = node.query.shape[2];
        const int64_t output_elements = tokens * heads * channels;
        const int64_t state_elements = heads * channels * channels;

        Mln::Sdk::Tensor query =
            normalize(operations, node.query, tokens, heads, channels, node.normalization_epsilon);
        Mln::Sdk::Tensor key =
            normalize(operations, node.key, tokens, heads, channels, node.normalization_epsilon);
        query = operations.insert<F::Multiply>(
            std::move(query), scalar(operations, 1.0 / std::sqrt(static_cast<double>(channels)),
                                     {tokens, heads, channels}));

        if (tokens > 1 && ctx.expand(node)) {
            Mln::Sdk::Tensor packed =
                expand_gated_delta_chunk(operations, node, std::move(query), std::move(key));
            return ctx.resolve(packed, builder, loc);
        }

        if (tokens == 1) {
            const std::vector<int64_t> vector_shape{heads, channels};
            const std::vector<int64_t> matrix_shape{heads, channels, channels};
            query = operations.insert<F::Reshape>(std::move(query), vector_shape);
            key = operations.insert<F::Reshape>(std::move(key), vector_shape);
            Mln::Sdk::Tensor value = operations.insert<F::Reshape>(node.value, vector_shape);
            Mln::Sdk::Tensor decay = operations.insert<F::Exp>(
                operations.insert<F::Reshape>(node.decay, std::vector<int64_t>{heads}));
            decay = operations.insert<F::Broadcast>(std::move(decay), matrix_shape,
                                                    std::vector<int64_t>{0});
            Mln::Sdk::Tensor memory =
                operations.insert<F::Multiply>(node.initial_state, std::move(decay));

            Mln::Sdk::Tensor key_matrix =
                operations.insert<F::Broadcast>(key, matrix_shape, std::vector<int64_t>{0, 1});
            Mln::Sdk::Tensor recalled =
                operations.insert<F::Multiply>(memory, std::move(key_matrix));
            recalled =
                operations.insert<F::ReduceSum>(std::move(recalled), std::vector<int64_t>{1});
            Mln::Sdk::Tensor delta =
                operations.insert<F::Subtract>(std::move(value), std::move(recalled));
            Mln::Sdk::Tensor beta =
                operations.insert<F::Reshape>(node.beta, std::vector<int64_t>{heads});
            beta = operations.insert<F::Broadcast>(std::move(beta), vector_shape,
                                                   std::vector<int64_t>{0});
            delta = operations.insert<F::Multiply>(std::move(delta), std::move(beta));

            Mln::Sdk::Tensor update_key =
                operations.insert<F::Broadcast>(key, matrix_shape, std::vector<int64_t>{0, 1});
            Mln::Sdk::Tensor update_value =
                operations.insert<F::Broadcast>(delta, matrix_shape, std::vector<int64_t>{0, 2});
            memory = operations.insert<F::Add>(
                std::move(memory),
                operations.insert<F::Multiply>(std::move(update_key), std::move(update_value)));

            Mln::Sdk::Tensor read_query =
                operations.insert<F::Broadcast>(query, matrix_shape, std::vector<int64_t>{0, 1});
            Mln::Sdk::Tensor output = operations.insert<F::Multiply>(memory, std::move(read_query));
            output = operations.insert<F::ReduceSum>(std::move(output), std::vector<int64_t>{1});
            output = operations.insert<F::Reshape>(std::move(output),
                                                   std::vector<int64_t>{output_elements});
            Mln::Sdk::Tensor final_state = operations.insert<F::Reshape>(
                std::move(memory), std::vector<int64_t>{state_elements});
            Mln::Sdk::Tensor packed =
                operations.insert<F::Concatenate>(std::move(output), std::move(final_state), 0);
            return ctx.resolve(packed, builder, loc);
        }

        Mln::Sdk::Tensor carried =
            operations.insert<F::Reshape>(node.initial_state, std::vector<int64_t>{state_elements});
        carried = operations.insert<F::Concatenate>(std::move(carried),
                                                    scalar(operations, 0.0, {output_elements}), 0);

        carried = operations.insert<F::Loop>(
            std::move(carried), tokens,
            [=](Mln::Sdk::Operations &body, const Mln::Sdk::Tensor &index,
                const Mln::Sdk::Tensor &state) {
                Mln::Sdk::Tensor memory = slice(body, state, 0, state_elements);
                memory = body.insert<F::Reshape>(std::move(memory),
                                                 std::vector<int64_t>{heads, channels, channels});
                Mln::Sdk::Tensor outputs =
                    slice(body, state, state_elements, state_elements + output_elements);
                outputs = body.insert<F::Reshape>(std::move(outputs),
                                                  std::vector<int64_t>{tokens, heads, channels});

                Mln::Sdk::Tensor zero =
                    body.insert<F::Const>(Mln::Sdk::Weight{"", {}, 0.0}, false, Mln::DType::Int32);
                const std::vector<int64_t> token_shape{1, heads, channels};
                const std::vector<int64_t> gate_shape{1, heads};
                Mln::Sdk::Tensor query_token = body.insert<F::DynamicSlice>(
                    query, std::vector<Mln::Sdk::Tensor>{index, zero, zero}, token_shape);
                Mln::Sdk::Tensor key_token = body.insert<F::DynamicSlice>(
                    key, std::vector<Mln::Sdk::Tensor>{index, zero, zero}, token_shape);
                Mln::Sdk::Tensor value_token = body.insert<F::DynamicSlice>(
                    node.value, std::vector<Mln::Sdk::Tensor>{index, zero, zero}, token_shape);
                Mln::Sdk::Tensor decay_token = body.insert<F::DynamicSlice>(
                    node.decay, std::vector<Mln::Sdk::Tensor>{index, zero}, gate_shape);
                Mln::Sdk::Tensor beta_token = body.insert<F::DynamicSlice>(
                    node.beta, std::vector<Mln::Sdk::Tensor>{index, zero}, gate_shape);

                query_token = body.insert<F::Reshape>(std::move(query_token),
                                                      std::vector<int64_t>{heads, channels});
                key_token = body.insert<F::Reshape>(std::move(key_token),
                                                    std::vector<int64_t>{heads, channels});
                value_token = body.insert<F::Reshape>(std::move(value_token),
                                                      std::vector<int64_t>{heads, channels});
                decay_token = body.insert<F::Exp>(
                    body.insert<F::Reshape>(std::move(decay_token), std::vector<int64_t>{heads}));
                decay_token = body.insert<F::Broadcast>(
                    std::move(decay_token), std::vector<int64_t>{heads, channels, channels},
                    std::vector<int64_t>{0});
                memory = body.insert<F::Multiply>(std::move(memory), std::move(decay_token));

                Mln::Sdk::Tensor key_matrix = body.insert<F::Broadcast>(
                    key_token, std::vector<int64_t>{heads, channels, channels},
                    std::vector<int64_t>{0, 1});
                Mln::Sdk::Tensor recalled = body.insert<F::Multiply>(memory, std::move(key_matrix));
                recalled = body.insert<F::ReduceSum>(std::move(recalled), std::vector<int64_t>{1});
                Mln::Sdk::Tensor delta = body.insert<F::Subtract>(value_token, std::move(recalled));
                beta_token =
                    body.insert<F::Reshape>(std::move(beta_token), std::vector<int64_t>{heads});
                beta_token = body.insert<F::Broadcast>(std::move(beta_token),
                                                       std::vector<int64_t>{heads, channels},
                                                       std::vector<int64_t>{0});
                delta = body.insert<F::Multiply>(std::move(delta), std::move(beta_token));

                Mln::Sdk::Tensor update_key = body.insert<F::Broadcast>(
                    key_token, std::vector<int64_t>{heads, channels, channels},
                    std::vector<int64_t>{0, 1});
                Mln::Sdk::Tensor update_value = body.insert<F::Broadcast>(
                    delta, std::vector<int64_t>{heads, channels, channels},
                    std::vector<int64_t>{0, 2});
                memory = body.insert<F::Add>(
                    std::move(memory),
                    body.insert<F::Multiply>(std::move(update_key), std::move(update_value)));

                Mln::Sdk::Tensor read_query = body.insert<F::Broadcast>(
                    query_token, std::vector<int64_t>{heads, channels, channels},
                    std::vector<int64_t>{0, 1});
                Mln::Sdk::Tensor output = body.insert<F::Multiply>(memory, std::move(read_query));
                output = body.insert<F::ReduceSum>(std::move(output), std::vector<int64_t>{1});
                output = body.insert<F::Reshape>(std::move(output), token_shape);
                outputs = body.insert<F::DynamicUpdateSlice>(
                    std::move(outputs), std::move(output),
                    std::vector<Mln::Sdk::Tensor>{index, zero, zero});

                Mln::Sdk::Tensor result = body.insert<F::Reshape>(
                    std::move(memory), std::vector<int64_t>{state_elements});
                return body.insert<F::Concatenate>(
                    std::move(result),
                    body.insert<F::Reshape>(std::move(outputs),
                                            std::vector<int64_t>{output_elements}),
                    0);
            });

        Mln::Sdk::Tensor final_state = slice(operations, carried, 0, state_elements);
        Mln::Sdk::Tensor outputs =
            slice(operations, carried, state_elements, state_elements + output_elements);
        Mln::Sdk::Tensor packed =
            operations.insert<F::Concatenate>(std::move(outputs), std::move(final_state), 0);
        return ctx.resolve(packed, builder, loc);
    }
}
