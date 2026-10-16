#include "mln/sdk/functional.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace Syn::Compiler::Functional {
    namespace {
        namespace F = Mln::Sdk::Functional;

        Mln::Sdk::Tensor scalar(Mln::Sdk::Operations &operations, double value,
                                std::vector<int64_t> shape) {
            Mln::Sdk::Tensor constant = operations.insert<F::Const>(Mln::Sdk::Weight{"", {}, value},
                                                                    false, Mln::DType::Float32);
            return operations.insert<F::Broadcast>(std::move(constant), std::move(shape),
                                                   std::vector<int64_t>{});
        }

        Mln::Sdk::Tensor slice(Mln::Sdk::Operations &operations, const Mln::Sdk::Tensor &value,
                               std::vector<int64_t> begin, std::vector<int64_t> end) {
            const std::size_t rank = begin.size();
            return operations.insert<F::Slice>(std::move(value), std::move(begin), std::move(end),
                                               std::vector<int64_t>(rank, 1));
        }

        Mln::Sdk::Tensor transpose(Mln::Sdk::Operations &operations, Mln::Sdk::Tensor value,
                                   std::vector<int64_t> permutation) {
            return operations.insert<F::Transpose>(std::move(value), std::move(permutation));
        }

        Mln::Sdk::Tensor append(Mln::Sdk::Operations &operations, Mln::Sdk::Tensor lhs,
                                Mln::Sdk::Tensor rhs, int64_t dimension) {
            return operations.insert<F::Concatenate>(std::move(lhs), std::move(rhs), dimension);
        }

        Mln::Sdk::Tensor cumulative_sum(Mln::Sdk::Operations &operations,
                                        const Mln::Sdk::Tensor &value, int64_t tokens,
                                        int64_t heads) {
            Mln::Sdk::Tensor current = slice(operations, value, {0, 0}, {1, heads});
            Mln::Sdk::Tensor result = current;
            for (int64_t token = 1; token < tokens; ++token) {
                current = operations.insert<F::Add>(
                    std::move(current), slice(operations, value, {token, 0}, {token + 1, heads}));
                result = append(operations, std::move(result), current, 0);
            }
            return result;
        }

        Mln::Sdk::Tensor decay_matrix(Mln::Sdk::Operations &operations,
                                      const Mln::Sdk::Tensor &cumulative, int64_t tokens,
                                      int64_t heads) {
            Mln::Sdk::Tensor result;
            for (int64_t row = 0; row < tokens; ++row) {
                Mln::Sdk::Tensor current =
                    slice(operations, cumulative, {0, row}, {heads, row + 1});
                current =
                    operations.insert<F::Reshape>(std::move(current), std::vector<int64_t>{heads});
                current = operations.insert<F::Broadcast>(std::move(current),
                                                          std::vector<int64_t>{heads, tokens},
                                                          std::vector<int64_t>{0});
                current = operations.insert<F::Exp>(
                    operations.insert<F::Subtract>(std::move(current), cumulative));
                current = slice(operations, current, {0, 0}, {heads, row + 1});
                if (row + 1 < tokens)
                    current = append(operations, std::move(current),
                                     scalar(operations, 0.0, {heads, tokens - row - 1}), 1);
                current = operations.insert<F::Reshape>(std::move(current),
                                                        std::vector<int64_t>{heads, 1, tokens});
                result = row == 0 ? std::move(current)
                                  : append(operations, std::move(result), std::move(current), 1);
            }
            return result;
        }

        Mln::Sdk::Tensor strict_lower(Mln::Sdk::Operations &operations,
                                      const Mln::Sdk::Tensor &value, int64_t tokens,
                                      int64_t heads) {
            Mln::Sdk::Tensor result;
            for (int64_t row = 0; row < tokens; ++row) {
                Mln::Sdk::Tensor current = scalar(operations, 0.0, {heads, 1, tokens});
                if (row > 0) {
                    Mln::Sdk::Tensor prefix =
                        slice(operations, value, {0, row, 0}, {heads, row + 1, row});
                    prefix = operations.insert<F::Reshape>(std::move(prefix),
                                                           std::vector<int64_t>{heads, 1, row});
                    current = append(operations, std::move(prefix),
                                     scalar(operations, 0.0, {heads, 1, tokens - row}), 2);
                }
                result = row == 0 ? std::move(current)
                                  : append(operations, std::move(result), std::move(current), 1);
            }
            return result;
        }

        Mln::Sdk::Tensor inverse_rows(Mln::Sdk::Operations &operations,
                                      const Mln::Sdk::Tensor &lower, int64_t tokens,
                                      int64_t heads) {
            Mln::Sdk::Tensor result = scalar(operations, 0.0, {heads, 1, tokens});
            for (int64_t row = 1; row < tokens; ++row) {
                Mln::Sdk::Tensor raw = slice(operations, lower, {0, row, 0}, {heads, row + 1, row});
                raw = operations.insert<F::Reshape>(std::move(raw),
                                                    std::vector<int64_t>{heads, 1, row});
                Mln::Sdk::Tensor previous = slice(operations, result, {0, 0, 0}, {heads, row, row});
                Mln::Sdk::Tensor prefix = operations.insert<F::Add>(
                    raw, operations.insert<F::MatMul>(raw, std::move(previous)));
                Mln::Sdk::Tensor current =
                    append(operations, std::move(prefix),
                           scalar(operations, 0.0, {heads, 1, tokens - row}), 2);
                result = append(operations, std::move(result), std::move(current), 1);
            }
            return result;
        }

        Mln::Sdk::Tensor identity(Mln::Sdk::Operations &operations, int64_t tokens, int64_t heads) {
            Mln::Sdk::Tensor result;
            for (int64_t row = 0; row < tokens; ++row) {
                Mln::Sdk::Tensor current;
                if (row > 0)
                    current = scalar(operations, 0.0, {heads, 1, row});
                Mln::Sdk::Tensor diagonal = scalar(operations, 1.0, {heads, 1, 1});
                current = row > 0 ? append(operations, std::move(current), std::move(diagonal), 2)
                                  : std::move(diagonal);
                if (row + 1 < tokens)
                    current = append(operations, std::move(current),
                                     scalar(operations, 0.0, {heads, 1, tokens - row - 1}), 2);
                result = row == 0 ? std::move(current)
                                  : append(operations, std::move(result), std::move(current), 1);
            }
            return result;
        }
    }

    namespace {
        Mln::Sdk::Tensor expand_one_chunk(Mln::Sdk::Operations &operations,
                                          const Mln::Sdk::Functional::GatedDeltaRule &node,
                                          Mln::Sdk::Tensor query, Mln::Sdk::Tensor key) {
            const int64_t tokens = node.query.shape[0];
            const int64_t heads = node.query.shape[1];
            const int64_t channels = node.query.shape[2];

            query = transpose(operations, std::move(query), {1, 0, 2});
            key = transpose(operations, std::move(key), {1, 0, 2});
            Mln::Sdk::Tensor value = transpose(operations, node.value, {1, 0, 2});
            Mln::Sdk::Tensor beta = transpose(operations, node.beta, {1, 0});
            Mln::Sdk::Tensor cumulative = cumulative_sum(operations, node.decay, tokens, heads);
            cumulative = transpose(operations, std::move(cumulative), {1, 0});

            Mln::Sdk::Tensor beta_values = operations.insert<F::Broadcast>(
                beta, std::vector<int64_t>{heads, tokens, channels}, std::vector<int64_t>{0, 1});
            Mln::Sdk::Tensor value_beta = operations.insert<F::Multiply>(value, beta_values);
            Mln::Sdk::Tensor key_beta = operations.insert<F::Multiply>(key, beta_values);
            Mln::Sdk::Tensor key_transposed = transpose(operations, key, {0, 2, 1});
            Mln::Sdk::Tensor decay = decay_matrix(operations, cumulative, tokens, heads);
            Mln::Sdk::Tensor affinity = operations.insert<F::Multiply>(
                operations.insert<F::MatMul>(key_beta, key_transposed), decay);
            affinity = operations.insert<F::Subtract>(
                scalar(operations, 0.0, {heads, tokens, tokens}), std::move(affinity));
            affinity = strict_lower(operations, affinity, tokens, heads);
            Mln::Sdk::Tensor transform = inverse_rows(operations, affinity, tokens, heads);
            transform = operations.insert<F::Add>(std::move(transform),
                                                  identity(operations, tokens, heads));

            Mln::Sdk::Tensor adjusted_value = operations.insert<F::MatMul>(transform, value_beta);
            Mln::Sdk::Tensor exponential = operations.insert<F::Exp>(cumulative);
            exponential = operations.insert<F::Broadcast>(
                exponential, std::vector<int64_t>{heads, tokens, channels},
                std::vector<int64_t>{0, 1});
            Mln::Sdk::Tensor cumulative_key = operations.insert<F::MatMul>(
                transform, operations.insert<F::Multiply>(key_beta, exponential));

            Mln::Sdk::Tensor corrected_value = operations.insert<F::Subtract>(
                std::move(adjusted_value),
                operations.insert<F::MatMul>(std::move(cumulative_key), node.initial_state));
            Mln::Sdk::Tensor attention = operations.insert<F::Multiply>(
                operations.insert<F::MatMul>(query, key_transposed), decay);
            Mln::Sdk::Tensor output = operations.insert<F::Add>(
                operations.insert<F::MatMul>(operations.insert<F::Multiply>(query, exponential),
                                             node.initial_state),
                operations.insert<F::MatMul>(std::move(attention), corrected_value));

            Mln::Sdk::Tensor last = slice(operations, cumulative, {0, tokens - 1}, {heads, tokens});
            last = operations.insert<F::Reshape>(std::move(last), std::vector<int64_t>{heads});
            Mln::Sdk::Tensor state_decay = operations.insert<F::Exp>(last);
            state_decay = operations.insert<F::Broadcast>(
                std::move(state_decay), std::vector<int64_t>{heads, channels, channels},
                std::vector<int64_t>{0});
            Mln::Sdk::Tensor final_state =
                operations.insert<F::Multiply>(node.initial_state, std::move(state_decay));
            last = operations.insert<F::Broadcast>(
                std::move(last), std::vector<int64_t>{heads, tokens}, std::vector<int64_t>{0});
            Mln::Sdk::Tensor relative = operations.insert<F::Exp>(
                operations.insert<F::Subtract>(std::move(last), cumulative));
            relative = operations.insert<F::Broadcast>(
                std::move(relative), std::vector<int64_t>{heads, tokens, channels},
                std::vector<int64_t>{0, 1});
            Mln::Sdk::Tensor final_key = operations.insert<F::Multiply>(key, std::move(relative));
            final_key = transpose(operations, std::move(final_key), {0, 2, 1});
            final_state = operations.insert<F::Add>(
                std::move(final_state),
                operations.insert<F::MatMul>(std::move(final_key), corrected_value));

            output = transpose(operations, std::move(output), {1, 0, 2});
            output = operations.insert<F::Reshape>(std::move(output),
                                                   std::vector<int64_t>{tokens * heads * channels});
            final_state = operations.insert<F::Reshape>(
                std::move(final_state), std::vector<int64_t>{heads * channels * channels});
            return append(operations, std::move(output), std::move(final_state), 0);
        }
    }

    Mln::Sdk::Tensor expand_gated_delta_chunk(Mln::Sdk::Operations &operations,
                                              const Mln::Sdk::Functional::GatedDeltaRule &node,
                                              Mln::Sdk::Tensor query, Mln::Sdk::Tensor key) {
        constexpr int64_t ChunkSize = 64;
        const int64_t tokens = node.query.shape[0];
        const int64_t heads = node.query.shape[1];
        const int64_t channels = node.query.shape[2];
        const int64_t state_elements = heads * channels * channels;

        Mln::Sdk::Tensor state = node.initial_state;
        Mln::Sdk::Tensor outputs;
        for (int64_t begin = 0; begin < tokens; begin += ChunkSize) {
            const int64_t end = std::min(begin + ChunkSize, tokens);
            const int64_t count = end - begin;
            Mln::Sdk::Tensor chunk_query =
                slice(operations, query, {begin, 0, 0}, {end, heads, channels});
            Mln::Sdk::Tensor chunk_key =
                slice(operations, key, {begin, 0, 0}, {end, heads, channels});
            Mln::Sdk::Functional::GatedDeltaRule chunk{
                chunk_query,
                chunk_key,
                slice(operations, node.value, {begin, 0, 0}, {end, heads, channels}),
                slice(operations, node.decay, {begin, 0}, {end, heads}),
                slice(operations, node.beta, {begin, 0}, {end, heads}),
                state,
                node.normalization_epsilon,
            };
            Mln::Sdk::Tensor packed =
                expand_one_chunk(operations, chunk, std::move(chunk_query), std::move(chunk_key));
            const int64_t output_elements = count * heads * channels;
            Mln::Sdk::Tensor chunk_output = slice(operations, packed, {0}, {output_elements});
            outputs = begin == 0
                          ? std::move(chunk_output)
                          : append(operations, std::move(outputs), std::move(chunk_output), 0);
            state =
                slice(operations, packed, {output_elements}, {output_elements + state_elements});
            state = operations.insert<F::Reshape>(std::move(state),
                                                  std::vector<int64_t>{heads, channels, channels});
        }

        state =
            operations.insert<F::Reshape>(std::move(state), std::vector<int64_t>{state_elements});
        return append(operations, std::move(outputs), std::move(state), 0);
    }
}
