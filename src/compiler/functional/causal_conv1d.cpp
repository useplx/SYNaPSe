#include "syn/functional/causal_conv1d.hpp"

#include "syn/operations.hpp"

#include "mln/sdk/functional.hpp"

#include <utility>
#include <vector>

namespace Syn::Compiler::Functional {
    mlir::Value CausalConv1D::emit_typed(const Mln::Sdk::Functional::CausalConv1D &node,
                                         Operations &ctx, mlir::OpBuilder &builder,
                                         mlir::Location loc) const {
        namespace F = Mln::Sdk::Functional;
        Mln::Sdk::Operations &operations = ctx.expansion_container();
        operations.set_policy_path(node.policy_path);

        const int64_t tokens = node.input.shape[0];
        const int64_t channels = node.input.shape[1];
        const int64_t kernel_size = node.kernel.shape[1];
        const int64_t state_elements = channels * (kernel_size - 1);
        const int64_t output_elements = tokens * channels;

        Mln::Sdk::Tensor input =
            operations.insert<F::Transpose>(node.input, std::vector<int64_t>{1, 0});
        Mln::Sdk::Tensor window =
            operations.insert<F::Concatenate>(node.state, std::move(input), 1);
        Mln::Sdk::Tensor convolution_input = operations.insert<F::Reshape>(
            window, std::vector<int64_t>{1, channels, 1, tokens + kernel_size - 1});
        Mln::Sdk::Tensor kernel = operations.insert<F::Reshape>(
            node.kernel, std::vector<int64_t>{channels, 1, 1, kernel_size});
        Mln::Sdk::Tensor zero =
            operations.insert<F::Const>(Mln::Sdk::Weight{"", {}, 0.0}, false, Mln::DType::Float32);
        Mln::Sdk::Tensor bias = operations.insert<F::Broadcast>(
            std::move(zero), std::vector<int64_t>{channels}, std::vector<int64_t>{});
        Mln::Sdk::Tensor output = operations.insert<F::Conv2D>(
            std::move(convolution_input), std::move(kernel), std::move(bias),
            std::vector<int64_t>{1, 1}, std::vector<int64_t>{0, 0, 0, 0},
            std::vector<int64_t>{1, 1}, channels);
        output = operations.insert<F::Reshape>(std::move(output),
                                               std::vector<int64_t>{channels, tokens});
        output = operations.insert<F::Transpose>(std::move(output), std::vector<int64_t>{1, 0});
        output = operations.insert<F::Multiply>(output, operations.insert<F::Logistic>(output));

        Mln::Sdk::Tensor next_state = operations.insert<F::Slice>(
            std::move(window), std::vector<int64_t>{0, tokens},
            std::vector<int64_t>{channels, tokens + kernel_size - 1}, std::vector<int64_t>{1, 1});
        Mln::Sdk::Tensor packed =
            operations.insert<F::Reshape>(std::move(output), std::vector<int64_t>{output_elements});
        packed = operations.insert<F::Concatenate>(
            std::move(packed),
            operations.insert<F::Reshape>(std::move(next_state),
                                          std::vector<int64_t>{state_elements}),
            0);
        return ctx.resolve(packed, builder, loc);
    }
}
