#include "mln/error.hpp"
#include "mln/sdk/functional.hpp"
#include "mln/sdk/module.hpp"
#include "mln/sdk/orchestration.hpp"
#include "syn/compiler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
    using namespace Mln::Sdk;
    using Mln::DType;

    class TestCodec final : public Mln::Sdk::Codec {
        public:
            TestCodec() { name = "F32"; }
            Tensor decode(Operations &operations, const Weight &weight, DType dtype) override {
                return operations.insert<Functional::Const>(weight, false, dtype);
            }
    };

    class PortableAttention final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                return context.operation.kind == Functional::Kind::Attention;
            }

            std::optional<bool> expand(const Context &) const override { return true; }
    };

    class TiledReduction final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                return context.operation.kind == Functional::Kind::ReduceSum;
            }

            int64_t tile(const Context &, int64_t, int64_t extent) const override {
                constexpr int64_t Tile = 2;
                return extent < 0 ? Tile : std::min(extent, Tile);
            }
    };

    class ChunkedGatedDeltaRule final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                return context.operation.kind == Functional::Kind::GatedDeltaRule;
            }

            std::optional<bool> expand(const Context &) const override { return true; }
    };

    Tensor literal(Operations &operations, double value, DType dtype = DType::Float32) {
        Weight w{"", {}, value};
        return operations.insert<Functional::Const>(w, false, dtype);
    }

    Tensor literal_array(Operations &operations, const std::vector<double> &values,
                         DType dtype = DType::Float32) {
        Tensor result = operations.insert<Functional::Broadcast>(
            literal(operations, values[0], dtype), std::vector<int64_t>{1}, std::vector<int64_t>{});
        for (std::size_t i = 1; i < values.size(); i++) {
            Tensor next = operations.insert<Functional::Broadcast>(
                literal(operations, values[i], dtype), std::vector<int64_t>{1},
                std::vector<int64_t>{});
            result = operations.insert<Functional::Concatenate>(result, next, 0);
        }
        return result;
    }

    class Trig final : public Module {
        public:
            Trig() { name = "trig"; }

            Tensor forward(Tensor x) {
                Tensor value = operations.insert<Functional::Sqrt>(x);
                value = operations.insert<Functional::Add>(value,
                                                           operations.insert<Functional::Rsqrt>(x));
                value = operations.insert<Functional::Add>(value,
                                                           operations.insert<Functional::Sin>(x));
                value = operations.insert<Functional::Add>(value,
                                                           operations.insert<Functional::Cos>(x));
                value = operations.insert<Functional::Add>(value,
                                                           operations.insert<Functional::Tanh>(x));
                return value;
            }
    };

    class Reduction final : public Module {
        public:
            Reduction() { name = "maxreduce"; }

            Tensor forward(Tensor x) {
                return operations.insert<Functional::ReduceMax>(x, std::vector<int64_t>{1});
            }
    };

    class GenericReduction final : public Module {
        public:
            GenericReduction() { name = "generic_reduction"; }

            Tensor forward(Tensor x) {
                return operations.insert<Functional::Reduce>(
                    std::move(x), literal(operations, 1.0), std::vector<int64_t>{1},
                    std::vector<int64_t>{2}, Functional::Reduction::Multiply);
            }
    };

    class Ordering final : public Module {
        public:
            Ordering() { name = "ordering"; }

            Tensor forward(Tensor x) {
                std::vector<Tensor> inputs;
                inputs.push_back(std::move(x));
                auto results = operations.insert_many<Functional::Sort>(
                    std::move(inputs), 1, true, Functional::SortDirection::Ascending);
                return std::move(results.front());
            }
    };

    class RandomValues final : public Module {
        public:
            RandomValues() { name = "random_values"; }

            Tensor forward(Tensor shape) {
                return operations.insert<Functional::Rng>(
                    literal(operations, 0.0), literal(operations, 1.0), std::move(shape),
                    std::vector<int64_t>{2, 4}, Functional::Distribution::Uniform);
            }
    };

    class RandomBits final : public Module {
        public:
            RandomBits() { name = "random_bits"; }

            Tensor forward(Tensor state) {
                auto results = operations.insert_many<Functional::RngBitGenerator>(
                    std::move(state), std::vector<int64_t>{2, 2}, DType::UInt32,
                    Functional::RngAlgorithm::Philox);
                return std::move(results[1]);
            }
    };

    class IndexedUpdate final : public Module {
        public:
            IndexedUpdate() { name = "indexed_update"; }

            Tensor forward(Tensor input, Tensor indices, Tensor updates) {
                return operations.insert<Functional::Scatter>(
                    std::move(input), std::move(indices), std::move(updates),
                    std::vector<int64_t>{}, std::vector<int64_t>{0}, std::vector<int64_t>{},
                    std::vector<int64_t>{}, std::vector<int64_t>{0}, 1, true, true,
                    Functional::ScatterUpdate::Add);
            }
    };

    class Elementwise final : public Module {
        public:
            Elementwise() { name = "elementwise"; }

            Tensor forward(Tensor lhs, Tensor rhs) {
                Tensor value =
                    operations.insert<Functional::Abs>(operations.insert<Functional::Negate>(lhs));
                value = operations.insert<Functional::Floor>(std::move(value));
                value = operations.insert<Functional::Ceil>(std::move(value));
                value = operations.insert<Functional::Maximum>(std::move(value), rhs);
                value = operations.insert<Functional::Minimum>(std::move(value), rhs);
                value = operations.insert<Functional::Power>(std::move(value), rhs);
                value = operations.insert<Functional::Remainder>(std::move(value), rhs);
                value = operations.insert<Functional::Cbrt>(std::move(value));
                value = operations.insert<Functional::Expm1>(std::move(value));
                value = operations.insert<Functional::Log1p>(std::move(value));
                value = operations.insert<Functional::Round>(std::move(value));
                value = operations.insert<Functional::RoundNearestEven>(std::move(value));
                value = operations.insert<Functional::Sign>(std::move(value));
                value = operations.insert<Functional::Tan>(std::move(value));
                value = operations.insert<Functional::Atan2>(std::move(value), rhs);
                Tensor finite = operations.insert<Functional::IsFinite>(value);
                Tensor ordered = operations.insert<Functional::Greater>(value, rhs);
                ordered = operations.insert<Functional::BitwiseAnd>(std::move(ordered), finite);
                Tensor distinct = operations.insert<Functional::NotEqual>(value, rhs);
                ordered = operations.insert<Functional::BitwiseOr>(std::move(ordered), distinct);
                Tensor greater_equal = operations.insert<Functional::GreaterEqual>(value, rhs);
                Tensor less = operations.insert<Functional::Less>(value, rhs);
                Tensor less_equal = operations.insert<Functional::LessEqual>(value, rhs);
                ordered =
                    operations.insert<Functional::BitwiseXor>(std::move(ordered), greater_equal);
                ordered = operations.insert<Functional::BitwiseXor>(std::move(ordered), less);
                ordered = operations.insert<Functional::BitwiseXor>(std::move(ordered), less_equal);
                value = operations.insert<Functional::Select>(std::move(ordered), value, rhs);
                value = operations.insert<Functional::Clamp>(rhs, std::move(value), lhs);
                value = operations.insert<Functional::Reverse>(std::move(value),
                                                               std::vector<int64_t>{0});
                value = operations.insert<Functional::ReducePrecision>(std::move(value), 8, 7);
                value = operations.insert<Functional::OptimizationBarrier>(std::move(value));
                Tensor zero = literal(operations, 0.0);
                value = operations.insert<Functional::Pad>(
                    std::move(value), std::move(zero), std::vector<int64_t>{0},
                    std::vector<int64_t>{0}, std::vector<int64_t>{0});
                Tensor extent = operations.insert<Functional::GetDimensionSize>(value, 0);
                extent = operations.insert<Functional::Convert>(std::move(extent), DType::Float32);
                extent = operations.insert<Functional::Broadcast>(std::move(extent), value.shape,
                                                                  std::vector<int64_t>{});
                value = operations.insert<Functional::Add>(std::move(value), std::move(extent));
                value = operations.insert<Functional::Convert>(std::move(value), DType::BFloat16);
                value = operations.insert<Functional::Convert>(std::move(value), DType::Float32);

                Tensor integers = operations.insert<Functional::Convert>(value, DType::Int32);
                Tensor mask = operations.insert<Functional::Broadcast>(
                    literal(operations, 3.0, DType::Int32), value.shape, std::vector<int64_t>{});
                integers = operations.insert<Functional::BitwiseOr>(std::move(integers), mask);
                integers = operations.insert<Functional::BitwiseXor>(std::move(integers), mask);
                integers = operations.insert<Functional::CountLeadingZeros>(std::move(integers));
                integers = operations.insert<Functional::PopulationCount>(std::move(integers));
                integers = operations.insert<Functional::BitwiseNot>(std::move(integers));
                integers = operations.insert<Functional::ShiftRightArithmetic>(std::move(integers),
                                                                               std::move(mask));
                return operations.insert<Functional::Convert>(std::move(integers), DType::Float32);
            }
    };

    class Shift final : public Module {
        public:
            Shift() { name = "bitshift"; }

            Tensor forward(Tensor x) {
                Tensor xi = operations.insert<Functional::Convert>(x, DType::Int32);
                Tensor shift = operations.insert<Functional::Broadcast>(
                    literal(operations, 2.0, DType::Int32), x.shape, std::vector<int64_t>{});
                Tensor shifted = operations.insert<Functional::ShiftLeftLogical>(xi, shift);
                return operations.insert<Functional::Convert>(shifted, DType::Float32);
            }
    };

    class DynamicShape final : public Module {
        public:
            DynamicShape() { name = "dynamic_shape"; }

            Tensor forward(Tensor x) {
                Tensor reshape_shape = literal_array(operations, {1.0, 5.0}, DType::Int32);
                Tensor value = operations.insert<Functional::DynamicReshape>(
                    std::move(x), std::move(reshape_shape), std::vector<int64_t>{1, 5});

                Tensor broadcast_shape = literal_array(operations, {2.0, 5.0}, DType::Int32);
                value = operations.insert<Functional::DynamicBroadcast>(
                    std::move(value), std::move(broadcast_shape), std::vector<int64_t>{2, 5},
                    std::vector<int64_t>{0, 1});

                Tensor starts = literal_array(operations, {0.0, 0.0}, DType::Int32);
                Tensor limits = literal_array(operations, {2.0, 5.0}, DType::Int32);
                Tensor strides = literal_array(operations, {1.0, 1.0}, DType::Int32);
                value = operations.insert<Functional::RealDynamicSlice>(
                    std::move(value), std::move(starts), std::move(limits), std::move(strides),
                    std::vector<int64_t>{2, 5});

                return value;
            }
    };

    class LinearAlgebra final : public Module {
        public:
            LinearAlgebra() { name = "linear_algebra"; }

            Tensor forward(Tensor a, Tensor b) {
                Tensor factor = operations.insert<Functional::Cholesky>(a, true);
                Tensor solved = operations.insert<Functional::TriangularSolve>(
                    std::move(a), std::move(b), true, true, false,
                    Functional::SolveTranspose::None);
                return operations.insert<Functional::Add>(std::move(factor), std::move(solved));
            }
    };

    class BatchNorm final : public Module {
        public:
            BatchNorm() { name = "batch_norm"; }

            Tensor forward(Tensor x, Tensor scale, Tensor offset, Tensor mean, Tensor variance) {
                return operations.insert<Functional::BatchNormInference>(
                    std::move(x), std::move(scale), std::move(offset), std::move(mean),
                    std::move(variance), 1.0e-5F, 1);
            }
    };

    class BatchNormTraining final : public Module {
        public:
            BatchNormTraining() { name = "batch_norm_training"; }

            Tensor forward(Tensor x, Tensor scale, Tensor offset) {
                auto results = operations.insert_many<Functional::BatchNormTraining>(
                    std::move(x), std::move(scale), std::move(offset), 1.0e-5F, 1);
                return std::move(results[1]);
            }
    };

    class BatchNormGrad final : public Module {
        public:
            BatchNormGrad() { name = "batch_norm_grad"; }

            Tensor forward(Tensor x, Tensor scale, Tensor mean, Tensor variance, Tensor gradient) {
                auto results = operations.insert_many<Functional::BatchNormGrad>(
                    std::move(x), std::move(scale), std::move(mean), std::move(variance),
                    std::move(gradient), 1.0e-5F, 1);
                return std::move(results[2]);
            }
    };

    class Contraction final : public Module {
        public:
            Contraction() { name = "contraction"; }

            Tensor forward(Tensor lhs, Tensor rhs) {
                return operations.insert<Functional::Einsum>(
                    std::move(lhs), std::move(rhs), "ab,bc->ac", std::vector<int64_t>{2, 4});
            }
    };

    class DotContraction final : public Module {
        public:
            DotContraction() { name = "dot_contraction"; }

            Tensor forward(Tensor lhs, Tensor rhs) {
                Tensor dot =
                    operations.insert<Functional::Dot>(lhs, rhs, std::vector<int64_t>{2, 4});
                Tensor general = operations.insert<Functional::DotGeneral>(
                    std::move(lhs), std::move(rhs), std::vector<int64_t>{}, std::vector<int64_t>{},
                    std::vector<int64_t>{1}, std::vector<int64_t>{0}, std::vector<int64_t>{2, 4});
                return operations.insert<Functional::Add>(std::move(dot), std::move(general));
            }
    };

    class DynamicSequence final : public Module {
        public:
            DynamicSequence() { name = "dynamic_sequence"; }

            Tensor forward(Tensor shape) {
                return operations.insert<Functional::DynamicIota>(
                    std::move(shape), std::vector<int64_t>{2, 4}, 1, DType::Float32);
            }
    };

    class Lookup final : public Module {
        public:
            Lookup() { name = "lookup"; }

            Tensor forward(Tensor x) {
                Tensor table = literal_array(operations, {100.0, 200.0, 300.0, 400.0});
                Tensor indices = operations.insert<Functional::Convert>(x, DType::Int32);
                return operations.insert<Functional::Gather>(table, indices, 0);
            }
    };

    class Read final : public Module {
        public:
            Read() { name = "cacheread"; }

            Tensor forward(Tensor position) {
                Tensor table =
                    literal_array(operations, {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0});
                Tensor pos = operations.insert<Functional::Convert>(position, DType::Int32);
                return operations.insert<Functional::DynamicSlice>(table, std::vector<Tensor>{pos},
                                                                   std::vector<int64_t>{3});
            }
    };

    class Write final : public Module {
        public:
            Write() { name = "cachewrite"; }

            Tensor forward(Tensor position) {
                Tensor base = literal_array(operations, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
                Tensor update = literal_array(operations, {9.0, 8.0, 7.0});
                Tensor pos = operations.insert<Functional::Convert>(position, DType::Int32);
                return operations.insert<Functional::DynamicUpdateSlice>(base, update,
                                                                         std::vector<Tensor>{pos});
            }
    };

    class Multiple final : public Module {
        public:
            Multiple() { name = "multi_io"; }

            std::tuple<Tensor, Tensor> forward(Tensor lhs, Tensor rhs) {
                return {
                    operations.insert<Functional::Add>(lhs, rhs),
                    operations.insert<Functional::Subtract>(lhs, rhs),
                };
            }
    };

    class Recurrent final : public Module {
        public:
            Recurrent() { name = "recurrent"; }

            Tensor forward(Tensor input) {
                return operations.insert<Functional::Loop>(
                    std::move(input), 3, [](Operations &body, const Tensor &, const Tensor &value) {
                        Tensor one = body.insert<Functional::Broadcast>(
                            literal(body, 1.0), value.shape, std::vector<int64_t>{});
                        return body.insert<Functional::Add>(value, std::move(one));
                    });
            }
    };

    class Attention final : public Module {
        public:
            Attention() { name = "attention"; }

            Tensor forward(Tensor query, Tensor key, Tensor value, Tensor mask) {
                const double scale = 1.0 / std::sqrt(static_cast<double>(query.shape.back()));
                return operations.insert<Functional::Attention>(
                    std::move(query), std::move(key), std::move(value), scale, std::move(mask));
            }
    };

    class GatedDeltaRule final : public Module {
        public:
            GatedDeltaRule() { name = "gated_delta_rule"; }

            Tensor forward(Tensor query, Tensor key, Tensor value, Tensor decay, Tensor beta,
                           Tensor state) {
                return operations.insert<Functional::GatedDeltaRule>(
                    std::move(query), std::move(key), std::move(value), std::move(decay),
                    std::move(beta), std::move(state));
            }
    };

    class CausalConv1D final : public Module {
        public:
            CausalConv1D() { name = "causal_conv1d"; }

            Tensor forward(Tensor input, Tensor kernel, Tensor state) {
                return operations.insert<Functional::CausalConv1D>(
                    std::move(input), std::move(kernel), std::move(state));
            }
    };

    class DenseMatMul final : public Module {
        public:
            DenseMatMul() { name = "dense_matmul"; }

            Tensor forward(Tensor lhs, Tensor rhs) {
                return operations.insert<Functional::MatMul>(std::move(lhs), std::move(rhs));
            }
    };

    class Q8MatMul final : public Module {
        public:
            Q8MatMul() { name = "q8_matmul"; }

            Tensor forward(Tensor lhs, Tensor codes, Tensor scale, Tensor zero) {
                Tensor shifts = literal_array(operations, {0.0, 8.0, 16.0, 24.0}, DType::Int32);
                Tensor decoded = operations.insert<Functional::Map>(
                    std::vector<Functional::Map::Operand>{
                        {std::move(codes), DType::Int32, {0, 1, 2}},
                        {std::move(shifts), DType::Int32, {3}},
                        {std::move(scale), DType::Float16, {0, 1}},
                        {std::move(zero), DType::Float16, {0, 1}},
                    },
                    std::vector<int64_t>{6144, 64, 8, 4}, DType::Float16,
                    [](Operations &body, std::span<const Tensor> args) {
                        Tensor values =
                            body.insert<Functional::ShiftRightLogical>(args[0], args[1]);
                        values = body.insert<Functional::BitwiseAnd>(
                            std::move(values), literal(body, 255.0, DType::Int32));
                        values = body.insert<Functional::Convert>(std::move(values), DType::UInt32);
                        values =
                            body.insert<Functional::Convert>(std::move(values), DType::Float16);
                        Tensor shifted =
                            body.insert<Functional::Subtract>(std::move(values), args[3]);
                        return body.insert<Functional::Multiply>(std::move(shifted), args[2]);
                    });
                decoded = operations.insert<Functional::Reshape>(std::move(decoded),
                                                                 std::vector<int64_t>{6144, 2048});
                decoded = operations.insert<Functional::Transpose>(std::move(decoded),
                                                                   std::vector<int64_t>{1, 0});
                return operations.insert<Functional::MatMul>(std::move(lhs), std::move(decoded));
            }
    };

    class Q8Decode final : public Module {
        public:
            Q8Decode() { name = "q8_decode"; }

            Tensor forward(Tensor codes, Tensor scale, Tensor zero) {
                Tensor shifts = literal_array(operations, {0.0, 8.0, 16.0, 24.0}, DType::Int32);
                Tensor decoded = operations.insert<Functional::Map>(
                    std::vector<Functional::Map::Operand>{
                        {std::move(codes), DType::Int32, {0, 1, 2}},
                        {std::move(shifts), DType::Int32, {3}},
                        {std::move(scale), DType::Float16, {0, 1}},
                        {std::move(zero), DType::Float16, {0, 1}},
                    },
                    std::vector<int64_t>{1, 64, 8, 4}, DType::Float16,
                    [](Operations &body, std::span<const Tensor> args) {
                        Tensor values =
                            body.insert<Functional::ShiftRightLogical>(args[0], args[1]);
                        values = body.insert<Functional::BitwiseAnd>(
                            std::move(values), literal(body, 255.0, DType::Int32));
                        values = body.insert<Functional::Convert>(std::move(values), DType::UInt32);
                        values =
                            body.insert<Functional::Convert>(std::move(values), DType::Float16);
                        Tensor shifted =
                            body.insert<Functional::Subtract>(std::move(values), args[3]);
                        return body.insert<Functional::Multiply>(std::move(shifted), args[2]);
                    });
                return operations.insert<Functional::Reshape>(std::move(decoded),
                                                              std::vector<int64_t>{2048});
            }
    };

    class Identity final : public Module {
        public:
            Tensor forward(Tensor input) { return input; }
    };

    class Hooked final : public Module {
        public:
            Hooked() { modules.insert<Identity>("child"); }

            Tensor forward(Tensor input) { return modules.forward<Identity>("child", input); }
    };

    class TestOrchestration final : public Mln::Sdk::Orchestration {
        public:
            TestOrchestration() {
                name = "ops";
                modules.insert<Trig>("trig");
                modules.insert<Reduction>("maxreduce");
                modules.insert<GenericReduction>("generic_reduction");
                modules.insert<Ordering>("ordering");
                modules.insert<RandomValues>("random_values");
                modules.insert<RandomBits>("random_bits");
                modules.insert<IndexedUpdate>("indexed_update");
                modules.insert<Elementwise>("elementwise");
                modules.insert<Shift>("bitshift");
                modules.insert<DynamicShape>("dynamic_shape");
                modules.insert<BatchNorm>("batch_norm");
                modules.insert<BatchNormTraining>("batch_norm_training");
                modules.insert<Contraction>("contraction");
                modules.insert<DotContraction>("dot_contraction");
                modules.insert<DynamicSequence>("dynamic_sequence");
                modules.insert<Lookup>("lookup");
                modules.insert<Read>("cacheread");
                modules.insert<Write>("cachewrite");
                modules.insert<Multiple>("multi_io");
                modules.insert<Recurrent>("recurrent");
                modules.insert<Attention>("attention");
                modules.insert<GatedDeltaRule>("gated_delta_rule");
                modules.insert<CausalConv1D>("causal_conv1d");
                modules.insert<DenseMatMul>("dense_matmul");
                modules.insert<Q8MatMul>("q8_matmul");
                modules.insert<Q8Decode>("q8_decode");
                modules.insert<Hooked>("hooked");
            }

            Mln::Buffer forward(Mln::Runtime::Session &, Mln::Buffer input) override {
                return input;
            }
    };
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 7)
        return 2;
    using Mln::Sdk::Module;
    using Mln::Sdk::Tensor;
    namespace Functional = Mln::Sdk::Functional;
    namespace Compiler = Syn::Compiler;

    Compiler::Manifest manifest{
        .policies =
            {
                {"model",
                 Compiler::Police{
                     .schemes = {Mln::Sdk::Codec::Scheme{
                         .scope = Mln::Sdk::Codec::Scope::Constant,
                         .lifetime = Mln::Sdk::Codec::Lifetime::Immutable,
                         .update = Mln::Sdk::Codec::Update::None,
                         .format = TestCodec{},
                     }},
                     .dtype = Mln::DType::Float32,
                     .backend =
                         Mln::Sdk::Backend{
                             .device = "local",
                             .flags = {{"iree-hal-local-target-device-backends", "llvm-cpu"}},
                         },
                 }},
            },
        .hooks =
            {
                {
                    .path = "hooked.*",
                    .forward =
                        [](Module &module, std::span<const Tensor> inputs, Tensor output) {
                            if (inputs.size() != 1)
                                throw std::runtime_error("Hook did not receive the original input");
                            return module.operations.insert<Functional::Add>(std::move(output),
                                                                             inputs.front());
                        },
                },
            },
    };

    {
        Compiler::Manifest invalid = manifest;
        invalid.hooks.front().path = "hooked.ch*ld";
        bool rejected = false;
        try {
            Compiler::validate_manifest(invalid);
        } catch (const Mln::Diagnostic &) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("Manifest accepted a partial Hook wildcard");
    }

    {
        Compiler::Manifest unmatched = manifest;
        unmatched.hooks.front().path = "missing.*";
        TestOrchestration isolated_orchestration;
        Compiler::Compiler isolated(unmatched, isolated_orchestration);
        Mln::Sdk::Tensor input({5});
        isolated.compile(*isolated_orchestration.modules.at<Trig>("trig"), input);
        bool rejected = false;
        try {
            isolated.finalize();
        } catch (const Mln::Diagnostic &) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("Compiler accepted a Hook without a matching path");
    }

    {
        Compiler::Manifest vulkan = manifest;
        PortableAttention portable_attention;
        TiledReduction tiled_reduction;
        vulkan.operation_compatibilities.push_back(&portable_attention);
        vulkan.operation_compatibilities.push_back(&tiled_reduction);
        vulkan.hooks.clear();
        vulkan.policies.front().second.dtype = Mln::DType::Float16;
        if (argc == 7) {
            TestOrchestration local_orchestration;
            Compiler::Manifest local = vulkan;
            local.operation_compatibilities.clear();
            local.policies.front().second.backend = Mln::Sdk::Backend{
                .device = "local",
                .flags = {{"iree-hal-local-target-device-backends", "llvm-cpu"}},
            };
            Compiler::Compiler local_compiler(local, local_orchestration);
            Mln::Sdk::Tensor query({8, 1, 256});
            Mln::Sdk::Tensor key({8, 32, 256});
            Mln::Sdk::Tensor value({8, 32, 256});
            Mln::Sdk::Tensor mask({1, 32});
            local_compiler.compile(*local_orchestration.modules.at<Attention>("attention"), query,
                                   key, value, mask);
            Mln::Sdk::Tensor position(std::vector<int64_t>{});
            local_compiler.compile(*local_orchestration.modules.at<Read>("cacheread"), position);
            local_compiler.compile(*local_orchestration.modules.at<Write>("cachewrite"), position);
            Mln::Sdk::Tensor convolution_input({8, 6144}, Mln::DType::Float32);
            Mln::Sdk::Tensor convolution_kernel({6144, 4}, Mln::DType::Float32);
            Mln::Sdk::Tensor convolution_state({6144, 3}, Mln::DType::Float32);
            local_compiler.compile(*local_orchestration.modules.at<CausalConv1D>("causal_conv1d"),
                                   convolution_input, convolution_kernel, convolution_state);
            Mln::Sdk::Tensor dense_lhs({2048}, Mln::DType::Float16);
            Mln::Sdk::Tensor dense_rhs({2048, 6144}, Mln::DType::Float16);
            local_compiler.compile(*local_orchestration.modules.at<DenseMatMul>("dense_matmul"),
                                   dense_lhs, dense_rhs);
            Mln::Sdk::Tensor q8_codes({6144, 64, 8}, Mln::DType::Int32);
            Mln::Sdk::Tensor q8_scale({6144, 64}, Mln::DType::Float16);
            Mln::Sdk::Tensor q8_zero({6144, 64}, Mln::DType::Float16);
            local_compiler.compile(*local_orchestration.modules.at<Q8MatMul>("q8_matmul"),
                                   dense_lhs, q8_codes, q8_scale, q8_zero);
            Mln::Sdk::Tensor decode_codes({1, 64, 8}, Mln::DType::Int32);
            Mln::Sdk::Tensor decode_scale({1, 64}, Mln::DType::Float16);
            Mln::Sdk::Tensor decode_zero({1, 64}, Mln::DType::Float16);
            local_compiler.compile(*local_orchestration.modules.at<Q8Decode>("q8_decode"),
                                   decode_codes, decode_scale, decode_zero);
            local_compiler.write_vmfb(argv[3], argv[2]);
        }
        vulkan.policies.front().second.backend = Mln::Sdk::Backend{
            .device = "vulkan",
            .flags = {{"iree-vulkan-target", argc == 7 ? "ampere" : "adreno-643"}},
        };
        TestOrchestration isolated_orchestration;
        Compiler::Compiler isolated(vulkan, isolated_orchestration);
        Mln::Sdk::Tensor query(argc == 7 ? std::vector<int64_t>{8, 1, 256}
                                         : std::vector<int64_t>{2, 4, 3});
        Mln::Sdk::Tensor key(argc == 7 ? std::vector<int64_t>{8, 32, 256}
                                       : std::vector<int64_t>{2, -1, 3});
        Mln::Sdk::Tensor value(argc == 7 ? std::vector<int64_t>{8, 32, 256}
                                         : std::vector<int64_t>{2, -1, 5});
        Mln::Sdk::Tensor mask(argc == 7 ? std::vector<int64_t>{1, 32}
                                        : std::vector<int64_t>{4, -1});
        isolated.compile(*isolated_orchestration.modules.at<Attention>("attention"), query, key,
                         value, mask);
        Mln::Sdk::Tensor position(std::vector<int64_t>{});
        isolated.compile(*isolated_orchestration.modules.at<Read>("cacheread"), position);
        isolated.compile(*isolated_orchestration.modules.at<Write>("cachewrite"), position);
        Mln::Sdk::Tensor convolution_input({8, 6144}, Mln::DType::Float32);
        Mln::Sdk::Tensor convolution_kernel({6144, 4}, Mln::DType::Float32);
        Mln::Sdk::Tensor convolution_state({6144, 3}, Mln::DType::Float32);
        isolated.compile(*isolated_orchestration.modules.at<CausalConv1D>("causal_conv1d"),
                         convolution_input, convolution_kernel, convolution_state);
        Mln::Sdk::Tensor dense_lhs({2048}, Mln::DType::Float16);
        Mln::Sdk::Tensor dense_rhs({2048, 6144}, Mln::DType::Float16);
        isolated.compile(*isolated_orchestration.modules.at<DenseMatMul>("dense_matmul"), dense_lhs,
                         dense_rhs);
        Mln::Sdk::Tensor q8_codes({6144, 64, 8}, Mln::DType::Int32);
        Mln::Sdk::Tensor q8_scale({6144, 64}, Mln::DType::Float16);
        Mln::Sdk::Tensor q8_zero({6144, 64}, Mln::DType::Float16);
        isolated.compile(*isolated_orchestration.modules.at<Q8MatMul>("q8_matmul"), dense_lhs,
                         q8_codes, q8_scale, q8_zero);
        Mln::Sdk::Tensor decode_codes({1, 64, 8}, Mln::DType::Int32);
        Mln::Sdk::Tensor decode_scale({1, 64}, Mln::DType::Float16);
        Mln::Sdk::Tensor decode_zero({1, 64}, Mln::DType::Float16);
        isolated.compile(*isolated_orchestration.modules.at<Q8Decode>("q8_decode"), decode_codes,
                         decode_scale, decode_zero);
        isolated.finalize();
        if (argc == 7)
            isolated.write_vmfb(argv[4], argv[2]);
        if (isolated.graph.find("iree_linalg_ext.attention") != std::string::npos ||
            isolated.graph.find("stablehlo.dot_general") == std::string::npos ||
            isolated.graph.find("stablehlo.exponential") == std::string::npos)
            throw std::runtime_error("Vulkan Attention did not use portable decomposition");
        if (argc != 7 && isolated.graph.find("stablehlo.while") == std::string::npos)
            throw std::runtime_error("dynamic ReduceSum was not segmented");
    }

    if (argc == 7) {
        Compiler::Manifest vulkan = manifest;
        ChunkedGatedDeltaRule chunked;
        vulkan.operation_compatibilities.push_back(&chunked);
        vulkan.hooks.clear();
        TestOrchestration local_orchestration;
        Compiler::Manifest local = vulkan;
        local.operation_compatibilities.clear();
        local.policies.front().second.backend = Mln::Sdk::Backend{
            .device = "local",
            .flags = {{"iree-hal-local-target-device-backends", "llvm-cpu"}},
        };
        Compiler::Compiler local_compiler(local, local_orchestration);
        Mln::Sdk::Tensor local_query({8, 16, 128});
        Mln::Sdk::Tensor local_key({8, 16, 128});
        Mln::Sdk::Tensor local_value({8, 16, 128});
        Mln::Sdk::Tensor local_decay({8, 16});
        Mln::Sdk::Tensor local_beta({8, 16});
        Mln::Sdk::Tensor local_state({16, 128, 128});
        local_compiler.compile(*local_orchestration.modules.at<GatedDeltaRule>("gated_delta_rule"),
                               local_query, local_key, local_value, local_decay, local_beta,
                               local_state);
        local_compiler.write_vmfb(argv[5], argv[2]);

        vulkan.policies.front().second.backend = Mln::Sdk::Backend{
            .device = "vulkan",
            .flags = {{"iree-vulkan-target", "ampere"}},
        };
        TestOrchestration isolated_orchestration;
        Compiler::Compiler isolated(vulkan, isolated_orchestration);
        Mln::Sdk::Tensor query({8, 16, 128});
        Mln::Sdk::Tensor key({8, 16, 128});
        Mln::Sdk::Tensor value({8, 16, 128});
        Mln::Sdk::Tensor decay({8, 16});
        Mln::Sdk::Tensor beta({8, 16});
        Mln::Sdk::Tensor state({16, 128, 128});
        isolated.compile(*isolated_orchestration.modules.at<GatedDeltaRule>("gated_delta_rule"),
                         query, key, value, decay, beta, state);
        isolated.write_vmfb(argv[6], argv[2]);
    }

    TestOrchestration orchestration;
    Compiler::Compiler compiler(manifest, orchestration);

    Mln::Sdk::Tensor trig_input({5});
    Mln::Sdk::Tensor maxreduce_input({2, 4});
    Mln::Sdk::Tensor bitshift_input({4});
    Mln::Sdk::Tensor lookup_input({4});
    Mln::Sdk::Tensor position_input(std::vector<int64_t>{});
    Mln::Sdk::Tensor multi_lhs({4});
    Mln::Sdk::Tensor multi_rhs({4});
    Mln::Sdk::Tensor attention_query({2, 4, 3});
    Mln::Sdk::Tensor attention_key({2, 4, 3});
    Mln::Sdk::Tensor attention_value({2, 4, 5});
    Mln::Sdk::Tensor attention_mask({4, 4});
    Mln::Sdk::Tensor delta_query({4, 2, 3});
    Mln::Sdk::Tensor delta_key({4, 2, 3});
    Mln::Sdk::Tensor delta_value({4, 2, 3});
    Mln::Sdk::Tensor delta_decay({4, 2});
    Mln::Sdk::Tensor delta_beta({4, 2});
    Mln::Sdk::Tensor delta_state({2, 3, 3});

    compiler.compile(*orchestration.modules.at<Trig>("trig"), trig_input);
    compiler.compile(*orchestration.modules.at<Reduction>("maxreduce"), maxreduce_input);
    compiler.compile(*orchestration.modules.at<GenericReduction>("generic_reduction"),
                     maxreduce_input);
    compiler.compile(*orchestration.modules.at<Ordering>("ordering"), maxreduce_input);
    Mln::Sdk::Tensor random_dimensions({2}, DType::Int32);
    compiler.compile(*orchestration.modules.at<RandomValues>("random_values"), random_dimensions);
    Mln::Sdk::Tensor random_state({3}, DType::UInt64);
    compiler.compile(*orchestration.modules.at<RandomBits>("random_bits"), random_state);
    Mln::Sdk::Tensor scatter_input({4});
    Mln::Sdk::Tensor scatter_indices({2, 1}, DType::Int32);
    Mln::Sdk::Tensor scatter_updates({2});
    compiler.compile(*orchestration.modules.at<IndexedUpdate>("indexed_update"), scatter_input,
                     scatter_indices, scatter_updates);
    compiler.compile(*orchestration.modules.at<Elementwise>("elementwise"), trig_input, trig_input);
    compiler.compile(*orchestration.modules.at<Shift>("bitshift"), bitshift_input);
    compiler.compile(*orchestration.modules.at<DynamicShape>("dynamic_shape"), trig_input);
    Mln::Sdk::Tensor batch({2, 3});
    Mln::Sdk::Tensor feature({3});
    compiler.compile(*orchestration.modules.at<BatchNorm>("batch_norm"), batch, feature, feature,
                     feature, feature);
    compiler.compile(*orchestration.modules.at<BatchNormTraining>("batch_norm_training"), batch,
                     feature, feature);
    Mln::Sdk::Tensor contraction_lhs({2, 3});
    Mln::Sdk::Tensor contraction_rhs({3, 4});
    compiler.compile(*orchestration.modules.at<Contraction>("contraction"), contraction_lhs,
                     contraction_rhs);
    compiler.compile(*orchestration.modules.at<DotContraction>("dot_contraction"), contraction_lhs,
                     contraction_rhs);
    Mln::Sdk::Tensor dynamic_dimensions({2}, DType::Int32);
    compiler.compile(*orchestration.modules.at<DynamicSequence>("dynamic_sequence"),
                     dynamic_dimensions);
    compiler.compile(*orchestration.modules.at<Lookup>("lookup"), lookup_input);
    compiler.compile(*orchestration.modules.at<Read>("cacheread"), position_input);
    compiler.compile(*orchestration.modules.at<Write>("cachewrite"), position_input);
    compiler.compile(*orchestration.modules.at<Multiple>("multi_io"), multi_lhs, multi_rhs);
    compiler.compile(*orchestration.modules.at<Recurrent>("recurrent"), trig_input);
    compiler.compile(*orchestration.modules.at<Attention>("attention"), attention_query,
                     attention_key, attention_value, attention_mask);
    compiler.compile(*orchestration.modules.at<GatedDeltaRule>("gated_delta_rule"), delta_query,
                     delta_key, delta_value, delta_decay, delta_beta, delta_state);
    compiler.compile(*orchestration.modules.at<Hooked>("hooked"), trig_input);

    compiler.write_vmfb(argv[1], argv[2]);

    const Compiler::Compilation::Report &report = compiler.report();
    if (report.graph.modules.value != 23 ||
        report.graph.modules.quality != Compiler::Metric::Quality::Exact ||
        !report.graph.operations_before_iree.value ||
        !report.memory.estimated_peak_live_bytes.value ||
        report.memory.estimated_peak_live_bytes.quality != Compiler::Metric::Quality::Estimated ||
        report.functions.size() != 23 ||
        !report.functions.front().estimated_peak_live_bytes.value ||
        report.functions.front().backend_affinity.empty() ||
        report.partitioning.partitions.value != 1 ||
        report.partitioning.estimated_transfers.value != 0 || report.lifetimes.empty() ||
        report.text().find("Functions\n") == std::string::npos ||
        report.text().find("operations after IREE: unknown [unknown]") == std::string::npos)
        throw std::runtime_error("incomplete compilation report");
    return 0;
}
