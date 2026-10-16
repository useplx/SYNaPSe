#include "F32/f32.hpp"
#include "mln/sdk/functional/broadcast.hpp"
#include "mln/sdk/functional/causal_conv1d.hpp"
#include "mln/sdk/functional/const.hpp"
#include "mln/sdk/functional/gated_delta_rule.hpp"
#include "mln/sdk/functional/matmul.hpp"
#include "mln/sdk/functional/reducesum.hpp"
#include "mln/sdk/functional/transpose.hpp"
#include "syn/compiler.hpp"
#include "syn/package.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

extern "C" const Syn::Compiler::Package::Object *mln_package_vulkan_gated_delta_rule();
extern "C" const Syn::Compiler::Package::Object *mln_package_vulkan_support();
extern "C" const Syn::Compiler::Package::Object *mln_package_vulkan_broadcast();
extern "C" const Syn::Compiler::Package::Object *mln_package_vulkan_reduction();
extern "C" const Syn::Compiler::Package::Object *mln_package_vulkan_matmul();
extern "C" const Syn::Compiler::Package::Object *mln_package_adreno_matmul();
extern "C" const Syn::Compiler::Package::Object *mln_package_adreno_reduction();
extern "C" const Syn::Compiler::Package::Object *mln_package_llvm_support();

namespace {
    class Expansion final : public Mln::Sdk::Operation::Compatibility {
        public:
            Expansion(bool value, int selected_priority)
                : value_(value), priority_(selected_priority) {}

            bool applies(const Context &) const override { return true; }
            std::optional<bool> expand(const Context &) const override { return value_; }
            int priority() const override { return priority_; }

        private:
            bool value_;
            int priority_;
    };

    void check_selection() {
        using namespace Mln::Sdk;
        auto *vulkan_support =
            std::get<Mln::Sdk::Operation::Compatibility *>(*mln_package_vulkan_support());
        auto *llvm_support =
            std::get<Mln::Sdk::Operation::Compatibility *>(*mln_package_llvm_support());
        Tensor support_input({16, 128}, Mln::DType::Float16);
        Functional::CausalConv1D support_operation(support_input,
                                                   Tensor({128, 4}, Mln::DType::Float16),
                                                   Tensor({128, 3}, Mln::DType::Float16));
        Mln::Sdk::Operation::Compatibility::Context support_context{
            .operation = support_operation,
            .target = Mln::Sdk::Operation::Compatibility::Target{.name = "vulkan",
                                                                 .device = "vulkan",
                                                                 .properties = {}},
        };
        if (!vulkan_support->supports(support_context) || vulkan_support->applies(support_context))
            throw std::runtime_error("Vulkan native support was confused with graph correction");
        support_context.target = Mln::Sdk::Operation::Compatibility::Target{
            .name = "local", .device = "local", .properties = {}};
        if (!llvm_support->supports(support_context) || llvm_support->applies(support_context))
            throw std::runtime_error("LLVM native support was confused with graph correction");

        auto *compatibility =
            std::get<Mln::Sdk::Operation::Compatibility *>(*mln_package_vulkan_reduction());
        for (const int64_t extent : std::array<int64_t, 2>{128, Tensor::Dynamic}) {
            Tensor input({16, extent, 128}, Mln::DType::Float32);
            Functional::ReduceSum reduction(input, {1});
            Functional::Broadcast broadcast(Tensor({16, 128}), {16, 128, 128}, {0, 2});
            std::vector<const Operation *> consumers{&broadcast};
            Mln::Sdk::Operation::Compatibility::Target target{
                .name = "vulkan", .device = "vulkan", .properties = {}};
            Mln::Sdk::Operation::Compatibility::Context context{
                .operation = reduction, .consumers = consumers, .target = target};
            if (!compatibility->applies(context) || !compatibility->barrier(context))
                throw std::runtime_error("strided reduction broadcast compatibility missing");
            context.consumers = {};
            if (compatibility->applies(context))
                throw std::runtime_error("standalone reduction unnecessarily separated");
            context.consumers = consumers;
            reduction.axes = {2};
            if (compatibility->applies(context))
                throw std::runtime_error("contiguous reduction unnecessarily separated");
        }

        auto *matmul_compatibility =
            std::get<Mln::Sdk::Operation::Compatibility *>(*mln_package_vulkan_matmul());
        Tensor lhs({8, 2048}, Mln::DType::Float16);
        Tensor weight({6144, 2048}, Mln::DType::Float16);
        weight.def = std::make_shared<Functional::Const>(Weight{"weight", {6144, 2048}}, true);
        Tensor rhs({2048, 6144}, Mln::DType::Float16);
        rhs.def = std::make_shared<Functional::Transpose>(weight, std::vector<int64_t>{1, 0});
        Functional::MatMul matmul(lhs, rhs);
        std::vector<std::pair<std::string, std::string>> target_properties{
            {"iree-vulkan-target", "adreno-643"}};
        Mln::Sdk::Operation::Compatibility::Context matmul_context{
            .operation = matmul,
            .target = Mln::Sdk::Operation::Compatibility::Target{.name = "vulkan",
                                                                 .device = "vulkan",
                                                                 .properties = target_properties},
        };
        if (!matmul_compatibility->applies(matmul_context) ||
            !matmul_compatibility->linearize_reduction(matmul_context) ||
            matmul_compatibility->contraction_tile(matmul_context) != 64)
            throw std::runtime_error("Vulkan mapped MatMul compatibility missing");
        auto *adreno = std::get<Mln::Sdk::Operation::Compatibility *>(*mln_package_adreno_matmul());
        if (!adreno->applies(matmul_context) || !adreno->unroll(matmul_context) ||
            adreno->tile(matmul_context, -1, 6144) != 256 ||
            adreno->fuse_producer(matmul_context) != true ||
            adreno->linearize_reduction(matmul_context) != false ||
            adreno->contraction_tile(matmul_context) != 2048)
            throw std::runtime_error("Adreno fused projection compatibility missing");
        matmul.lhs.shape = {2048};
        if (adreno->fuse_producer(matmul_context) != false)
            throw std::runtime_error("Adreno decode producer resolution missing");
        matmul.rhs.shape = {2048, 151936};
        if (adreno->tile(matmul_context, -1, 151936) != 2048 ||
            adreno->fuse_producer(matmul_context) != true ||
            adreno->linearize_reduction(matmul_context) != false ||
            adreno->contraction_tile(matmul_context) != 512 || !adreno->unroll(matmul_context))
            throw std::runtime_error("Adreno wide matvec compatibility missing");
        matmul.lhs.shape = {8, 2048};
        matmul.lhs.shape[0] = Tensor::Dynamic;
        if (adreno->applies(matmul_context))
            throw std::runtime_error("static Adreno tiling selected for a dynamic projection");
        matmul.lhs.shape = {8, 2048};
        matmul.rhs.def.reset();
        if (adreno->applies(matmul_context))
            throw std::runtime_error("Adreno weight projection selected for a generic MatMul");

        auto *adreno_reduction =
            std::get<Mln::Sdk::Operation::Compatibility *>(*mln_package_adreno_reduction());
        Tensor reduction_input({2048}, Mln::DType::Float32);
        Functional::ReduceSum reduction(reduction_input, {0});
        Mln::Sdk::Operation::Compatibility::Context reduction_context{
            .operation = reduction,
            .target = Mln::Sdk::Operation::Compatibility::Target{.name = "vulkan",
                                                                 .device = "vulkan",
                                                                 .properties = target_properties},
        };
        if (!adreno_reduction->applies(reduction_context) ||
            adreno_reduction->tile(reduction_context, 0, 2048) != 64)
            throw std::runtime_error("Adreno reduction tile compatibility missing");

        Expansion portable_expansion{true, 0};
        Expansion target_preservation{false, 100};
        Syn::Compiler::Manifest resolution_manifest{
            .policies = {{"", Syn::Compiler::Police{}}},
            .operation_compatibilities = {&target_preservation, &portable_expansion},
        };
        std::vector<Mln::Weight::Descriptor> required_weights;
        const std::vector<Mln::Weight::Descriptor> available_weights;
        std::vector<Syn::Compiler::Compilation::Legalization> legalizations;
        Syn::Compiler::Compatibility::Matrix matrix(resolution_manifest);
        Syn::Compiler::Operations resolver(resolution_manifest, "delta", required_weights,
                                           available_weights, legalizations, matrix);
        if (resolver.expand(reduction))
            throw std::runtime_error("higher-priority compatibility did not preserve operation");
    }
    class Reduction final : public Mln::Sdk::Module {
        public:
            Reduction() { name = "delta"; }
            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor input) {
                return operations.insert<Mln::Sdk::Functional::ReduceSum>(input,
                                                                          std::vector<int64_t>{1});
            }
    };
    class Delta final : public Mln::Sdk::Module {
        public:
            Delta() { name = "delta"; }

            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor query, Mln::Sdk::Tensor key,
                                     Mln::Sdk::Tensor value, Mln::Sdk::Tensor decay,
                                     Mln::Sdk::Tensor beta, Mln::Sdk::Tensor state) {
                return operations.insert<Mln::Sdk::Functional::GatedDeltaRule>(
                    query, key, value, decay, beta, state, 1e-6);
            }
    };

    class Convolution final : public Mln::Sdk::Module {
        public:
            Convolution() { name = "convolution"; }

            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor input, Mln::Sdk::Tensor kernel,
                                     Mln::Sdk::Tensor state) {
                return operations.insert<Mln::Sdk::Functional::CausalConv1D>(
                    std::move(input), std::move(kernel), std::move(state));
            }
    };
}

int main(int argc, char **argv) {
    check_selection();
    if (argc == 2 && std::string_view(argv[1]) == "--selection")
        return 0;
    if (argc < 2 || argc > 5)
        return 2;
    const bool recurrent = argc == 5 && std::string_view(argv[4]) == "--recurrent";
    std::filesystem::create_directories(argv[1]);
    for (const std::string device : {"local", "vulkan"}) {
        for (const int64_t tokens : {1, 8, 64}) {
            Mln::Sdk::Backend backend{.device = device};
            backend.flags =
                device == "local"
                    ? std::vector<std::pair<std::string,
                                            std::string>>{{"iree-hal-local-target-device-backends",
                                                           "llvm-cpu"}}
                    : std::vector<std::pair<std::string, std::string>>{
                          {"iree-vulkan-target", argc >= 3 ? argv[2] : "ampere"}};
            if (device == "vulkan") {
                if (argc == 4) {
                    std::ifstream file(argv[3]);
                    const std::string target{std::istreambuf_iterator<char>(file), {}};
                    if (target.empty())
                        throw std::runtime_error("empty target description");
                    backend.flags.emplace_back("iree-hal-target-device", target);
                }
                const std::string dumps =
                    std::string(argv[1]) + "/executables-" + std::to_string(tokens);
                backend.flags.emplace_back("iree-hal-dump-executable-sources-to", dumps);
                backend.flags.emplace_back("iree-hal-dump-executable-binaries-to", dumps);
            }
            Syn::Compiler::Manifest manifest{
                .policies = {{"delta",
                              {.schemes = {Mln::Sdk::Codec::Scheme{
                                   .scope = Mln::Sdk::Codec::Scope::Constant,
                                   .lifetime = Mln::Sdk::Codec::Lifetime::Immutable,
                                   .update = Mln::Sdk::Codec::Update::None,
                                   .format = Quantizations::F32{},
                               }},
                               .dtype = Mln::DType::Float32,
                               .backend = backend}}},
                .formats = {Quantizations::F32{}},
                .operation_compatibilities =
                    {std::get<Mln::Sdk::Operation::Compatibility *>(
                         *mln_package_vulkan_broadcast()),
                     std::get<Mln::Sdk::Operation::Compatibility *>(*mln_package_vulkan_support()),
                     std::get<Mln::Sdk::Operation::Compatibility *>(*mln_package_llvm_support()),
                     std::get<Mln::Sdk::Operation::Compatibility *>(
                         *mln_package_vulkan_reduction()),
                     std::get<Mln::Sdk::Operation::Compatibility *>(*mln_package_vulkan_matmul()),
                     std::get<Mln::Sdk::Operation::Compatibility *>(
                         *mln_package_adreno_reduction())},
            };
            if (!recurrent)
                manifest.operation_compatibilities.push_back(
                    std::get<Mln::Sdk::Operation::Compatibility *>(
                        *mln_package_vulkan_gated_delta_rule()));
            Delta delta;
            Syn::Compiler::Compiler compiler(manifest, "delta");
            Mln::Sdk::Tensor query({tokens, 16, 128}, Mln::DType::Float32);
            Mln::Sdk::Tensor key({tokens, 16, 128}, Mln::DType::Float32);
            Mln::Sdk::Tensor value({tokens, 16, 128}, Mln::DType::Float32);
            Mln::Sdk::Tensor decay({tokens, 16}, Mln::DType::Float32);
            Mln::Sdk::Tensor beta({tokens, 16}, Mln::DType::Float32);
            Mln::Sdk::Tensor state({16, 128, 128}, Mln::DType::Float32);
            compiler.compile_as_impl("update", "update", true, delta, query, key, value, decay,
                                     beta, state);
            Convolution convolution;
            Mln::Sdk::Tensor convolution_input({tokens, 6144}, Mln::DType::Float32);
            Mln::Sdk::Tensor convolution_kernel({6144, 4}, Mln::DType::Float32);
            Mln::Sdk::Tensor convolution_state({6144, 3}, Mln::DType::Float32);
            compiler.compile_as_impl("convolution", "convolution", true, convolution,
                                     convolution_input, convolution_kernel, convolution_state);
            Reduction reduction;
            Mln::Sdk::Tensor matrix({16, 128, 128}, Mln::DType::Float32);
            compiler.compile_as_impl("reduce", "reduce", true, reduction, matrix);
            const std::string prefix = std::string(argv[1]) + "/" + device + "-" +
                                       std::to_string(tokens) + (recurrent ? "-recurrent" : "");
            compiler.write_vmfb(prefix + ".vmfb", prefix + ".devices");
        }
    }
}
