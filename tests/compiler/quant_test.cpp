#include "mln/error.hpp"
#include "mln/sdk/codec.hpp"
#include "mln/sdk/functional.hpp"
#include "mln/sdk/module.hpp"
#include "mln/sdk/orchestration.hpp"
#include "mln/weight.hpp"
#include "syn/compiler.hpp"
#include "syn/errors/manifest.hpp"

#include <algorithm>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    using MQ = Mln::Sdk::Codec;
    namespace Sdk = Mln;

    std::vector<uint8_t> raw_bytes(std::span<const float> values) {
        std::vector<uint8_t> bytes(values.size() * sizeof(float));
        if (!bytes.empty())
            std::memcpy(bytes.data(), values.data(), bytes.size());
        return bytes;
    }

    std::vector<float> raw_floats(std::span<const uint8_t> bytes) {
        std::vector<float> values(bytes.size() / sizeof(float));
        if (!bytes.empty())
            std::memcpy(values.data(), bytes.data(), bytes.size());
        return values;
    }

    class Codec final : public Mln::Sdk::Codec {
        public:
            explicit Codec(std::string label) { name = std::move(label); }

            Mln::Sdk::Tensor decode(Mln::Sdk::Operations &ops, const Mln::Sdk::Weight &weight,
                                    Mln::DType dtype) override {
                Mln::Sdk::Tensor value =
                    ops.insert<Mln::Sdk::Functional::Const>(weight, false, Mln::DType::Float32);
                return dtype == Mln::DType::Float32
                           ? value
                           : ops.insert<Mln::Sdk::Functional::Convert>(std::move(value), dtype);
            }

            std::vector<uint8_t> encode(std::span<const float> values) const override {
                return raw_bytes(values);
            }

            std::vector<float> decode(std::span<const uint8_t> bytes) const override {
                return raw_floats(bytes);
            }
    };

    template <typename Format> MQ::Scheme constant(Format format) {
        return MQ::Scheme{
            .scope = MQ::Scope::Constant,
            .lifetime = MQ::Lifetime::Immutable,
            .update = MQ::Update::None,
            .format = std::move(format),
        };
    }

    template <typename Format>
    MQ::Scheme dynamic(Format format, MQ::Lifetime lifetime, MQ::Update update) {
        MQ::Scheme scheme{
            .scope = MQ::Scope::Internal,
            .lifetime = lifetime,
            .update = update,
            .format = std::move(format),
        };
        return scheme;
    }

    class Vision final : public Mln::Sdk::Module {
        public:
            Vision() { parameters.insert("weight", {{}, {512}}); }

            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor input) {
                return operations.insert<Mln::Sdk::Functional::Multiply>(input,
                                                                         parameters.at("weight"));
            }
    };

    class Language final : public Mln::Sdk::Module {
        public:
            Language() {
                parameters.insert("q4", {{}, {512}});
                parameters.insert("q6", {{}, {512}});
            }

            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor input) {
                auto value =
                    operations.insert<Mln::Sdk::Functional::Multiply>(input, parameters.at("q4"));
                return operations.insert<Mln::Sdk::Functional::Multiply>(value,
                                                                         parameters.at("q6"));
            }
    };

    class Convolution final : public Mln::Sdk::Module {
        public:
            Convolution() {
                parameters.insert("weight", {{}, {32, 16, 3, 3}});
                parameters.insert("bias", {{}, {32}});
            }

            Mln::Sdk::Tensor forward(Mln::Sdk::Tensor input) {
                return operations.insert<Mln::Sdk::Functional::Conv2D>(
                    input, parameters.at("weight"), parameters.at("bias"),
                    std::vector<int64_t>{1, 1}, std::vector<int64_t>{0, 0, 0, 0},
                    std::vector<int64_t>{1, 1}, 1);
            }
    };

    class Orchestration final : public Mln::Sdk::Orchestration {
        public:
            Orchestration() {
                name = "test";
                modules.insert<Vision>("vision");
                modules.insert<Language>("language");
                modules.insert<Convolution>("convolution");
            }

            Mln::Buffer forward(Mln::Runtime::Session &, Mln::Buffer input) override {
                return input;
            }
    };
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5)
        return 2;
    namespace Compiler = Syn::Compiler;

    Compiler::Manifest manifest{
        .policies =
            {
                {"model",
                 Compiler::Police{
                     .schemes =
                         {
                             constant(Codec{"Q4_K"}),
                         },
                     .dtype = Mln::DType::Float32,
                     .backend =
                         Mln::Sdk::Backend{
                             .device = "local",
                             .flags = {{"iree-hal-local-target-device-backends", "llvm-cpu"}},
                         },
                 }},
                {"vision",
                 Compiler::Police{
                     .schemes = {constant(Codec{"F16"})},
                 }},
                {"language.q4",
                 Compiler::Police{
                     .schemes = {constant(Codec{"Q4_K"})},
                 }},
                {"language.q6",
                 Compiler::Police{
                     .schemes = {constant(Codec{"Q6_K"})},
                 }},
                {"convolution.weight",
                 Compiler::Police{
                     .schemes = {constant(Codec{"Q4_K"})},
                 }},
                {"convolution.bias",
                 Compiler::Police{
                     .schemes = {constant(Codec{"F32"})},
                 }},
            },
        .formats =
            {
                Codec{"Q4_K"},
                Codec{"Q6_K"},
                Codec{"F16"},
                Codec{"F32"},
            },
    };

    std::vector<Mln::Weight::Descriptor> weights{
        {"vision.weight", "F32", "F32", {512}, 2048},
        {"language.q4", "F32", "F32", {512}, 2048},
        {"language.q6", "F32", "F32", {512}, 2048},
        {"convolution.weight", "F32", "F32", {32, 16, 3, 3}, 18432},
        {"convolution.bias", "F32", "F32", {32}, 128},
    };
    if (argc == 5) {
        weights = Mln::Weight::decode_weight_index(Mln::Weight::read_binary_file(argv[4]),
                                                   Mln::Weight::Index::Kind::Descriptors);
    }

    try {
        Compiler::Manifest invalid{
            .policies = {{"model",
                          Compiler::Police{
                              .schemes =
                                  {
                                      constant(Codec{"F32"}),
                                      dynamic(Codec{"F32"}, MQ::Lifetime::Session,
                                              MQ::Update::Region),
                                  },
                              .dtype = Mln::DType::Float32,
                          }}},
        };
        Compiler::validate_manifest(invalid);
        throw std::runtime_error("Constant-only scheme accepted Session lifetime");
    } catch (const Mln::Diagnostic &error) {
        if (error.error() != Compiler::Errors::Manifest::Scheme::Lifetime::Unsupported)
            throw;
    }

    const Compiler::Police vision_policy = Compiler::police_for(manifest, "vision.weight");
    const auto immutable_count = std::count_if(
        vision_policy.schemes.begin(), vision_policy.schemes.end(),
        [](const auto &scheme) { return scheme.lifetime == MQ::Lifetime::Immutable; });
    if (immutable_count != 1 || vision_policy.schemes.back().format.get().name != "F16")
        throw std::runtime_error("Scheme override did not replace only the Immutable lifetime");

    Compiler::Manifest precedence{
        .policies =
            {
                {"model",
                 Compiler::Police{
                     .schemes = {constant(Codec{"ROOT"})},
                     .dtype = Mln::DType::Float32,
                 }},
                {"model.child",
                 Compiler::Police{
                     .schemes = {MQ::Scheme{
                         .scope = MQ::Scope::Any,
                         .lifetime = MQ::Lifetime::Any,
                         .update = MQ::Update::None,
                         .format = Codec{"CHILD"},
                     }},
                 }},
            },
    };
    const auto selected = Compiler::scheme_for(precedence, "model.child.weight",
                                               MQ::Scope::Constant, MQ::Lifetime::Immutable);
    if (!selected || selected.scheme->format.get().name != "CHILD")
        throw std::runtime_error("Scheme raiz venceu indevidamente a Police filha");

    Compiler::Manifest wildcard_precedence{
        .policies =
            {
                {"model",
                 Compiler::Police{
                     .schemes = {constant(Codec{"ROOT"})},
                     .dtype = Mln::DType::Float32,
                     .backend = Mln::Sdk::Backend{.device = "vulkan"},
                 }},
                {"model.layers.*.projection.weight",
                 Compiler::Police{
                     .schemes = {constant(Codec{"WILDCARD"})},
                 }},
                {"model.layers.0.projection.weight",
                 Compiler::Police{
                     .schemes = {constant(Codec{"LITERAL"})},
                     .backend = Mln::Sdk::Backend{.device = "qnn"},
                 }},
            },
    };
    const std::string literal_path = "model.layers.0.projection.weight";
    const auto literal_scheme = Compiler::scheme_for(wildcard_precedence, literal_path,
                                                     MQ::Scope::Constant, MQ::Lifetime::Immutable);
    const Compiler::Police literal_policy = Compiler::police_for(wildcard_precedence, literal_path);
    if (!literal_scheme || literal_scheme.scheme->format.get().name != "LITERAL" ||
        !literal_policy.backend || literal_policy.backend->device != "qnn")
        throw std::runtime_error("policy wildcard venceu indevidamente o caminho literal");

    Orchestration orchestration;
    Compiler::Compiler compiler(manifest, orchestration, weights);
    Mln::Sdk::Tensor vision_input({512});
    Mln::Sdk::Tensor language_input({512});
    Mln::Sdk::Tensor convolution_input({1, 16, 3, 3});
    compiler.compile(*orchestration.modules.at<Vision>("vision"), vision_input);
    compiler.compile(*orchestration.modules.at<Language>("language"), language_input);
    compiler.compile(*orchestration.modules.at<Convolution>("convolution"), convolution_input);
    compiler.write_vmfb(argv[1], argv[2], argv[3]);

    const auto requirements = Mln::Weight::decode_weight_index(
        Mln::Weight::read_binary_file(argv[3]), Mln::Weight::Index::Kind::Requirements);
    if (requirements.size() != 5 || requirements[0].quant != "F16" ||
        requirements[1].quant != "Q4_K" || requirements[2].quant != "Q6_K" ||
        requirements[3].quant != "Q4_K" || requirements[4].quant != "F32")
        throw std::runtime_error("mixed plan did not preserve per-tensor codecs");
    return 0;
}
