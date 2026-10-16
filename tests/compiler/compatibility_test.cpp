#include "mln/error.hpp"
#include "mln/sdk/functional/reducesum.hpp"
#include "syn/compatibility.hpp"
#include "syn/manifest.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {
    class Format final : public Mln::Sdk::Codec {
        public:
            Format() { name = "test"; }

            Mln::Sdk::Tensor decode(Mln::Sdk::Operations &, const Mln::Sdk::Weight &weight,
                                    Mln::DType dtype) override {
                return Mln::Sdk::Tensor(weight.shape, dtype);
            }
    };

    class Codec final : public Syn::Compiler::Codec::Compatibility {
        public:
            bool applies(const Context &context) const override {
                return context.backend == "vulkan" && context.codec.name == "test";
            }
    };

    class Reduction final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &context) const override {
                return context.target && context.target->device == "vulkan" &&
                       context.target->property("chip") == "adreno-643" &&
                       context.operation.kind == Mln::Sdk::Functional::Kind::ReduceSum;
            }
    };

    class NativeReduction final : public Mln::Sdk::Operation::Compatibility {
        public:
            bool applies(const Context &) const override { return false; }
            bool supports(const Context &context) const override {
                return context.target && context.target->device == "local" &&
                       context.operation.kind == Mln::Sdk::Functional::Kind::ReduceSum;
            }
    };

    Mln::DiagnosticAction count(const Mln::Diagnostic &diagnostic, void *data) {
        if (diagnostic.code() != "synapse:E2102")
            return Mln::DiagnosticAction::Continue;
        ++*static_cast<int *>(data);
        return Mln::DiagnosticAction::Ignore;
    }

    Mln::DiagnosticAction promote(const Mln::Diagnostic &, void *) {
        return Mln::DiagnosticAction::PromoteToError;
    }
}

int main() {
    Reduction reduction;
    Codec codec_rule;
    Syn::Compiler::Manifest manifest{
        .codec_compatibilities = {&codec_rule},
        .operation_compatibilities = {&reduction},
    };
    Syn::Compiler::Compatibility::Matrix matrix(manifest);

    Mln::Sdk::Tensor input({4, 8}, Mln::DType::Float32);
    Mln::Sdk::Functional::ReduceSum operation(input, {1});
    operation.policy_path = "model.norm";
    const std::vector<std::pair<std::string, std::string>> adreno{{"chip", "adreno-643"}};
    Mln::Sdk::Operation::Compatibility::Target vulkan{
        .name = "vulkan", .device = "vulkan", .properties = adreno};
    Mln::Sdk::Operation::Compatibility::Context supported{.operation = operation, .target = vulkan};
    if (!matrix.operation(supported, operation.policy_path))
        throw std::runtime_error("implemented compatibility was reported missing");
    const auto *target_entry = matrix.find(Syn::Compiler::Compatibility::Subject::Operation,
                                           "vulkan", "reduce-sum", adreno);
    if (!target_entry || !target_entry->implemented)
        throw std::runtime_error("target properties were not indexed");

    const std::vector<std::pair<std::string, std::string>> mali{{"chip", "mali"}};
    Mln::Sdk::Operation::Compatibility::Target other{
        .name = "vulkan", .device = "vulkan", .properties = mali};
    Mln::Sdk::Operation::Compatibility::Context uncovered{.operation = operation, .target = other};
    int target_warnings = 0;
    {
        Mln::ScopedDiagnosticHandler handler(count, &target_warnings);
        if (matrix.operation(uncovered, operation.policy_path))
            throw std::runtime_error("target-specific compatibility leaked to another device");
    }
    if (target_warnings != 1 || matrix.find(Syn::Compiler::Compatibility::Subject::Operation,
                                            "vulkan", "reduce-sum", mali) == nullptr)
        throw std::runtime_error("target-specific compatibility gap was not recorded");

    Mln::Sdk::Operation::Compatibility::Target local{
        .name = "local", .device = "local", .properties = {}};
    Mln::Sdk::Operation::Compatibility::Context missing{.operation = operation, .target = local};
    int warnings = 0;
    {
        Mln::ScopedDiagnosticHandler handler(count, &warnings);
        if (matrix.operation(missing, operation.policy_path))
            throw std::runtime_error("missing compatibility was reported implemented");
        matrix.operation(missing, "model.other");
    }
    if (warnings != 1)
        throw std::runtime_error("missing compatibility warning was not deduplicated");

    const auto *entry =
        matrix.find(Syn::Compiler::Compatibility::Subject::Operation, "local", "reduce-sum");
    if (!entry || entry->implemented || entry->uses != 2 || entry->covered != 0 ||
        entry->path != "model.norm")
        throw std::runtime_error("compatibility observation is incomplete");

    NativeReduction native_reduction;
    Syn::Compiler::Manifest native_manifest{.operation_compatibilities = {&native_reduction}};
    Syn::Compiler::Compatibility::Matrix native_matrix(native_manifest);
    if (!native_matrix.operation(missing, operation.policy_path) ||
        native_reduction.applies(missing))
        throw std::runtime_error("native support was confused with graph correction");

    Format format;
    Mln::Sdk::Codec::Scheme scheme;
    Mln::Sdk::Weight weight{"model.weight", {4, 8}};
    Syn::Compiler::Codec::Compatibility::Context codec{
        .codec = format,
        .scheme = scheme,
        .weight = weight,
        .dtype = Mln::DType::Float16,
        .parameter_path = weight.name,
        .backend = "vulkan",
    };
    if (!matrix.codec(codec))
        throw std::runtime_error("implemented codec compatibility was reported missing");
    const auto *codec_entry =
        matrix.find(Syn::Compiler::Compatibility::Subject::Codec, "vulkan", "test");
    if (!codec_entry || !codec_entry->implemented || codec_entry->covered != 1)
        throw std::runtime_error("codec compatibility observation is incomplete");

    Syn::Compiler::Manifest canonical_manifest;
    Syn::Compiler::Compatibility::Matrix canonical_matrix(canonical_manifest);
    if (!canonical_matrix.codec(codec))
        throw std::runtime_error("canonical codec graph required a redundant target adapter");

    Syn::Compiler::Manifest empty;
    Syn::Compiler::Compatibility::Matrix promoted(empty);
    bool raised = false;
    try {
        Mln::ScopedDiagnosticHandler handler(promote);
        promoted.operation(missing, operation.policy_path);
    } catch (const Mln::Diagnostic &diagnostic) {
        raised = diagnostic.code() == "synapse:E2102";
    }
    if (!raised)
        throw std::runtime_error("compatibility warning did not follow diagnostic policy");
}
