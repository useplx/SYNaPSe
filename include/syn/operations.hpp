#pragma once

#include "mln/error.hpp"
#include "mln/sdk/backend/compatibility.hpp"
#include "mln/sdk/functional/const.hpp"
#include "mln/sdk/functional/input.hpp"
#include "mln/sdk/kind.hpp"
#include "mln/sdk/operation.hpp"
#include "mln/sdk/operations.hpp"
#include "mln/sdk/tensor.hpp"

#include "mln/dtype.hpp"
#include "syn/compatibility.hpp"
#include "syn/deployment.hpp"
#include "syn/errors/graph.hpp"
#include "syn/manifest.hpp"
#include "syn/operation.hpp"
#include "syn/report.hpp"

#include "mln/weight.hpp"

#include <iree/compiler/Dialect/Util/IR/UtilOps.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <llvm/Support/Casting.h>

#include <algorithm>
#include <format>
#include <memory>
#include <print>
#include <stdexcept>
#include <string>

namespace Syn::Compiler {
    namespace Format = Mln::Weight;
}
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Syn::Compiler {
    namespace Diagnostics {
        inline constexpr Mln::Error BackendCorrectorMissing{
            "synapse", 2100, "backend '{}' has no loaded graph corrector"};
        inline constexpr Mln::Error BackendLegalization{"synapse", 2101,
                                                        "backend '{}' {} operation kind {}"};
    }

    class Operations {
        private:
            std::unordered_map<Mln::Sdk::Functional::Kind, std::unique_ptr<Operation::Emitter>>
                registry;
            std::vector<Mln::Sdk::Operation::Compatibility *> compatibilities;

            std::unordered_map<const Mln::Sdk::Operation *, std::vector<mlir::Value>> cache;
            std::vector<const Mln::Sdk::Operation *> emission_stack;
            std::unordered_set<std::string> legalization_diagnostics;

            Mln::Sdk::Operations expansions;
            const Manifest &manifest_;
            std::string_view entrypoint_;
            std::vector<Mln::Weight::Descriptor> &weights_;
            const std::vector<Mln::Weight::Descriptor> &available_weights_;
            std::vector<Compilation::Legalization> &legalizations_;
            Compatibility::Matrix &matrix_;

            static bool within(const std::string &path, std::string_view prefix) {
                return path == prefix ||
                       (path.size() > prefix.size() &&
                        path.compare(0, prefix.size(), prefix) == 0 && path[prefix.size()] == '.');
            }

            std::span<const Mln::Sdk::Operation *const> consumer_chain() const {
                std::size_t size = emission_stack.size();
                if (size && emission_stack.back()->kind == Mln::Sdk::Functional::Kind::Const)
                    --size;
                return {emission_stack.data(), size};
            }

            static std::optional<Mln::Sdk::Operation::Compatibility::Target>
            target(const Police &police) {
                if (!police.backend)
                    return std::nullopt;
                return Mln::Sdk::Operation::Compatibility::Target{
                    .name = police.backend->device,
                    .device = police.backend->device,
                    .properties = police.backend->flags,
                };
            }

            Mln::Sdk::Backend::Compatibility *backend(std::string_view device) const {
                const auto found = std::ranges::find_if(
                    manifest_.backend_compatibilities, [&](const auto *candidate) {
                        return candidate && candidate->name() == device;
                    });
                return found == manifest_.backend_compatibilities.end() ? nullptr : *found;
            }

            void validate_backend(const Mln::Sdk::Operation &operation, const Police &police) {
                const Mln::Sdk::Operation::Compatibility::Context context{
                    .operation = operation,
                    .consumers = consumer_chain(),
                    .target = target(police),
                };
                matrix_.operation(context, operation.policy_path);
                if (!police.backend)
                    return;
                auto *corrector = backend(police.backend->device);
                if (!corrector) {

                    const bool native =
                        police.backend->device == "local" || police.backend->device == "vulkan" ||
                        police.backend->device == "cuda" || police.backend->device == "hip";
                    if (native)
                        return;
                    const std::string diagnostic = std::format(
                        "backend '{}' has no loaded graph corrector", police.backend->device);
                    if (manifest_.validation.compatibility == Manifest::Enforcement::Strict ||
                        !manifest_.validation.allow_fallback)
                        Mln::raise_at(Diagnostics::BackendCorrectorMissing, operation.policy_path,
                                      police.backend->device);
                    if (legalization_diagnostics.insert(diagnostic).second)
                        Mln::warn_at(Diagnostics::BackendCorrectorMissing, operation.policy_path,
                                     police.backend->device);
                    return;
                }
                const auto legality = corrector->legality(context);
                const bool approximate =
                    legality == Mln::Sdk::Backend::Compatibility::Legality::Approximate;
                legalizations_.push_back(Compilation::Legalization{
                    .backend = police.backend->device,
                    .path = operation.policy_path,
                    .operation = static_cast<int>(operation.kind),
                    .outcome = legality == Mln::Sdk::Backend::Compatibility::Legality::Exact
                                   ? "exact"
                               : approximate ? "approximate"
                                             : "unsupported",
                });
                if (legality == Mln::Sdk::Backend::Compatibility::Legality::Exact)
                    return;
                const bool strict =
                    approximate
                        ? manifest_.validation.numeric == Manifest::Enforcement::Strict
                        : manifest_.validation.compatibility == Manifest::Enforcement::Strict ||
                              !manifest_.validation.allow_fallback;
                const std::string diagnostic =
                    std::format("backend '{}' {} operation kind {} at '{}'", police.backend->device,
                                approximate ? "only approximates" : "cannot legalize",
                                static_cast<int>(operation.kind), operation.policy_path);
                if (strict)
                    Mln::raise_at(Diagnostics::BackendLegalization, operation.policy_path,
                                  police.backend->device,
                                  approximate ? "only approximates" : "cannot legalize",
                                  static_cast<int>(operation.kind));
                if (legalization_diagnostics.insert(diagnostic).second)
                    Mln::warn_at(Diagnostics::BackendLegalization, operation.policy_path,
                                 police.backend->device,
                                 approximate ? "only approximates" : "cannot legalize",
                                 static_cast<int>(operation.kind));
            }

            static bool cacheable(const Mln::Sdk::Operation &) { return true; }

            bool crosses_device_boundary(const Police &producer) const {
                if (!producer.backend)
                    return false;
                const Police consumer = emission_stack.empty()
                                            ? police_for(std::string(entrypoint_))
                                            : police_for(emission_stack.back()->policy_path);
                return consumer.backend && (producer.backend->device != consumer.backend->device ||
                                            producer.backend->ordinal != consumer.backend->ordinal);
            }

        public:
            Operations(const Manifest &manifest, std::string_view entrypoint,
                       std::vector<Mln::Weight::Descriptor> &weights,
                       const std::vector<Mln::Weight::Descriptor> &available_weights,
                       std::vector<Compilation::Legalization> &legalizations,
                       Compatibility::Matrix &matrix)
                : compatibilities(manifest.operation_compatibilities), manifest_(manifest),
                  entrypoint_(entrypoint), weights_(weights), available_weights_(available_weights),
                  legalizations_(legalizations), matrix_(matrix) {
                for (auto *backend : manifest.backend_compatibilities)
                    compatibilities.push_back(backend);
            }

            template <typename T> void insert() {
                auto op = std::make_unique<T>();
                registry[op->kind] = std::move(op);
            }

            Police police_for(const std::string &path) const {
                Police result = Syn::Compiler::police_for(manifest_, path);
                if (result.backend && result.entrypoints &&
                    std::ranges::find(*result.entrypoints, entrypoint_) ==
                        result.entrypoints->end()) {
                    result.backend = manifest_.policies.front().second.backend;
                }
                return result;
            }

            Mln::Sdk::Operations &expansion_container() { return expansions; }

            bool expand(const Mln::Sdk::Operation &operation) const {
                const Police police = police_for(operation.policy_path);
                const Mln::Sdk::Operation::Compatibility::Context context{
                    .operation = operation,
                    .consumers = consumer_chain(),
                    .target = target(police),
                };
                int selected_priority = std::numeric_limits<int>::min();
                std::optional<bool> result;
                for (const auto *compatibility : compatibilities) {
                    if (!compatibility->applies(context))
                        continue;
                    const std::optional<bool> candidate = compatibility->expand(context);
                    if (!candidate || compatibility->priority() < selected_priority)
                        continue;
                    if (compatibility->priority() > selected_priority || !result)
                        result = candidate;
                    else
                        result = *result || *candidate;
                    selected_priority = compatibility->priority();
                }
                return result.value_or(false);
            }

            bool fuse_producer(const Mln::Sdk::Operation &operation) const {
                const Police police = police_for(operation.policy_path);
                const Mln::Sdk::Operation::Compatibility::Context context{
                    .operation = operation,
                    .consumers = consumer_chain(),
                    .target = target(police),
                };
                int selected_priority = std::numeric_limits<int>::min();
                std::optional<bool> result;
                for (const auto *compatibility : compatibilities) {
                    if (!compatibility->applies(context))
                        continue;
                    const std::optional<bool> candidate = compatibility->fuse_producer(context);
                    if (!candidate || compatibility->priority() < selected_priority)
                        continue;
                    if (compatibility->priority() > selected_priority || !result)
                        result = candidate;
                    else
                        result = *result || *candidate;
                    selected_priority = compatibility->priority();
                }
                return result.value_or(true);
            }

            bool linearize_reduction(const Mln::Sdk::Operation &operation) const {
                const Police police = police_for(operation.policy_path);
                const Mln::Sdk::Operation::Compatibility::Context context{
                    .operation = operation,
                    .consumers = consumer_chain(),
                    .target = target(police),
                };
                int selected_priority = std::numeric_limits<int>::min();
                std::optional<bool> result;
                for (const auto *compatibility : compatibilities) {
                    if (!compatibility->applies(context))
                        continue;
                    const std::optional<bool> candidate =
                        compatibility->linearize_reduction(context);
                    if (!candidate || compatibility->priority() < selected_priority)
                        continue;
                    if (compatibility->priority() > selected_priority || !result)
                        result = candidate;
                    else
                        result = *result || *candidate;
                    selected_priority = compatibility->priority();
                }
                return result.value_or(false);
            }

            bool unroll(const Mln::Sdk::Operation &operation) const {
                const Police police = police_for(operation.policy_path);
                const Mln::Sdk::Operation::Compatibility::Context context{
                    .operation = operation,
                    .consumers = consumer_chain(),
                    .target = target(police),
                };
                return std::ranges::any_of(compatibilities, [&](const auto *compatibility) {
                    return compatibility->applies(context) && compatibility->unroll(context);
                });
            }

            int64_t contraction_tile(const Mln::Sdk::Operation &operation, int64_t extent) const {
                const Police police = police_for(operation.policy_path);
                const Mln::Sdk::Operation::Compatibility::Context context{
                    .operation = operation,
                    .consumers = consumer_chain(),
                    .target = target(police),
                };
                int selected_priority = std::numeric_limits<int>::min();
                int64_t result = extent;
                for (const auto *compatibility : compatibilities) {
                    if (!compatibility->applies(context))
                        continue;
                    const std::optional<int64_t> candidate =
                        compatibility->contraction_tile(context);
                    if (!candidate || *candidate <= 0)
                        continue;
                    const int priority = compatibility->priority();
                    if (priority > selected_priority) {
                        selected_priority = priority;
                        result = *candidate;
                    } else if (priority == selected_priority) {
                        result = std::min(result, *candidate);
                    }
                }
                return std::min(result, extent);
            }

            int64_t tile(const Mln::Sdk::Operation &operation, int64_t axis, int64_t extent) const {
                const Police police = police_for(operation.policy_path);
                const Mln::Sdk::Operation::Compatibility::Context context{
                    .operation = operation,
                    .consumers = {},
                    .target = target(police),
                };
                int64_t result = extent;
                bool selected = false;
                for (const auto *compatibility : compatibilities) {
                    if (!compatibility->applies(context))
                        continue;
                    const int64_t candidate = compatibility->tile(context, axis, extent);
                    result = selected ? std::min(result, candidate) : candidate;
                    selected = true;
                }
                return result;
            }

            const Mln::Sdk::Codec::Scheme &select_scheme(const Mln::Sdk::Weight &weight) const;

            void require_weights(const Mln::Sdk::Weight &weight, const Mln::Sdk::Codec &codec,
                                 const Mln::Sdk::Operations &decoded);

            Mln::Sdk::Tensor decode_weight(const Mln::Sdk::Functional::Const &constant);

            void seed(const Mln::Sdk::Operation *def, mlir::Value value) { cache[def] = {value}; }
            mlir::Value resolve(const Mln::Sdk::Tensor &t, mlir::OpBuilder &builder,
                                mlir::Location location) {
                std::vector<mlir::Value> values;
                if (auto it = cache.find(t.def.get());

                    it != cache.end()) {
                    values = it->second;
                } else {
                    emission_stack.push_back(t.def.get());

                    try {
                        const auto emitter = registry.find(t.def->kind);
                        if (emitter == registry.end())
                            raise(Errors::Graph::Operation::Emitter::Missing,
                                  static_cast<int>(t.def->kind), t.def->policy_path);
                        values = emitter->second->emit_results(*t.def, *this, builder, location);
                    } catch (...) {
                        emission_stack.pop_back();
                        throw;
                    }

                    emission_stack.pop_back();
                    if (values.empty())
                        raise(Errors::Graph::Operation::Emitter::Missing,
                              static_cast<int>(t.def->kind), t.def->policy_path);
                    const Police police = police_for(t.def->policy_path);
                    validate_backend(*t.def, police);
                    for (mlir::Value value : values)
                        apply_backend(value.getDefiningOp(), manifest_, police);
                    const Mln::Sdk::Operation::Compatibility::Context compatibility_context{
                        .operation = *t.def,
                        .consumers = consumer_chain(),
                        .target = target(police),
                    };
                    for (const auto *compatibility : compatibilities) {
                        if (!compatibility->applies(compatibility_context))
                            continue;
                        if (compatibility->barrier(compatibility_context) &&
                            !llvm::isa<mlir::iree_compiler::IREE::Util::OptimizationBarrierOp>(
                                values.front().getDefiningOp())) {
                            for (mlir::Value &value : values) {
                                value =
                                    mlir::iree_compiler::IREE::Util::OptimizationBarrierOp::create(
                                        builder, location, mlir::ValueRange{value})
                                        .getResult(0);
                                apply_backend(value.getDefiningOp(), manifest_, police);
                            }
                        }
                    }
                    if ((police.optimization_barrier.value_or(false) ||
                         crosses_device_boundary(police)) &&
                        !llvm::isa<mlir::iree_compiler::IREE::Util::OptimizationBarrierOp>(
                            values.front().getDefiningOp())) {
                        for (mlir::Value &value : values) {
                            auto barrier =
                                mlir::iree_compiler::IREE::Util::OptimizationBarrierOp::create(
                                    builder, location, mlir::ValueRange{value});
                            apply_backend(barrier.getOperation(), manifest_, police);
                            value = barrier.getResult(0);
                        }
                    }

                    if (cacheable(*t.def))
                        cache[t.def.get()] = values;
                }

                if (t.result >= values.size())
                    throw std::out_of_range(
                        std::format("operation '{}' produced {} results, requested {}",
                                    t.def->policy_path, values.size(), t.result));
                return values[t.result];
            }

            mlir::Value resolve_output(const Mln::Sdk::Tensor &t, mlir::OpBuilder &builder,
                                       mlir::Location location) {
                return resolve(t, builder, location);
            }
    };
    mlir::Type dtype_to_mlir_type(Mln::DType dtype, mlir::MLIRContext &ctx);
    mlir::Type tensor_type(const Mln::Sdk::Tensor &t, Mln::DType dtype, mlir::MLIRContext &ctx);
    int64_t dtype_byte_width(Mln::DType dtype);
}
