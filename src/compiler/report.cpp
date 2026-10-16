#include "syn/report.hpp"

#include <iree/compiler/Dialect/Util/IR/UtilOps.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>

#include <llvm/ADT/DenseMap.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace Syn::Compiler {
    namespace {
        std::optional<uint64_t> tensor_bytes(mlir::Type type) {
            auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
            if (!shaped || !shaped.hasStaticShape())
                return std::nullopt;

            const mlir::Type element = shaped.getElementType();
            if (!element.isIntOrFloat())
                return std::nullopt;
            const uint64_t elements = static_cast<uint64_t>(shaped.getNumElements());
            const uint64_t bits = element.getIntOrFloatBitWidth();
            if (bits != 0 && elements > std::numeric_limits<uint64_t>::max() / bits)
                return std::nullopt;
            return (elements * bits + 7) / 8;
        }

        bool is_view_candidate(mlir::Operation &op) {
            const std::string_view name = op.getName().getStringRef();

            return name == "stablehlo.reshape" || name == "stablehlo.bitcast_convert";
        }

        mlir::Operation *top_level_owner(mlir::Operation *operation, mlir::Block &block) {
            while (operation && operation->getBlock() != &block)
                operation = operation->getParentOp();
            return operation;
        }

        const char *quality_name(Metric::Quality quality) {
            switch (quality) {
            case Metric::Quality::Exact:
                return "exact";
            case Metric::Quality::Estimated:
                return "estimated";
            case Metric::Quality::Unknown:
                return "unknown";
            }
            return "unknown";
        }

        void print_metric(std::ostringstream &out, std::string_view name, const Metric &metric) {
            out << "  " << name << ": ";
            if (metric.value)
                out << *metric.value;
            else
                out << "unknown";
            out << " [" << quality_name(metric.quality) << "]\n";
        }

        void print_ratio(std::ostringstream &out, std::string_view name,
                         const Metric::Ratio &metric) {
            out << "  " << name << ": ";
            if (metric.value)
                out << *metric.value;
            else
                out << "unknown";
            out << " [" << quality_name(metric.quality) << "]\n";
        }

        std::string affinity_name(mlir::func::FuncOp function) {
            const mlir::Attribute affinity = function->getAttr("myelin.backend");
            if (!affinity)
                return {};
            std::string text;
            llvm::raw_string_ostream stream(text);
            affinity.print(stream);
            return text;
        }
    }

    Compilation::Report
    analyze_compilation(mlir::ModuleOp module,
                        const std::vector<Mln::Weight::Descriptor> &required_weights,
                        const std::vector<Mln::Weight::Descriptor> &available_weights) {
        Compilation::Report report;
        report.graph.parameters = Metric::exact(required_weights.size());

        std::unordered_map<std::string, const Mln::Weight::Descriptor *> available;
        uint64_t required_source_bytes = 0;
        uint64_t eliminated_bytes = 0;
        uint64_t matched_requirements = 0;
        for (const auto &descriptor : available_weights)
            available.emplace(descriptor.name, &descriptor);
        for (const auto &requirement : required_weights) {
            const auto found = available.find(requirement.name);
            if (found == available.end())
                continue;
            required_source_bytes += found->second->byte_length;
            ++matched_requirements;
        }
        if (!available_weights.empty() && matched_requirements == required_weights.size()) {
            for (const auto &descriptor : available_weights)
                if (std::none_of(
                        required_weights.begin(), required_weights.end(),
                        [&](const auto &required) { return required.name == descriptor.name; }))
                    eliminated_bytes += descriptor.byte_length;
            report.quantization.bytes_before = Metric::exact(required_source_bytes);
            report.optimization.eliminated_parameters =
                Metric::exact(available_weights.size() - matched_requirements);
            report.optimization.eliminated_parameter_bytes = Metric::exact(eliminated_bytes);
        }

        uint64_t function_count = 0;
        uint64_t operation_count = 0;
        uint64_t constant_count = 0;
        uint64_t temporary_value_bytes = 0;
        uint64_t peak_live_bytes = 0;
        uint64_t view_candidates = 0;
        uint64_t inplace_candidates = 0;
        uint64_t reuse_candidates = 0;

        module.walk([&](mlir::func::FuncOp function) {
            if (function.isDeclaration() || function.getName().ends_with("_bytes"))
                return;
            ++function_count;
            if (!function.getBody().hasOneBlock())
                return;
            mlir::Block &block = function.getBody().front();

            std::vector<mlir::Operation *> operations;
            std::unordered_map<mlir::Operation *, std::size_t> positions;
            for (mlir::Operation &operation : block) {
                if (mlir::isa<mlir::func::ReturnOp>(operation))
                    continue;
                positions.emplace(&operation, operations.size());
                operations.push_back(&operation);
                ++operation_count;
                if (operation.getName().getStringRef() == "stablehlo.constant")
                    ++constant_count;
            }

            const std::size_t return_position = operations.size();
            auto return_op = mlir::dyn_cast<mlir::func::ReturnOp>(block.back());
            std::vector<std::size_t> function_lifetime_indices;
            llvm::DenseMap<mlir::Value, std::size_t> lifetime_by_value;

            auto append_value = [&](mlir::Value value, std::string producer,
                                    std::size_t producer_position, bool persistent, bool external,
                                    bool view_candidate) {
                Value::Lifetime lifetime;
                lifetime.function = function.getName().str();
                lifetime.producer = std::move(producer);
                lifetime.producer_index = producer_position;
                lifetime.first_use = producer_position;
                lifetime.last_use = producer_position;
                lifetime.bytes = tensor_bytes(value.getType());
                lifetime.persistent = persistent;
                lifetime.external = external;
                lifetime.view_candidate = view_candidate;

                bool saw_use = false;
                for (mlir::OpOperand &use : value.getUses()) {
                    mlir::Operation *owner = top_level_owner(use.getOwner(), block);
                    std::size_t position = return_position;
                    if (owner && !mlir::isa<mlir::func::ReturnOp>(owner)) {
                        const auto found = positions.find(owner);
                        if (found == positions.end())
                            continue;
                        position = found->second;
                    } else if (!owner) {
                        continue;
                    }
                    if (!saw_use)
                        lifetime.first_use = position;
                    lifetime.first_use = std::min(lifetime.first_use, position);
                    lifetime.last_use = std::max(lifetime.last_use, position);
                    saw_use = true;
                }
                if (return_op &&
                    std::any_of(return_op.getOperands().begin(), return_op.getOperands().end(),
                                [&](mlir::Value result) { return result == value; })) {
                    lifetime.external = true;
                    lifetime.last_use = std::max(lifetime.last_use, return_position);
                }

                lifetime_by_value[value] = report.lifetimes.size();
                function_lifetime_indices.push_back(report.lifetimes.size());
                report.lifetimes.push_back(std::move(lifetime));
            };

            for (mlir::BlockArgument argument : block.getArguments())
                append_value(argument, "argument", 0, false, true, false);

            for (std::size_t position = 0; position < operations.size(); ++position) {
                mlir::Operation &operation = *operations[position];
                const bool persistent = operation.getName().getStringRef() == "util.global.load";
                const bool view = is_view_candidate(operation);
                if (view)
                    ++view_candidates;
                for (mlir::Value result : operation.getResults())
                    if (mlir::isa<mlir::ShapedType>(result.getType()))
                        append_value(result, operation.getName().getStringRef().str(), position,
                                     persistent, persistent, view);
            }

            std::vector<uint64_t> allocations(return_position + 1);
            std::vector<uint64_t> retirements(return_position + 2);
            for (std::size_t index : function_lifetime_indices) {
                const Value::Lifetime &value = report.lifetimes[index];
                if (value.persistent || !value.bytes)
                    continue;
                allocations[value.producer_index] += *value.bytes;
                retirements[value.last_use + 1] += *value.bytes;
            }
            uint64_t function_peak = 0;
            uint64_t live = 0;
            for (std::size_t position = 0; position <= return_position; ++position) {
                live -= retirements[position];
                live += allocations[position];
                function_peak = std::max(function_peak, live);
            }
            peak_live_bytes = std::max(peak_live_bytes, function_peak);

            using Retirement = std::pair<std::size_t, uint64_t>;
            std::priority_queue<Retirement, std::vector<Retirement>, std::greater<>> active;
            std::multiset<uint64_t> available_slots;
            uint64_t function_temporary_bytes = 0;
            for (std::size_t index : function_lifetime_indices) {
                const Value::Lifetime &value = report.lifetimes[index];
                if (value.persistent || value.external || !value.bytes)
                    continue;
                temporary_value_bytes += *value.bytes;
                function_temporary_bytes += *value.bytes;

                while (!active.empty() && active.top().first < value.producer_index) {
                    available_slots.insert(active.top().second);
                    active.pop();
                }
                const auto reusable = available_slots.lower_bound(*value.bytes);
                uint64_t capacity = *value.bytes;
                if (reusable != available_slots.end()) {
                    capacity = *reusable;
                    available_slots.erase(reusable);
                    ++reuse_candidates;
                }
                active.emplace(value.last_use, capacity);
            }

            for (std::size_t position = 0; position < operations.size(); ++position) {
                mlir::Operation &operation = *operations[position];
                if (!operation.hasTrait<mlir::OpTrait::Elementwise>() ||
                    operation.getNumResults() != 1)
                    continue;
                const auto result_bytes = tensor_bytes(operation.getResult(0).getType());
                if (!result_bytes)
                    continue;
                for (mlir::Value operand : operation.getOperands()) {
                    const auto operand_bytes = tensor_bytes(operand.getType());
                    if (!operand_bytes || *operand_bytes != *result_bytes)
                        continue;
                    const auto found = lifetime_by_value.find(operand);
                    if (found != lifetime_by_value.end()) {
                        const Value::Lifetime &value = report.lifetimes[found->second];
                        if (value.last_use != position || value.persistent || value.external)
                            continue;
                        ++inplace_candidates;
                        break;
                    }
                }
            }

            report.functions.push_back(Function::Memory::Estimate{
                .function = function.getName().str(),
                .backend_affinity = affinity_name(function),
                .temporary_value_bytes = Metric::exact(function_temporary_bytes),
                .estimated_peak_live_bytes = Metric::estimated(function_peak),
            });
        });

        report.graph.modules = Metric::exact(function_count);
        report.graph.operations_before_iree = Metric::exact(operation_count);
        report.graph.operations_after_iree = Metric::unknown();
        report.graph.constants = Metric::exact(constant_count);
        report.memory.temporary_value_bytes = Metric::exact(temporary_value_bytes);
        report.memory.estimated_peak_live_bytes = Metric::estimated(peak_live_bytes);
        report.optimization.view_candidates = Metric::estimated(view_candidates);
        report.optimization.eliminated_values = Metric::unknown();
        report.optimization.eliminated_operations = Metric::unknown();
        report.optimization.inplace_candidates = Metric::estimated(inplace_candidates);
        report.optimization.reused_storage_candidates = Metric::estimated(reuse_candidates);
        report.partitioning.partitions = Metric::unknown();
        report.partitioning.estimated_transfers = Metric::unknown();
        return report;
    }

    std::string Compilation::Report::text() const {
        std::ostringstream out;
        out << "Model storage\n";
        print_metric(out, "parameter bytes", storage.parameter_bytes);
        print_metric(out, "executable bytes", storage.executable_bytes);
        print_metric(out, "metadata bytes", storage.metadata_bytes);
        print_metric(out, "total artifact bytes", storage.total_artifact_bytes);
        out << "Memory\n";
        print_metric(out, "persistent bytes", memory.persistent_bytes);
        print_metric(out, "temporary value bytes", memory.temporary_value_bytes);
        print_metric(out, "peak live bytes", memory.estimated_peak_live_bytes);
        out << "Functions\n";
        if (functions.empty()) {
            out << "  unknown [unknown]\n";
        } else {
            for (const Function::Memory::Estimate &function : functions) {
                out << "  " << function.function << "\n";
                out << "    backend affinity: "
                    << (function.backend_affinity.empty() ? "unknown [unknown]"
                                                          : function.backend_affinity + " [exact]")
                    << "\n";
                out << "    temporary value bytes: ";
                if (function.temporary_value_bytes.value)
                    out << *function.temporary_value_bytes.value;
                else
                    out << "unknown";
                out << " [" << quality_name(function.temporary_value_bytes.quality) << "]\n";
                out << "    peak live bytes: ";
                if (function.estimated_peak_live_bytes.value)
                    out << *function.estimated_peak_live_bytes.value;
                else
                    out << "unknown";
                out << " [" << quality_name(function.estimated_peak_live_bytes.quality) << "]\n";
            }
        }
        out << "Graph\n";
        print_metric(out, "modules", graph.modules);
        print_metric(out, "operations before IREE", graph.operations_before_iree);
        print_metric(out, "operations after IREE", graph.operations_after_iree);
        print_metric(out, "constants", graph.constants);
        print_metric(out, "parameters", graph.parameters);
        out << "Quantization\n";
        print_metric(out, "bytes before", quantization.bytes_before);
        print_metric(out, "bytes after", quantization.bytes_after);
        print_ratio(out, "compression ratio", quantization.compression_ratio);
        out << "Packaging\n";
        print_metric(out, "source bytes", packaging.source_bytes);
        print_metric(out, "output bytes", packaging.output_bytes);
        print_metric(out, "source parameters", packaging.source_parameters);
        print_metric(out, "output parameters", packaging.output_parameters);
        print_metric(out, "passthrough parameters", packaging.passthrough_parameters);
        print_metric(out, "converted parameters", packaging.converted_parameters);
        out << "Optimization\n";
        print_metric(out, "eliminated values", optimization.eliminated_values);
        print_metric(out, "eliminated operations", optimization.eliminated_operations);
        print_metric(out, "eliminated parameters", optimization.eliminated_parameters);
        print_metric(out, "eliminated parameter bytes", optimization.eliminated_parameter_bytes);
        print_metric(out, "view candidates", optimization.view_candidates);
        print_metric(out, "in-place candidates", optimization.inplace_candidates);
        print_metric(out, "reused storage candidates", optimization.reused_storage_candidates);
        out << "Partitioning\n";
        print_metric(out, "partitions", partitioning.partitions);
        print_metric(out, "estimated transfers", partitioning.estimated_transfers);
        out << "  target backends: ";
        if (partitioning.target_backends.empty()) {
            out << "unknown [unknown]\n";
        } else {
            for (std::size_t i = 0; i < partitioning.target_backends.size(); ++i) {
                if (i)
                    out << ", ";
                out << partitioning.target_backends[i];
            }
            out << " [exact]\n";
        }
        out << "Backend legalization\n";
        if (legalizations.empty()) {
            out << "  none [exact]\n";
        } else {
            for (const Compilation::Legalization &item : legalizations)
                out << "  " << item.backend << " operation " << item.operation << " at "
                    << item.path << ": " << item.outcome << " [exact]\n";
        }
        out << "Compatibility\n";
        if (compatibilities.empty()) {
            out << "  none [exact]\n";
        } else {
            for (const Compatibility::Entry &item : compatibilities) {
                out << "  "
                    << (item.subject == Compatibility::Subject::Operation ? "operation " : "codec ")
                    << item.name << " on " << item.backend;
                if (!item.properties.empty()) {
                    out << " (";
                    for (std::size_t i = 0; i < item.properties.size(); ++i) {
                        if (i)
                            out << ", ";
                        out << item.properties[i].first << '=' << item.properties[i].second;
                    }
                    out << ')';
                }
                out << ": "
                    << (item.covered == item.uses ? "implemented"
                        : item.covered == 0       ? "not explicit"
                                                  : "partially implemented")
                    << ", " << item.covered << '/' << item.uses << " covered, first at "
                    << item.path << " [exact]\n";
            }
        }
        return out.str();
    }
}
