#include "syn/lowering.hpp"

#include "mln/error.hpp"
#include "syn/errors/compiler.hpp"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <cstdlib>
#include <format>
#include <print>
#include <ranges>
#include <unordered_set>
#include <vector>

namespace Syn::Compiler::Lowering {
    namespace {
        bool outline(mlir::ModuleOp module, std::span<mlir::Operation *const> selected,
                     std::string_view suffix) {
            llvm::SmallPtrSet<mlir::Operation *, 32> candidates(selected.begin(), selected.end());
            llvm::SmallPtrSet<mlir::Operation *, 32> consumed;
            std::size_t ordinal = 0;

            for (mlir::Operation *root : selected) {

                if (consumed.contains(root))
                    continue;
                const bool escapes = llvm::any_of(root->getResults(), [&](mlir::Value value) {
                    return llvm::any_of(value.getUsers(), [&](mlir::Operation *user) {
                        return !candidates.contains(user);
                    });
                });
                if (!escapes)
                    continue;

                auto parent = root->getParentOfType<mlir::func::FuncOp>();
                if (!parent)
                    return false;
                llvm::SmallPtrSet<mlir::Operation *, 32> component;
                std::vector<mlir::Operation *> pending{root};
                while (!pending.empty()) {
                    mlir::Operation *operation = pending.back();
                    pending.pop_back();
                    if (!candidates.contains(operation) || !component.insert(operation).second)
                        continue;
                    if (operation->getParentOfType<mlir::func::FuncOp>() != parent)
                        return false;
                    for (mlir::Value operand : operation->getOperands()) {
                        if (mlir::Operation *producer = operand.getDefiningOp();
                            producer && candidates.contains(producer)) {
                            pending.push_back(producer);
                        }
                    }
                }

                std::vector<mlir::Operation *> operations;
                parent.walk([&](mlir::Operation *operation) {
                    if (component.contains(operation))
                        operations.push_back(operation);
                });
                if (operations.empty())
                    return false;

                llvm::SetVector<mlir::Value> inputs;
                llvm::SetVector<mlir::Value> outputs;
                for (mlir::Operation *operation : operations) {
                    for (mlir::Value operand : operation->getOperands()) {
                        mlir::Operation *producer = operand.getDefiningOp();
                        if (!producer || !component.contains(producer))
                            inputs.insert(operand);
                    }
                    for (mlir::Value result : operation->getResults()) {
                        if (llvm::any_of(result.getUsers(), [&](mlir::Operation *user) {
                                return !component.contains(user);
                            })) {
                            outputs.insert(result);
                        }
                    }
                }
                if (outputs.empty())
                    return false;

                std::vector<mlir::Type> input_types;
                std::vector<mlir::Type> output_types;
                for (mlir::Value input : inputs)
                    input_types.push_back(input.getType());
                for (mlir::Value output : outputs)
                    output_types.push_back(output.getType());

                mlir::OpBuilder module_builder(module.getBodyRegion());
                module_builder.setInsertionPoint(parent);
                const std::string name = parent.getSymName().str() + "_" + std::string(suffix) +
                                         "_" + std::to_string(ordinal++);
                auto function = mlir::iree_compiler::IREE::Util::FuncOp::create(
                    module_builder, root->getLoc(), name,
                    module_builder.getFunctionType(input_types, output_types));
                function.setVisibility(mlir::SymbolTable::Visibility::Private);
                function.setInliningPolicyAttr(
                    module_builder.getAttr<mlir::iree_compiler::IREE::Util::InlineNeverAttr>());
                if (mlir::Attribute affinity = root->getAttr("stream.affinity"))
                    function->setAttr("stream.affinity", affinity);

                mlir::Block *body = function.addEntryBlock();
                mlir::OpBuilder body_builder = mlir::OpBuilder::atBlockBegin(body);
                mlir::IRMapping mapping;
                for (auto [input, argument] : llvm::zip(inputs, body->getArguments()))
                    mapping.map(input, argument);
                for (mlir::Operation *operation : operations)
                    body_builder.clone(*operation, mapping);

                std::vector<mlir::Value> returned;
                returned.reserve(outputs.size());
                for (mlir::Value output : outputs)
                    returned.push_back(mapping.lookup(output));
                mlir::iree_compiler::IREE::Util::ReturnOp::create(body_builder, root->getLoc(),
                                                                  returned);

                mlir::OpBuilder call_builder(operations.front());
                auto call = mlir::iree_compiler::IREE::Util::CallOp::create(
                    call_builder, root->getLoc(), function, inputs.getArrayRef());
                for (auto [output, replacement] : llvm::zip(outputs, call.getResults())) {
                    for (mlir::OpOperand &use : llvm::make_early_inc_range(output.getUses())) {
                        if (!component.contains(use.getOwner()))
                            use.set(replacement);
                    }
                }
                for (mlir::Operation *operation : llvm::reverse(operations)) {
                    consumed.insert(operation);
                    operation->erase();
                }
            }
            return consumed.size() == candidates.size();
        }
    }

    std::string apply(mlir::ModuleOp module, const Mln::Sdk::Backend &backend,
                      std::span<Mln::Sdk::Lowering::Compatibility *const> compatibilities,
                      std::string_view source) {
        std::string representation(source);
        std::unordered_set<std::string> visited{representation};
        std::vector<mlir::Operation *> operations;
        const std::string assignment = std::format("{}:{}", backend.device, backend.ordinal);
        module.walk([&](mlir::Operation *operation) {
            const auto selected = operation->getAttrOfType<mlir::StringAttr>("myelin.backend");
            if (selected && selected.getValue() == assignment)
                operations.push_back(operation);
        });

        if (const char *trace = std::getenv("MLN_TRACE_LOWERING"); trace && *trace) {
            std::println(stderr, "lowering backend {}: {} selected operations", assignment,
                         operations.size());
        }

        if (operations.empty())
            return representation;

        while (true) {
            std::vector<Mln::Sdk::Lowering::Compatibility *> candidates;
            for (Mln::Sdk::Lowering::Compatibility *compatibility : compatibilities) {
                if (compatibility->source() == representation && compatibility->applies(backend))
                    candidates.push_back(compatibility);
            }
            if (candidates.empty())
                return representation;

            const int priority = std::ranges::min(candidates, {}, [](const auto *compatibility) {
                                     return compatibility->priority();
                                 })->priority();
            const auto selected =
                std::ranges::find_if(candidates, [priority](const auto *compatibility) {
                    return compatibility->priority() == priority;
                });
            const std::size_t matches = static_cast<std::size_t>(
                std::ranges::count_if(candidates, [priority](const auto *compatibility) {
                    return compatibility->priority() == priority;
                }));
            if (matches != 1)
                raise(Errors::Compiler::Lowering::Chain::Ambiguous, priority, representation,
                      backend.device);

            Mln::Sdk::Lowering::Compatibility &compatibility = **selected;
            const std::string target(compatibility.target());
            if (!visited.insert(target).second)
                raise(Errors::Compiler::Lowering::Chain::Cyclic, target, backend.device);

            const Mln::Sdk::Lowering::Compatibility::Context context{
                .backend = backend,
                .source = representation,
                .target = target,
                .operations = operations,
                .outline = outline,
            };
            if (!compatibility.lower(module, context))
                raise(Errors::Compiler::Lowering::Failed, representation, target, backend.device);
            representation = target;
        }
    }
}
