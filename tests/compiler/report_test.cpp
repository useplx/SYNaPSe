#include "syn/report.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
    void check(std::size_t count, bool reference) {
        mlir::MLIRContext context;
        context.loadDialect<mlir::func::FuncDialect>();
        context.allowUnregisteredDialects();
        mlir::OpBuilder builder(&context);
        const auto location = builder.getUnknownLoc();
        mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
        const auto type = mlir::RankedTensorType::get({16}, builder.getF32Type());
        auto function = mlir::func::FuncOp::create(builder, location, "test",
                                                   builder.getFunctionType({type}, {type}));
        module->push_back(function);
        builder.setInsertionPointToStart(function.addEntryBlock());
        std::vector<mlir::Value> values{function.getArgument(0)};
        for (std::size_t index = 0; index < count; ++index) {
            mlir::OperationState state(location, index % 17 == 0 ? "util.global.load" : "test.op");
            const int64_t width = index % 13 == 0 ? mlir::ShapedType::kDynamic
                                                  : static_cast<int64_t>((index % 5 + 1) * 16);
            state.addTypes(mlir::RankedTensorType::get({width}, builder.getF32Type()));
            state.addOperands(values.back());
            if (index % 7 == 0)
                state.addOperands(values[index / 2]);
            values.push_back(builder.create(state)->getResult(0));
        }
        mlir::func::ReturnOp::create(builder, location, values.back());
        function.setType(builder.getFunctionType({type}, {values.back().getType()}));
        const auto report = Syn::Compiler::analyze_compilation(*module, {}, {});
        if (report.graph.operations_before_iree.value != count || report.functions.size() != 1)
            throw std::runtime_error("incorrect report operation count");
        if (!reference)
            return;

        std::uint64_t peak = 0;
        for (std::size_t position = 0; position <= count; ++position) {
            std::uint64_t live = 0;
            for (const auto &value : report.lifetimes) {
                if (!value.persistent && value.bytes && value.producer_index <= position &&
                    position <= value.last_use)
                    live += *value.bytes;
            }
            peak = std::max(peak, live);
        }
        struct Slot {
                std::uint64_t bytes;
                std::size_t last_use;
        };
        std::vector<Slot> slots;
        std::uint64_t reused = 0;
        for (const auto &value : report.lifetimes) {
            if (value.persistent || value.external || !value.bytes)
                continue;
            auto best = slots.end();
            for (auto slot = slots.begin(); slot != slots.end(); ++slot) {
                if (slot->last_use < value.producer_index && slot->bytes >= *value.bytes &&
                    (best == slots.end() || slot->bytes < best->bytes))
                    best = slot;
            }
            if (best == slots.end()) {
                slots.push_back({*value.bytes, value.last_use});
            } else {
                best->last_use = value.last_use;
                ++reused;
            }
        }
        if (report.memory.estimated_peak_live_bytes.value != peak ||
            report.optimization.reused_storage_candidates.value != reused)
            throw std::runtime_error("report differs from exhaustive lifetime analysis");
    }
}

int main() {
    check(0, true);
    check(1, true);
    check(257, true);
    check(100000, false);
}
