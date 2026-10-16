#pragma once

#include <algorithm>
#include <array>
#include <format>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "mln/sdk/native/symbol.hpp"
#include "mln/weight.hpp"
#include "syn/artifact.hpp"
#include "syn/compatibility.hpp"
#include "syn/deployment.hpp"
#include "syn/dynamic_shape.hpp"
#include "syn/errors/compiler.hpp"
#include "syn/functional.hpp"
#include "syn/manifest.hpp"
#include "syn/operations.hpp"
#include "syn/package.hpp"
#include "syn/report.hpp"

#include "mln/sdk/functional/input.hpp"
#include "mln/sdk/module.hpp"
#include "mln/sdk/operation.hpp"
#include "mln/sdk/orchestration.hpp"
#include "mln/sdk/tensor.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace Syn::Compiler {
    namespace Format = Mln::Weight;
}
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"

#include "llvm/Support/raw_ostream.h"

namespace Syn::Compiler {
    namespace Result {
        template <typename T> struct is_tensor_tuple : std::false_type {};

        template <typename... T>
        struct is_tensor_tuple<std::tuple<T...>>
            : std::bool_constant<(std::is_same_v<T, Mln::Sdk::Tensor> && ...)> {};

        template <typename Result, typename Function>
        void for_each_tensor(Result &result, Function &&function) {
            if constexpr (std::is_same_v<std::remove_cvref_t<Result>, Mln::Sdk::Tensor>) {
                function(result);
            } else if constexpr (is_tensor_tuple<std::remove_cvref_t<Result>>::value) {
                std::apply([&](auto &...tensor) { (function(tensor), ...); }, result);
            } else if constexpr (std::is_same_v<std::remove_cvref_t<Result>,
                                                std::vector<Mln::Sdk::Tensor>>) {
                for (Mln::Sdk::Tensor &tensor : result)
                    function(tensor);
            } else {
                static_assert(std::is_same_v<Result, void>,
                              "Module::forward must return Tensor, "
                              "std::tuple<Tensor...>, or std::vector<Tensor>");
            }
        }
    }

    class Compiler : private Mln::Sdk::Graph::Hook {
        private:
            const Manifest &manifest_;
            Compatibility::Matrix matrix_;
            std::string name_;
            std::string vm_module_name_;
            std::vector<Mln::Weight::Descriptor> weights_;
            std::vector<Mln::Weight::Descriptor> available_weights_;
            std::vector<Mln::Sdk::Backend::Resource> backend_resources_;
            std::string backend_resources_identity_;
            Compilation::Report report_;
            std::vector<Compilation::Legalization> legalizations_;
            std::vector<std::pair<std::string, Mln::Sdk::Backend>> entrypoint_backends_;

            std::vector<std::string> typed_entrypoints_;
            std::vector<uint64_t> hook_hits_;
            std::set<std::string> entrypoints_compiled_;
            bool finalized_ = false;

            Mln::Sdk::Tensor apply(std::string_view path, Mln::Sdk::Module &model,
                                   std::span<const Mln::Sdk::Tensor> inputs,
                                   Mln::Sdk::Tensor output) override;

            struct Compiled {
                    struct Graph {
                            std::vector<uint8_t> vmfb;
                            std::vector<std::string> devices;
                            std::vector<std::pair<std::string, uint32_t>> deployment;
                    };
            };
            Compiled::Graph compile_graph();

        public:
            std::unique_ptr<mlir::MLIRContext> context;
            mlir::OwningOpRef<mlir::ModuleOp> module;
            std::string graph;

            const Compilation::Report &report() const { return report_; }

            void backend_resources(std::vector<Mln::Sdk::Backend::Resource> resources,
                                   std::string identity) {
                backend_resources_ = std::move(resources);
                backend_resources_identity_ = std::move(identity);
            }

            void write_vmfb(const std::string &vmfb_path, const std::string &devices_path,
                            const std::string &weights_path = {});

            void write_graph_cache(const std::string &output_path,
                                   const std::string &requirements_path = {});
            void write_requirements(const std::string &output_path);

            void write_parameters_cache(const std::vector<std::string> &weight_paths,
                                        const std::vector<Mln::Weight::Descriptor> &requirements,
                                        const std::string &output_path,
                                        std::size_t conversion_jobs = 1,
                                        uint64_t conversion_memory_bytes = uint64_t{2} * 1024 *
                                                                           1024 * 1024);

            void write_package(const std::vector<std::string> &weight_paths,
                               const std::string &output_path, const Mln::Artifact::Plan &artifacts,
                               bool deployable = true);

            void write_package(const std::string &weights_path, const std::string &output_path,
                               const Mln::Artifact::Plan &artifacts, bool deployable = true) {
                write_package(weights_path.empty() ? std::vector<std::string>{}
                                                   : std::vector<std::string>{weights_path},
                              output_path, artifacts, deployable);
            }

            Compiler(const Manifest &manifest, std::string name,
                     std::vector<Mln::Weight::Descriptor> available_weights = {})
                : manifest_(manifest), matrix_(manifest), name_(std::move(name)),
                  available_weights_(std::move(available_weights)) {
                mlir::DialectRegistry registry;
                registry.insert<mlir::stablehlo::StablehloDialect>();
                registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect>();
                registry.insert<mlir::func::FuncDialect>();
                registry.insert<mlir::arith::ArithDialect>();
                registry.insert<mlir::linalg::LinalgDialect>();
                registry.insert<mlir::tensor::TensorDialect>();
                registry.insert<mlir::iree_compiler::IREE::Util::UtilDialect>();
                registry.insert<mlir::iree_compiler::IREE::Flow::FlowDialect>();
                registry.insert<mlir::iree_compiler::IREE::Stream::StreamDialect>();
                registry.insert<mlir::iree_compiler::IREE::HAL::HALDialect>();
                for (const Mln::Sdk::Lowering::Compatibility *compatibility :
                     manifest_.lowering_compatibilities)
                    compatibility->register_dialects(registry);
                context = std::make_unique<mlir::MLIRContext>(registry);
                context->loadAllAvailableDialects();
                mlir::OpBuilder builder(context.get());

                module = mlir::OwningOpRef<mlir::ModuleOp>(
                    mlir::ModuleOp::create(builder.getUnknownLoc()));

                validate_manifest(manifest_);
                hook_hits_.resize(manifest_.hooks.size());
            }

            void vm_module_name(std::string name) {
                vm_module_name_ = std::move(name);
                if (!vm_module_name_.empty())
                    module->setSymName(vm_module_name_);
            }

            Compiler(const Manifest &manifest, const Mln::Sdk::Orchestration &orchestration,
                     std::vector<Mln::Weight::Descriptor> available_weights = {})
                : Compiler(manifest, orchestration.name, std::move(available_weights)) {}

            template <typename C, typename... Args>
            void compile_as_impl(const std::string &logical_entrypoint,
                                 const std::string &physical_entrypoint, bool typed_abi,
                                 C &instance, Args &...inputs) {
                Operations operations(manifest_, logical_entrypoint, weights_, available_weights_,
                                      legalizations_, matrix_);
                register_emits(operations);
                mlir::OpBuilder builder(context.get());
                auto loc = builder.getUnknownLoc();
                builder.setInsertionPointToStart(module->getBody());
                Mln::DType root_dtype = *manifest_.policies[0].second.dtype;

                std::array<mlir::Type, sizeof...(Args)> arg_types{
                    tensor_type(inputs, root_dtype, *context)...};

                auto function_type = builder.getFunctionType(arg_types, {});

                if (logical_entrypoint.empty() || physical_entrypoint.empty()) {
                    raise(Errors::Compiler::Entrypoint::Specialization::Name::Missing);
                }

                const std::string logical_name = logical_entrypoint;
                const std::string physical_name = name_ + "." + physical_entrypoint;
                const std::string function_name =
                    ::Mln::Sdk::Native::sanitize_symbol(physical_name);
                auto func = mlir::func::FuncOp::create(builder, loc, function_name, function_type);

                func.setPrivate();
                const Police function_police = police_for(manifest_, instance.path());

                if (!function_police.backend) {
                    raise(Errors::Compiler::Entrypoint::Backend::Missing);
                }

                if (std::none_of(entrypoint_backends_.begin(), entrypoint_backends_.end(),
                                 [&](const auto &entry) { return entry.first == logical_name; })) {
                    entrypoint_backends_.emplace_back(logical_name, *function_police.backend);
                }

                apply_backend(func.getOperation(), manifest_, function_police);
                auto *entry = func.addEntryBlock();
                builder.setInsertionPointToStart(entry);
                std::vector<std::shared_ptr<Mln::Sdk::Operation>> input_nodes;
                std::vector<Mln::Sdk::Tensor *> input_ptrs = {&inputs...};

                for (std::size_t i = 0; i < input_ptrs.size(); i++) {
                    input_nodes.push_back(std::make_shared<Mln::Sdk::Functional::Input>());
                    input_nodes.back()->policy_path = instance.path();
                    input_ptrs[i]->def = input_nodes.back();
                    operations.seed(input_nodes.back().get(), entry->getArgument(i));
                }

                instance.bind_hook(this);
                std::array<Mln::Sdk::Tensor, sizeof...(Args)> hook_inputs{inputs...};
                auto result = instance.forward(inputs...);
                Mln::Sdk::Graph::Invocation::apply_hook(instance, hook_inputs, result);
                std::vector<mlir::Value> result_values;

                Result::for_each_tensor(result, [&](Mln::Sdk::Tensor &tensor) {
                    result_values.push_back(operations.resolve_output(tensor, builder, loc));
                });

                if (result_values.empty()) {
                    raise(Errors::Compiler::Entrypoint::Result::Missing);
                }

                mlir::func::ReturnOp::create(builder, loc, result_values);
                std::vector<mlir::Type> result_types;
                result_types.reserve(result_values.size());

                for (mlir::Value value : result_values) {
                    result_types.push_back(value.getType());
                }

                func.setFunctionType(builder.getFunctionType(arg_types, result_types));

                const bool has_dynamic_argument =
                    std::any_of(arg_types.begin(), arg_types.end(), [](mlir::Type type) {
                        return !mlir::cast<mlir::RankedTensorType>(type).hasStaticShape();
                    });

                if (typed_abi || has_dynamic_argument) {
                    func.setPublic();
                    std::string signature = logical_name;
                    signature += "\t@";
                    if (!vm_module_name_.empty()) {
                        signature += vm_module_name_;
                        signature += "::";
                    }
                    signature += physical_name;
                    for (mlir::Type type : arg_types) {
                        const auto shaped = mlir::cast<mlir::RankedTensorType>(type);
                        const mlir::Type element = shaped.getElementType();
                        std::string dtype;
                        if (element.isF32())
                            dtype = "f32";
                        else if (element.isF16())
                            dtype = "f16";
                        else if (element.isBF16())
                            dtype = "bf16";
                        else if (const auto integer = mlir::dyn_cast<mlir::IntegerType>(element)) {
                            const char prefix = integer.isUnsigned() ? 'u' : 'i';
                            dtype = std::format("{}{}", prefix, integer.getWidth());
                        } else
                            raise(Errors::Compiler::Abi::DType::Unsupported);
                        signature.push_back('\t');
                        signature += dtype;
                        signature.push_back(':');
                        for (int64_t dimension = 0; dimension < shaped.getRank(); ++dimension) {
                            if (dimension)
                                signature.push_back(',');
                            signature.append(std::format("{}", shaped.isDynamicDim(dimension)
                                                                   ? -1
                                                                   : shaped.getDimSize(dimension)));
                        }
                    }
                    signature.push_back('\n');
                    typed_entrypoints_.push_back(std::move(signature));
                    return;
                }

                auto byte_type = builder.getIntegerType(8, false);
                std::vector<mlir::Type> byte_argument_types;
                std::vector<int64_t> dynamic_argument_divisors;

                for (auto t : arg_types) {
                    auto shaped = mlir::cast<mlir::RankedTensorType>(t);
                    int64_t elems = 1;
                    int64_t dynamic_dimensions = 0;

                    for (auto d : shaped.getShape()) {
                        if (d == mlir::ShapedType::kDynamic) {
                            ++dynamic_dimensions;
                        } else {
                            elems *= d;
                        }
                    }

                    const unsigned bits = shaped.getElementType().getIntOrFloatBitWidth();

                    if (bits == 0 || bits % 8 != 0) {
                        raise(Errors::Compiler::Abi::Argument::DType::Unsupported);
                    }

                    if (dynamic_dimensions > 1)
                        raise(Errors::Compiler::Abi::Argument::Dynamic::Dimensions::Unsupported);

                    const int64_t element_bytes = static_cast<int64_t>(bits / 8);
                    dynamic_argument_divisors.push_back(elems * element_bytes);
                    byte_argument_types.push_back(mlir::RankedTensorType::get(
                        {dynamic_dimensions == 0 ? elems * element_bytes
                                                 : mlir::ShapedType::kDynamic},
                        byte_type));
                }

                std::vector<mlir::RankedTensorType> result_shaped_types;
                std::vector<int64_t> result_byte_counts;
                std::vector<mlir::Type> byte_result_types;

                for (mlir::Value value : result_values) {
                    auto shaped = mlir::cast<mlir::RankedTensorType>(value.getType());

                    if (!shaped.hasStaticShape())
                        raise(Errors::Compiler::Abi::Result::Dynamic::Shape::Unsupported);

                    int64_t elements = 1;

                    for (auto dimension : shaped.getShape()) {
                        elements *= dimension;
                    }

                    const mlir::Type element_type = shaped.getElementType();
                    const unsigned bits = element_type.getIntOrFloatBitWidth();

                    if (bits == 0 || bits % 8 != 0) {
                        raise(Errors::Compiler::Abi::Result::DType::Unsupported);
                    }

                    const int64_t bytes = elements * static_cast<int64_t>(bits / 8);
                    result_shaped_types.push_back(shaped);
                    result_byte_counts.push_back(bytes);
                    byte_result_types.push_back(mlir::RankedTensorType::get({bytes}, byte_type));
                }

                builder.setInsertionPointToEnd(module->getBody());

                auto bytes_function_type =
                    builder.getFunctionType(byte_argument_types, byte_result_types);

                auto bytes_function = mlir::func::FuncOp::create(
                    builder, loc, function_name + "_bytes", bytes_function_type);

                apply_backend(bytes_function.getOperation(), manifest_, function_police);
                auto *bytes_entry = bytes_function.addEntryBlock();
                builder.setInsertionPointToStart(bytes_entry);
                std::vector<mlir::Value> call_arguments;

                for (std::size_t i = 0; i < arg_types.size(); i++) {
                    auto arg_shaped = mlir::cast<mlir::RankedTensorType>(arg_types[i]);

                    std::vector<int64_t> shape_with_width(arg_shaped.getShape().begin(),
                                                          arg_shaped.getShape().end());

                    shape_with_width.push_back(static_cast<int64_t>(
                        arg_shaped.getElementType().getIntOrFloatBitWidth() / 8));

                    mlir::Value reshaped;
                    auto expanded_type = mlir::RankedTensorType::get(shape_with_width, byte_type);
                    if (arg_shaped.hasStaticShape()) {
                        reshaped = mlir::stablehlo::ReshapeOp::create(builder, loc, expanded_type,
                                                                      bytes_entry->getArgument(i))
                                       .getResult();
                    } else {
                        mlir::Value byte_count = mlir::tensor::DimOp::create(
                                                     builder, loc, bytes_entry->getArgument(i), 0)
                                                     .getResult();
                        mlir::Value divisor = mlir::arith::ConstantIndexOp::create(
                                                  builder, loc, dynamic_argument_divisors[i])
                                                  .getResult();
                        mlir::Value dynamic_size =
                            mlir::arith::DivUIOp::create(builder, loc, byte_count, divisor)
                                .getResult();

                        std::vector<mlir::OpFoldResult> output_shape;
                        output_shape.reserve(shape_with_width.size());
                        for (int64_t dimension : shape_with_width) {
                            output_shape.push_back(
                                dimension == mlir::ShapedType::kDynamic
                                    ? mlir::OpFoldResult(dynamic_size)
                                    : mlir::OpFoldResult(builder.getIndexAttr(dimension)));
                        }

                        std::vector<mlir::ReassociationIndices> reassociation(1);
                        for (int64_t dimension = 0; dimension < expanded_type.getRank();
                             ++dimension)
                            reassociation.front().push_back(dimension);

                        reshaped = mlir::tensor::ExpandShapeOp::create(builder, loc, expanded_type,
                                                                       bytes_entry->getArgument(i),
                                                                       reassociation, output_shape)
                                       .getResult();
                    }

                    auto typed = mlir::stablehlo::BitcastConvertOp::create(builder, loc,
                                                                           arg_types[i], reshaped)
                                     .getResult();

                    call_arguments.push_back(typed);
                }

                auto call_operation =
                    mlir::func::CallOp::create(builder, loc, func, call_arguments);
                std::vector<mlir::Value> byte_results;
                byte_results.reserve(result_values.size());

                for (std::size_t i = 0; i < result_values.size(); ++i) {
                    const auto shaped = result_shaped_types[i];

                    std::vector<int64_t> shape_with_width(shaped.getShape().begin(),
                                                          shaped.getShape().end());

                    shape_with_width.push_back(
                        static_cast<int64_t>(shaped.getElementType().getIntOrFloatBitWidth() / 8));

                    auto bytes_shaped =
                        mlir::stablehlo::BitcastConvertOp::create(
                            builder, loc, mlir::RankedTensorType::get(shape_with_width, byte_type),
                            call_operation.getResult(i))
                            .getResult();

                    byte_results.push_back(mlir::stablehlo::ReshapeOp::create(
                                               builder, loc, byte_result_types[i], bytes_shaped)
                                               .getResult());
                }

                mlir::func::ReturnOp::create(builder, loc, byte_results);
            }

            template <typename C, typename... Args> void compile(C &instance, Args &...inputs) {
                compile_as_impl(instance.path(), instance.path(), false, instance, inputs...);
            }

            template <std::size_t Arity, typename C> void compile_entrypoints(C &instance) {
                const std::string owner = instance.path();
                std::size_t matched = 0;

                for (const Entrypoint &entrypoint : manifest_.entrypoints) {
                    if (entrypoint.name.size() <= owner.size() ||
                        entrypoint.name.compare(0, owner.size(), owner) != 0 ||
                        entrypoint.name[owner.size()] != '.')
                        continue;

                    ++matched;
                    entrypoints_compiled_.insert(entrypoint.name);
                    for (std::size_t specialization_index = 0;
                         specialization_index < entrypoint.specializations.size();
                         ++specialization_index) {
                        const Specialization &specialization =
                            entrypoint.specializations[specialization_index];
                        if (specialization.inputs.size() != Arity) {
                            raise(Errors::Compiler::Entrypoint::Specialization::Arity::Mismatch,
                                  entrypoint.name, specialization.inputs.size(), Arity);
                        }

                        std::array<Mln::Sdk::Tensor, Arity> inputs;
                        for (std::size_t i = 0; i < Arity; ++i) {
                            const Shape &shape = specialization.inputs[i];
                            inputs[i] = shape.dtype
                                            ? Mln::Sdk::Tensor(shape.dimensions, *shape.dtype)
                                            : Mln::Sdk::Tensor(shape.dimensions);
                        }

                        const std::string physical = std::format(
                            "{}__specialization{}", entrypoint.name, specialization_index);
                        [&]<std::size_t... I>(std::index_sequence<I...>) {
                            compile_as_impl(entrypoint.name, physical, true, instance,
                                            inputs[I]...);
                        }(std::make_index_sequence<Arity>{});
                    }
                }

                if (matched == 0)
                    raise(Errors::Compiler::Entrypoint::Module::Unmatched, owner);
            }

            struct Plugin {
                    class Graph : public Mln::Sdk::Module {
                        public:
                            Mln::Sdk::Module *module = nullptr;

                            template <typename... Args>
                            std::vector<Mln::Sdk::Tensor> forward(const Args &...args) {
                                const std::array<Mln::Sdk::Tensor, sizeof...(Args)> inputs{args...};
                                std::vector<Mln::Sdk::Tensor> outputs;
                                outputs = module->graph(inputs);
                                if (outputs.empty())
                                    raise(Errors::Compiler::Package::Graph::Output::Missing,
                                          module->name);
                                return outputs;
                            }
                    };
            };

            template <std::size_t Arity = 1>
            void compile_definition_dispatch(const std::string &logical_entrypoint,
                                             const std::string &physical_entrypoint, bool typed_abi,
                                             Plugin::Graph &adapter,
                                             std::span<Mln::Sdk::Tensor> inputs) {
                if (inputs.size() == Arity) {
                    [&]<std::size_t... Index>(std::index_sequence<Index...>) {
                        compile_as_impl(logical_entrypoint, physical_entrypoint, typed_abi, adapter,
                                        inputs[Index]...);
                    }(std::make_index_sequence<Arity>{});
                    return;
                }
                if constexpr (Arity < 16) {
                    compile_definition_dispatch<Arity + 1>(logical_entrypoint, physical_entrypoint,
                                                           typed_abi, adapter, inputs);
                } else {
                    raise(Errors::Compiler::Abi::Arity::Unsupported, inputs.size());
                }
            }

            void compile_definition(const std::string &logical_entrypoint,
                                    const std::string &physical_entrypoint, bool typed_abi,
                                    Mln::Sdk::Module &model, std::span<Mln::Sdk::Tensor> inputs) {
                Plugin::Graph adapter;
                adapter.name = model.name;
                adapter.module = &model;

                model.bind_hook(this);
                compile_definition_dispatch(logical_entrypoint, physical_entrypoint, typed_abi,
                                            adapter, inputs);
            }

            void compile_package_entrypoints(Mln::Sdk::Module &model,
                                             std::string_view implementation = {},
                                             std::string_view graph_namespace = {}) {
                struct Restore {
                        std::string &target;
                        std::string previous;
                        ~Restore() { target = std::move(previous); }
                } restore{name_, name_};
                if (!graph_namespace.empty())
                    name_ = std::string(graph_namespace);
                const std::string owner = model.name;
                std::size_t matched = 0;

                for (const Entrypoint &entrypoint : manifest_.entrypoints) {
                    if (!implementation.empty() && !entrypoint.implementation.empty() &&
                        entrypoint.implementation != implementation)
                        continue;
                    if (entrypoint.name.size() <= owner.size() ||
                        entrypoint.name.compare(0, owner.size(), owner) != 0 ||
                        entrypoint.name[owner.size()] != '.')
                        continue;

                    ++matched;
                    entrypoints_compiled_.insert(entrypoint.name);

                    for (std::size_t index = 0; index < entrypoint.specializations.size();
                         ++index) {
                        const Specialization &specialization = entrypoint.specializations[index];
                        std::vector<Mln::Sdk::Tensor> inputs;
                        for (const Shape &shape : specialization.inputs)
                            inputs.push_back(shape.dtype
                                                 ? Mln::Sdk::Tensor(shape.dimensions, *shape.dtype)
                                                 : Mln::Sdk::Tensor(shape.dimensions));

                        compile_definition(
                            entrypoint.name,
                            std::format("{}__specialization{}", entrypoint.name, index), true,
                            model, inputs);
                    }
                }

                if (matched == 0)
                    raise(Errors::Compiler::Entrypoint::Export::Unmatched, implementation, owner);
            }

            void finalize();
    };
}
