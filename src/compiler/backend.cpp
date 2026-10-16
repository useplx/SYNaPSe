#include "syn/backend.hpp"

#include "mln/error.hpp"
#include "syn/errors/compiler.hpp"

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace Syn::Compiler {
    namespace {
        namespace Hal = mlir::iree_compiler::IREE::HAL;
        std::span<const Mln::Sdk::Backend::Resource> active_resources;
        std::mutex active_resources_mutex;
        std::condition_variable active_resources_changed;
        std::string active_resources_identity;
        std::size_t active_resources_users = 0;
        bool active_resources_closing = false;

        std::string normalized_resource(std::string_view name) {
            std::string result;
            result.reserve(name.size());
            for (const char character : name)
                result.push_back(std::isalnum(static_cast<unsigned char>(character)) ? character
                                                                                     : '_');
            return result;
        }

        struct ResourceBinding {
                std::string entrypoint;
                std::uint32_t ordinal = 0;
                std::string_view resource;
        };

        std::optional<std::string_view>
        global_origin(mlir::Value value, std::unordered_set<mlir::Operation *> &visited) {
            mlir::Operation *definition = value.getDefiningOp();
            if (!definition || !visited.insert(definition).second)
                return std::nullopt;
            if (auto load =
                    mlir::dyn_cast<mlir::iree_compiler::IREE::Util::GlobalLoadOp>(definition))
                return std::string_view(load.getGlobal());
            for (mlir::Value operand : definition->getOperands())
                if (auto origin = global_origin(operand, visited))
                    return origin;
            return std::nullopt;
        }

        std::vector<ResourceBinding> resource_bindings(Hal::ExecutableVariantOp executable) {
            std::vector<ResourceBinding> result;
            mlir::ModuleOp root = executable->getParentOfType<mlir::ModuleOp>();
            if (!root)
                return result;
            root.walk([&](Hal::ExecutableLookupFunctionOp lookup) {
                const llvm::StringRef leaf = lookup.getEntryPoint().getLeafReference();
                for (mlir::Operation *user : lookup.getResult().getUsers()) {
                    auto dispatch = mlir::dyn_cast<Hal::CommandBufferDispatchOp>(user);
                    if (!dispatch)
                        continue;
                    std::uint32_t ordinal = 0;
                    for (mlir::Value buffer : dispatch.getBindingBuffers()) {
                        std::unordered_set<mlir::Operation *> visited;
                        if (const auto global = global_origin(buffer, visited)) {
                            for (const auto &resource : active_resources) {
                                const std::string marker =
                                    "__parameter_" + normalized_resource(resource.name) + "_tensor";
                                if (global->starts_with(marker)) {
                                    result.push_back({leaf.str(), ordinal, resource.name});
                                    break;
                                }
                            }
                        }
                        ++ordinal;
                    }
                }
            });
            return result;
        }

        Mln::Sdk::Backend::Compilation::Element element(mlir::Type type) {
            using Element = Mln::Sdk::Backend::Compilation::Element;
            if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
                integer && integer.getWidth() == 4)
                return integer.isUnsigned() ? Element::U4 : Element::I4;
            if (type.isInteger(8))
                return Element::I8;
            if (type.isInteger(16))
                return Element::I16;
            if (type.isInteger(32))
                return Element::I32;
            if (type.isF16())
                return Element::F16;
            if (type.isBF16())
                return Element::BF16;
            return Element::F32;
        }

        struct ExecutableDescription {
                std::vector<Mln::Sdk::Backend::Compilation::Entrypoint> entrypoints;
                std::vector<Mln::Sdk::Backend::Compilation::Binding> bindings;
        };

        ExecutableDescription describe(Hal::ExecutableVariantOp executable) {
            ExecutableDescription result;
            const auto resources = resource_bindings(executable);
            mlir::ModuleOp module = executable.getInnerModule();
            for (Hal::ExecutableExportOp exported : executable.getOps<Hal::ExecutableExportOp>()) {
                const std::string name = exported.getSymName().str();
                auto function = module.lookupSymbol<mlir::func::FuncOp>(name);
                if (!function)
                    continue;

                Mln::Sdk::Backend::Compilation::Entrypoint entrypoint{.name = name};
                function.walk([&](mlir::linalg::LinalgOp operation) {
                    if (operation.getNumReductionLoops() != 0)
                        ++entrypoint.contractions;
                });
                result.entrypoints.push_back(std::move(entrypoint));

                function.walk([&](Hal::InterfaceBindingSubspanOp operation) {
                    const auto dispatch =
                        mlir::dyn_cast<mlir::iree_compiler::IREE::TensorExt::DispatchTensorType>(
                            operation.getResult().getType());
                    if (!dispatch)
                        return;
                    const auto tensor =
                        mlir::dyn_cast<mlir::RankedTensorType>(dispatch.getBoundType());
                    if (!tensor)
                        return;
                    Mln::Sdk::Backend::Compilation::Binding binding{
                        .entrypoint = name,
                        .ordinal =
                            static_cast<std::uint32_t>(operation.getBinding().getZExtValue()),
                        .shape = {tensor.getShape().begin(), tensor.getShape().end()},
                        .element = element(tensor.getElementType()),
                    };
                    if (mlir::Value offset = operation.getByteOffset()) {
                        llvm::APInt value;
                        if (mlir::matchPattern(offset, mlir::m_ConstantInt(&value)) &&
                            !value.isNegative())
                            binding.offset = value.getZExtValue();
                    }
                    std::uint64_t elements = 1;
                    for (const std::int64_t dimension : binding.shape)
                        elements *= static_cast<std::uint64_t>(dimension);
                    const std::uint64_t bits = tensor.getElementTypeBitWidth();
                    binding.length = (elements * bits + 7) / 8;
                    for (mlir::Operation *user : operation.getResult().getUsers()) {
                        binding.read |=
                            mlir::isa<mlir::iree_compiler::IREE::TensorExt::DispatchTensorLoadOp>(
                                user);
                        binding.written |=
                            mlir::isa<mlir::iree_compiler::IREE::TensorExt::DispatchTensorStoreOp>(
                                user);
                    }
                    const auto related = std::ranges::find_if(resources, [&](const auto &value) {
                        return value.entrypoint == binding.entrypoint &&
                               value.ordinal == binding.ordinal;
                    });
                    if (related != resources.end())
                        binding.resource = related->resource;
                    result.bindings.push_back(std::move(binding));
                });
            }
            std::ranges::sort(result.bindings, [](const auto &left, const auto &right) {
                return std::tie(left.entrypoint, left.ordinal, left.offset) <
                       std::tie(right.entrypoint, right.ordinal, right.offset);
            });
            return result;
        }

        class MaterializeExternalWorkgroups final
            : public mlir::PassWrapper<MaterializeExternalWorkgroups,
                                       mlir::OperationPass<Hal::ExecutableVariantOp>> {
            public:
                MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeExternalWorkgroups)

                void runOnOperation() final {
                    getOperation().walk(
                        [](mlir::iree_compiler::IREE::TensorExt::DispatchWorkgroupCountFromSliceOp
                               operation) {
                            mlir::OpBuilder builder(operation);
                            mlir::Value one = mlir::arith::ConstantIndexOp::create(
                                builder, operation.getLoc(), 1);
                            operation.replaceAllUsesWith(mlir::ValueRange{one, one, one});
                            operation.erase();
                        });
                }
        };

        class Target final : public Hal::TargetBackend {
            public:
                explicit Target(Mln::Sdk::Backend::Compatibility &compatibility)
                    : compatibility_(compatibility) {}

                std::string getLegacyDefaultDeviceID() const final {
                    return std::string(compatibility_.name());
                }

                void getDefaultExecutableTargets(
                    mlir::MLIRContext *context, llvm::StringRef, mlir::DictionaryAttr,
                    llvm::SmallVectorImpl<Hal::ExecutableTargetAttr> &targets) const final {
                    if (const char *trace = std::getenv("MLN_TRACE_BACKEND"); trace && *trace)
                        std::println(stderr, "backend {}: resolving executable target",
                                     compatibility_.name());
                    mlir::Builder builder(context);
                    targets.push_back(Hal::ExecutableTargetAttr::get(
                        context, builder.getStringAttr(compatibility_.name()),
                        builder.getStringAttr(compatibility_.format()),
                        builder.getDictionaryAttr({})));
                }

                SupportedTypes getSupportedTypes(mlir::MLIRContext *context) const final {
                    SupportedTypes types;
                    mlir::Builder builder(context);
                    types.addScalarType(builder.getI8Type());
                    types.addScalarType(builder.getI16Type());
                    types.addScalarType(builder.getI32Type());
                    types.addScalarType(builder.getI64Type());
                    types.addScalarType(builder.getF16Type());
                    types.addScalarType(builder.getBF16Type());
                    types.addScalarType(builder.getF32Type());
                    types.addElementType(builder.getI8Type());
                    types.addElementType(builder.getI16Type());
                    types.addElementType(builder.getI32Type());
                    types.addElementType(builder.getF16Type());
                    types.addElementType(builder.getBF16Type());
                    types.addElementType(builder.getF32Type());
                    return types;
                }

                void getDependentDialects(mlir::DialectRegistry &registry) const final {
                    (void)registry;
                }

                void buildTranslationPassPipeline(Hal::ExecutableTargetAttr,
                                                  mlir::OpPassManager &pass_manager) final {
                    pass_manager.addPass(std::make_unique<MaterializeExternalWorkgroups>());
                }

                mlir::LogicalResult serializeExecutable(const SerializationOptions &,
                                                        Hal::ExecutableVariantOp executable,
                                                        mlir::OpBuilder &builder) final {
                    const bool trace = [] {
                        const char *value = std::getenv("MLN_TRACE_BACKEND");
                        return value && *value;
                    }();
                    if (trace)
                        std::println(stderr, "backend {}: serializing variant {}",
                                     compatibility_.name(), executable.getSymName());
                    if (const char *path = std::getenv("MLN_DUMP_BACKEND_VARIANTS");
                        path && *path) {
                        std::string text;
                        llvm::raw_string_ostream stream(text);
                        executable.print(stream);
                        stream.flush();
                        std::ofstream output(path, std::ios::app);
                        output << "// backend: " << compatibility_.name() << '\n' << text << "\n\n";
                    }
                    const ExecutableDescription description = describe(executable);
                    if (trace)
                        std::println(stderr, "backend {}: {} entrypoints, {} bindings",
                                     compatibility_.name(), description.entrypoints.size(),
                                     description.bindings.size());
                    const Mln::Sdk::Backend::Compilation compilation{
                        .resources = active_resources,
                        .entrypoints = description.entrypoints,
                        .bindings = description.bindings,
                    };
                    auto serialized = compatibility_.serialize(compilation);
                    if (!serialized)
                        return executable.emitOpError() << serialized.error();
                    if (trace)
                        std::println(stderr, "backend {}: serialized {} bytes",
                                     compatibility_.name(), serialized->size());

                    llvm::SmallVector<char> bytes;
                    bytes.reserve(serialized->size());
                    for (const std::byte value : *serialized)
                        bytes.push_back(static_cast<char>(value));
                    const auto type = mlir::VectorType::get({static_cast<int64_t>(bytes.size())},
                                                            builder.getI8Type());
                    const auto data = mlir::DenseIntElementsAttr::get(type, std::move(bytes));
                    Hal::ExecutableBinaryOp::create(builder, executable.getLoc(),
                                                    executable.getSymName(),
                                                    executable.getTarget().getFormat(), data);
                    return mlir::success();
                }

            private:
                Mln::Sdk::Backend::Compatibility &compatibility_;
        };

        class Device final : public Hal::TargetDevice {
            public:
                explicit Device(Mln::Sdk::Backend::Compatibility &compatibility)
                    : compatibility_(compatibility) {}

                Hal::DeviceTargetAttr
                getDefaultDeviceTarget(mlir::MLIRContext *context,
                                       const Hal::TargetRegistry &registry) const final {
                    if (const char *trace = std::getenv("MLN_TRACE_BACKEND"); trace && *trace)
                        std::println(stderr, "backend {}: resolving device target",
                                     compatibility_.name());
                    mlir::Builder builder(context);
                    auto target = registry.getTargetBackend(compatibility_.name());
                    llvm::SmallVector<Hal::ExecutableTargetAttr> executable_targets;
                    target->getDefaultExecutableTargets(context, compatibility_.name(),
                                                        builder.getDictionaryAttr({}),
                                                        executable_targets);
                    return Hal::DeviceTargetAttr::get(
                        context, builder.getStringAttr(compatibility_.name()),
                        builder.getDictionaryAttr({}), executable_targets);
                }

                mlir::Value buildDeviceTargetMatch(mlir::Location location, mlir::Value device,
                                                   Hal::DeviceTargetAttr target,
                                                   mlir::OpBuilder &builder) const final {
                    const std::string pattern = std::string(compatibility_.name()) + "*";
                    return Hal::DeviceTargetAttr::buildDeviceIDAndExecutableFormatsMatch(
                        location, device, pattern, target.getExecutableTargets(), builder);
                }

            private:
                Mln::Sdk::Backend::Compatibility &compatibility_;
        };
    }

    BackendResources::BackendResources(std::string_view identity,
                                       std::span<const Mln::Sdk::Backend::Resource> resources) {
        std::unique_lock lock(active_resources_mutex);
        active_resources_changed.wait(lock, [&] {
            return active_resources_users == 0 ||
                   (!active_resources_closing && active_resources_identity == identity);
        });
        if (active_resources_users == 0) {
            active_resources = resources;
            active_resources_identity = identity;
            active_resources_closing = false;
            owner_ = true;
        }
        ++active_resources_users;
    }

    BackendResources::~BackendResources() {
        std::unique_lock lock(active_resources_mutex);
        if (owner_) {
            active_resources_closing = true;
            active_resources_changed.wait(lock, [] { return active_resources_users == 1; });
            active_resources_users = 0;
            active_resources = {};
            active_resources_identity.clear();
            active_resources_closing = false;
        } else {
            --active_resources_users;
        }
        lock.unlock();
        active_resources_changed.notify_all();
    }

    void register_backends(std::span<Mln::Sdk::Backend::Compatibility *const> compatibilities) {
        static std::mutex mutex;
        static std::unordered_map<std::string, Mln::Sdk::Backend::Compatibility *> registered;
        std::scoped_lock lock(mutex);

        Hal::TargetBackendList backends;
        Hal::TargetDeviceList devices;
        for (Mln::Sdk::Backend::Compatibility *compatibility : compatibilities) {
            const std::string name(compatibility->name());
            if (const char *trace = std::getenv("MLN_TRACE_BACKEND"); trace && *trace)
                std::println(stderr, "registering backend compatibility {} ({})", name,
                             compatibility->format());
            if (const auto existing = registered.find(name); existing != registered.end()) {
                if (existing->second != compatibility)
                    raise(Errors::Compiler::Backend::Compatibility::Duplicated, name);
                continue;
            }
            registered.emplace(name, compatibility);
            backends.add(name,
                         [compatibility] { return std::make_shared<Target>(*compatibility); });
            devices.add(name, [compatibility] { return std::make_shared<Device>(*compatibility); });
        }
        Hal::TargetRegistry::getMutableTargetRegistry().mergeFrom(backends);
        Hal::TargetRegistry::getMutableTargetRegistry().mergeFrom(devices);
    }
}
