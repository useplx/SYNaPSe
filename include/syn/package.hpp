#pragma once

#include "mln/sdk/codec.hpp"
#include "mln/sdk/module.hpp"
#include "mln/sdk/operation/compatibility.hpp"
#include "syn/hook.hpp"
#include "syn/lowering.hpp"
#include "syn/package_registry.hpp"

#include <memory>
#include <type_traits>
#include <variant>

namespace Syn::Compiler {
    class Compiler;
    struct Manifest;

    namespace Codec {
        class Compatibility;
    }

    namespace Package {
        namespace Invocation {
            enum class Kind {
                Describe,
                Codec,
                CodecCompatibility,
                OperationCompatibility,
                LoweringCompatibility,
                BackendCompatibility,
                Hook,
                Graph,
            };

            struct Context {
                    Kind kind;
                    Manifest *manifest = nullptr;
                    Compiler *compiler = nullptr;
                    void *output = nullptr;
            };
        }

        using GraphFactory = std::unique_ptr<Mln::Sdk::Module> (*)();

        using Object =
            std::variant<GraphFactory, Hook *, Mln::Sdk::Codec::Format *, Codec::Compatibility *,
                         Mln::Sdk::Operation::Compatibility *, Mln::Sdk::Lowering::Compatibility *,
                         Mln::Sdk::Backend::Compatibility *>;

        template <class> inline constexpr bool unsupported_package_export = false;

        template <class T> Object package_object(T &value) {
            using Type = std::remove_cvref_t<T>;
            if constexpr (std::is_same_v<Type, Hook>)
                return Object{&value};
            else if constexpr (std::is_same_v<Type, Mln::Sdk::Codec::Format>)
                return Object{&value};
            else if constexpr (std::is_base_of_v<Codec::Compatibility, Type>)
                return Object{static_cast<Codec::Compatibility *>(&value)};

            else if constexpr (std::is_base_of_v<Mln::Sdk::Backend::Compatibility, Type>)
                return Object{static_cast<Mln::Sdk::Backend::Compatibility *>(&value)};
            else if constexpr (std::is_base_of_v<Mln::Sdk::Operation::Compatibility, Type>)
                return Object{static_cast<Mln::Sdk::Operation::Compatibility *>(&value)};
            else if constexpr (std::is_base_of_v<Mln::Sdk::Lowering::Compatibility, Type>)
                return Object{static_cast<Mln::Sdk::Lowering::Compatibility *>(&value)};
            else
                static_assert(unsupported_package_export<Type>,
                              "unsupported PUBLISH implementation type");
        }

        template <class T, auto Factory> const Object *published_object() {
            if constexpr (std::is_base_of_v<Mln::Sdk::Module, T>) {
                static const Object object{static_cast<GraphFactory>(
                    [] { return std::unique_ptr<Mln::Sdk::Module>(Factory()); })};
                return &object;
            } else {
                static auto instance = Factory();
                static const Object object = package_object(*instance);
                return &object;
            }
        }
    }
}

#ifdef PUBLISH
#undef PUBLISH
#endif
#define PUBLISH(symbol, implementation)                                                            \
    extern "C" const ::Syn::Compiler::Package::Object *mln_package_##symbol() {                    \
        using Published = std::remove_cvref_t<decltype(implementation)>;                           \
        return ::Syn::Compiler::Package::published_object<Published, [] {                          \
            return std::unique_ptr<Published>(new Published(implementation));                      \
        }>();                                                                                      \
    }
