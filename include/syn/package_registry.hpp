#pragma once

#include "syn/package_abi.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Syn::Compiler {
    namespace Package {
        namespace Abi {
            inline constexpr uint32_t Version = 6;

            using Invocation = mln_package_invocation;
            using Invoke = mln_package_invoke;
            using Export = mln_package_export;
            using Registration = mln_package_registration;
            using Register = mln_package_register_function;
        }

        struct Export {

                enum class Kind : uint32_t {
                    Graph = mln_package_graph,
                    Codec = mln_package_codec,
                    Hook = mln_package_hook,
                    CodecCompatibility = mln_package_codec_compatibility,
                    OperationCompatibility = mln_package_operation_compatibility,
                    LoweringCompatibility = mln_package_lowering_compatibility,
                    BackendCompatibility = mln_package_backend_compatibility,
                };

                std::string package;
                std::string package_version;
                std::string name;
                Kind kind;
                Abi::Invoke invoke = nullptr;
                void *user_data = nullptr;

                std::string id() const {
                    std::string result;
                    result.reserve(package.size() + name.size() + 1);
                    result.append(package);
                    result += ':';
                    result.append(name);
                    return result;
                }
        };

        class Registry {
            public:
                void load(std::string_view path);

                const Export *resolve(Export::Kind kind, std::string_view id) const;

                std::vector<Export> exports() const;

                void pin();

                ~Registry();
                Registry() = default;
                Registry(const Registry &) = delete;
                Registry &operator=(const Registry &) = delete;

            private:
                using Library = std::unique_ptr<void, void (*)(void *)>;
                std::unordered_map<std::string, Export> exports_;
                std::vector<Library> libraries_;
        };
    }
}
