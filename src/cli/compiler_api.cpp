#include "syn/compiler/api.hpp"

#include "syn/error.hpp"

#include <syn/compiler_abi.h>

#include <dlfcn.h>

#include <format>
#include <print>
#include <string_view>

namespace Syn::Compiler {
    struct Api::Library {
            explicit Library(void *library) : handle(library) {}
            ~Library() {
                if (handle)
                    ::dlclose(handle);
            }

            Library(const Library &) = delete;
            Library &operator=(const Library &) = delete;
            void *handle = nullptr;
    };

    Api::~Api() = default;
    Api::Api(std::unique_ptr<Library> library, const mln_compiler_api *api)
        : library_(std::move(library)), api_(api) {}
    Api::Api(Api &&) noexcept = default;
    Api &Api::operator=(Api &&) noexcept = default;

    Api Api::load(const std::filesystem::path &library) {
        ::dlerror();
        void *handle = ::dlopen(library.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!handle) {
            const char *detail = ::dlerror();
            Errors::at(Error::Compiler::Api::Unloadable, library.string(),
                       detail ? detail : "dlopen did not report a cause");
        }
        auto owned = std::make_unique<Library>(handle);
        ::dlerror();
        auto getter = reinterpret_cast<const mln_compiler_api *(*)()>(
            ::dlsym(handle, "mln_compiler_api_get"));
        if (const char *detail = ::dlerror(); detail || !getter)
            Errors::at(Error::Compiler::Api::Symbol::Missing, library.string(),
                       detail ? detail : "mln_compiler_api_get is missing");

        const mln_compiler_api *api = getter();
        if (!api || api->abi_version != mln_compiler_api_version ||
            api->struct_size < sizeof(mln_compiler_api) || !api->compile)
            Errors::at(Error::Compiler::Api::Incompatible, library.string());
        return Api(std::move(owned), api);
    }

    void Api::compile(const Config::Value &plan, const std::filesystem::path &output) const {
        const std::string serialized = Config::json(plan);
        std::string diagnostics;
        const std::string output_path = output.string();
        const mln_compiler_request request{
            mln_compiler_api_version,
            sizeof(mln_compiler_request),
            serialized.data(),
            serialized.size(),
            output_path.c_str(),
            [](void *context, const char *message, std::size_t length) {
                auto &destination = *static_cast<std::string *>(context);
                if (!destination.empty())
                    destination.push_back('\n');
                destination.append(message, length);
            },
            &diagnostics,
            [](void *context, mln_compiler_diagnostic_severity severity, const char *domain,
               std::size_t domain_length, std::uint16_t code, const char *message,
               std::size_t message_length, const char *location, std::size_t location_length) {
                auto &destination = *static_cast<std::string *>(context);
                const std::string_view origin(domain, domain_length);
                const std::string local_code = std::format("E{}", code);
                const std::string display_code =
                    origin.empty() ? local_code : std::format("{}:{}", origin, local_code);
                const std::string_view text(message, message_length);
                const std::string_view place(location, location_length);
                const std::string rendered =
                    place.empty()
                        ? std::format("{}[{}]: {}",
                                      severity == MLN_COMPILER_DIAGNOSTIC_ERROR ? "error"
                                                                                : "warning",
                                      display_code, text)
                        : std::format("{}: {}[{}]: {}", place,
                                      severity == MLN_COMPILER_DIAGNOSTIC_ERROR ? "error"
                                                                                : "warning",
                                      display_code, text);
                if (severity == MLN_COMPILER_DIAGNOSTIC_WARNING) {
                    std::println(stderr, "{}", rendered);
                } else {
                    if (!destination.empty())
                        destination.push_back('\n');
                    destination += rendered;
                }
            },
            nullptr,
        };
        const int32_t status = api_->compile(&request);
        if (status != 0)
            Errors::at(Error::Compilation::Failed, output_path,
                       diagnostics.empty() ? "status " + std::to_string(status) : diagnostics);
    }
}
