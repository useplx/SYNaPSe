#pragma once

#include "syn/config/project.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Syn::Toolchain {
    enum class Runtime { Iree, Native, Driver, Wasm };

    std::string_view runtime_name(Runtime runtime);

    enum class Output { SharedLibrary, StaticLibrary, WasmModule, Executable };

    struct Request {
            Runtime runtime = Runtime::Iree;
            Output output = Output::SharedLibrary;

            std::vector<std::filesystem::path> sources;
            std::vector<std::filesystem::path> includes;
            std::vector<std::string> defines;
            std::vector<std::string> libraries;
            std::vector<std::filesystem::path> library_directories;
            std::vector<std::string> flags;

            std::filesystem::path artifact;
            std::filesystem::path working_directory;
            std::filesystem::path intermediate_directory;
            bool optimize = false;
    };

    class Driver {
        public:
            virtual ~Driver() = default;

            virtual std::string_view language() const = 0;

            virtual void probe(const Config::Toolchain &toolchain) const = 0;

            virtual bool supports(Runtime runtime, Output output) const = 0;

            virtual void build(const Config::Toolchain &toolchain,
                               const Request &request) const = 0;
    };

    void register_driver(const Driver &driver);

    const Driver &driver(std::string_view language);
    std::vector<std::string_view> languages();

    struct Selection {
            std::string name;
            Config::Toolchain toolchain;
            std::string origin;
    };

    class Registry {
        public:
            explicit Registry(std::map<std::string, Config::Toolchain, std::less<>> available);

            Selection select(Runtime runtime, std::string_view language,
                             const std::vector<std::string> &preference) const;

            const std::map<std::string, Config::Toolchain, std::less<>> &declared() const noexcept {
                return declared_;
            }

        private:
            std::map<std::string, Config::Toolchain, std::less<>> declared_;
    };

    std::optional<Config::Toolchain> discover(std::string_view language, Runtime runtime);
}
