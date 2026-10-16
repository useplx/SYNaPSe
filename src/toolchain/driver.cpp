#include "syn/toolchain/driver.hpp"
#include "syn/toolchain/abi.hpp"

#include "syn/error.hpp"
#include "syn/system/process.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iterator>
#include <map>
#include <string_view>

namespace Syn::Toolchain {
    namespace {
        std::filesystem::path mln_root_;

        std::filesystem::path which(std::string_view executable) {
            const char *path = std::getenv("PATH");
            if (!path)
                return {};

            std::string_view remaining(path);
            while (!remaining.empty()) {
                const std::size_t separator = remaining.find(':');
                const std::string_view directory = remaining.substr(0, separator);
                if (!directory.empty()) {
                    std::filesystem::path candidate = std::filesystem::path(directory) / executable;
                    std::error_code error;
                    if (std::filesystem::exists(candidate, error) &&
                        ::access(candidate.c_str(), X_OK) == 0)
                        return candidate;
                }
                if (separator == std::string_view::npos)
                    break;
                remaining.remove_prefix(separator + 1);
            }
            return {};
        }

        std::filesystem::path first_existing(std::initializer_list<std::string_view> candidates) {
            for (std::string_view candidate : candidates) {
                if (candidate.find('/') != std::string_view::npos) {
                    std::error_code error;
                    if (std::filesystem::exists(candidate, error))
                        return std::filesystem::path(candidate);
                    continue;
                }
                if (std::filesystem::path found = which(candidate); !found.empty())
                    return found;
            }
            return {};
        }
    }

    std::string_view runtime_name(Runtime runtime) {
        switch (runtime) {
        case Runtime::Iree:
            return "iree";
        case Runtime::Native:
            return "native";
        case Runtime::Driver:
            return "driver";
        case Runtime::Wasm:
            return "wasm";
        }
        return "unknown";
    }

    namespace Myelin {
        void set_root(std::filesystem::path root) { mln_root_ = std::move(root); }

        const std::filesystem::path &root() {
            if (mln_root_.empty())
                Errors::raise(Error::Myelin::Root::Unresolved);
            return mln_root_;
        }
    }

    namespace Abi {
        Requirements requirements(Runtime runtime, Output output) {
            Requirements requirements;
            switch (runtime) {
            case Runtime::Wasm:
                requirements.reactor_model = true;
                requirements.disable_exceptions = true;
                requirements.mln_sdk_core = true;
                break;
            case Runtime::Native:
                if (output == Output::SharedLibrary) {
                    requirements.position_independent = true;
                    requirements.mln_sdk_core = true;
                    requirements.defines = {"MLN_NATIVE_BUILD"};
                    requirements.required_symbols = {"mln_native_module"};
                } else if (output == Output::StaticLibrary)
                    requirements.position_independent = true;
                break;
            case Runtime::Driver:
                if (output == Output::SharedLibrary) {
                    requirements.position_independent = true;
                    requirements.defines = {"MLN_RUNTIME_DRIVER_BUILD"};
                    requirements.required_symbols = {"mln_runtime_driver"};
                } else if (output == Output::StaticLibrary)
                    requirements.position_independent = true;
                break;
            case Runtime::Iree:
                requirements.position_independent = output == Output::SharedLibrary;
                requirements.mln_sdk_core = output == Output::SharedLibrary;
                if (output == Output::SharedLibrary)
                    requirements.required_symbols = {"mln_package_register"};
                break;
            }
            return requirements;
        }
    }

    namespace {
        std::map<std::string_view, const Driver *> &registry() {
            static std::map<std::string_view, const Driver *> drivers;
            return drivers;
        }
    }

    void register_driver(const Driver &driver) { registry()[driver.language()] = &driver; }

    const Driver &driver(std::string_view language) {
        const auto found = registry().find(language);
        if (found == registry().end()) {
            std::string known;
            for (const auto &[name, _] : registry()) {
                if (!known.empty())
                    known += ", ";
                known += name;
            }
            Errors::raise(Error::Language::Without::Driver, language, known);
        }
        return *found->second;
    }

    std::vector<std::string_view> languages() {
        std::vector<std::string_view> names;
        for (const auto &[name, _] : registry())
            names.push_back(name);
        return names;
    }

    std::optional<Config::Toolchain> discover(std::string_view language, Runtime runtime) {
        Config::Toolchain toolchain;
        toolchain.language = std::string(language);

        if (language == "cpp" || language == "c") {
            const bool c = language == "c";
            if (runtime == Runtime::Wasm) {

                if (const char *sdk = std::getenv("WASI_SDK_PATH")) {
                    const std::filesystem::path base = std::filesystem::path(sdk) / "bin";
                    for (const char *name : {"wasm32-wasip1-clang++", "wasm32-wasi-clang++"}) {
                        std::error_code error;
                        if (std::filesystem::exists(base / name, error)) {
                            toolchain.compiler = base / name;
                            break;
                        }
                    }
                }
                if (toolchain.compiler.empty())
                    toolchain.compiler =
                        first_existing({"/opt/wasi-sdk/bin/wasm32-wasip1-clang++",
                                        "/opt/wasi-sdk/bin/wasm32-wasi-clang++",
                                        "wasm32-wasip1-clang++", "wasm32-wasi-clang++"});
                toolchain.triple = "wasm32-wasi";
            } else {
                toolchain.compiler = c ? first_existing({"cc", "clang", "gcc"})
                                       : first_existing({"c++", "clang++", "g++"});
            }
        } else if (language == "rust") {
            toolchain.compiler = first_existing({"cargo"});
        } else if (language == "zig") {
            toolchain.compiler = first_existing({"zig"});
        } else if (language == "go") {
            toolchain.compiler =
                runtime == Runtime::Wasm ? first_existing({"tinygo"}) : first_existing({"go"});
        }

        if (toolchain.compiler.empty())
            return std::nullopt;
        return toolchain;
    }

    Registry::Registry(std::map<std::string, Config::Toolchain, std::less<>> available)
        : declared_(std::move(available)) {}

    Selection Registry::select(Runtime runtime, std::string_view language,
                               const std::vector<std::string> &preference) const {
        for (const std::string &name : preference) {
            if (name.empty())
                continue;
            const auto found = declared_.find(name);
            if (found == declared_.end())
                Errors::raise(Error::Toolchain::Not::Declared, name);
            if (found->second.language != language)
                Errors::raise(Error::Toolchain::Language::Mismatch, name, found->second.language,
                              language);
            return Selection{name, found->second, "declarada"};
        }

        const std::string domain = runtime == Runtime::Iree
                                       ? "graph"
                                       : (runtime == Runtime::Wasm ? "orchestration" : "native");
        const std::string architecture = [] {
#if defined(__aarch64__)
            return std::string("aarch64");
#elif defined(__x86_64__)
            return std::string("x86_64");
#elif defined(__riscv)
            return std::string("riscv64");
#else
            return std::string("unknown");
#endif
        }();
        const std::string prefix = std::string(language) + ".";
        const Config::Toolchain *best = nullptr;
        std::string best_name;
        int best_score = -1;
        for (const auto &[name, candidate] : declared_) {
            if (candidate.language != language)
                continue;
            int score = name == language ? 1 : -1;
            if (name.starts_with(prefix)) {
                std::string_view suffix(name.data() + prefix.size(), name.size() - prefix.size());
                const std::size_t separator = suffix.find('.');
                const std::string_view candidate_domain = suffix.substr(0, separator);
                if (candidate_domain != domain)
                    continue;
                score = 10;
                if (separator != std::string_view::npos) {
                    std::string_view labels = suffix.substr(separator + 1);
                    while (!labels.empty()) {
                        const std::size_t next = labels.find('.');
                        const std::string_view label = labels.substr(0, next);
                        if (label == "*")
                            score += 1;
                        else if (label == architecture ||
                                 (runtime == Runtime::Wasm && label == "wasm32"))
                            score += 3;
                        else {
                            score = -1;
                            break;
                        }
                        if (next == std::string_view::npos)
                            break;
                        labels.remove_prefix(next + 1);
                    }
                }
            }
            if (score > best_score) {
                best = &candidate;
                best_name = name;
                best_score = score;
            } else if (score >= 0 && score == best_score) {
                Errors::raise(Error::Toolchain::Ambiguous, language);
            }
        }
        if (best)
            return Selection{best_name, *best, "toolchain hierarquica"};

        const auto candidate =
            std::find_if(declared_.begin(), declared_.end(),
                         [&](const auto &item) { return item.second.language == language; });
        if (candidate != declared_.end()) {
            const auto another =
                std::find_if(std::next(candidate), declared_.end(),
                             [&](const auto &item) { return item.second.language == language; });
            if (another != declared_.end())
                Errors::raise(Error::Toolchain::Ambiguous, language);
            return Selection{candidate->first, candidate->second,
                             "unica declarada para a linguagem"};
        }

        if (std::optional<Config::Toolchain> found = discover(language, runtime))
            return Selection{std::string(language) + "-" + std::string(runtime_name(runtime)),
                             *found, "descoberta no ambiente"};

        Errors::raise(Error::Toolchain::Unavailable, language, runtime_name(runtime));
    }
}
