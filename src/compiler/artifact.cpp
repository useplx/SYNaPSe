#include "syn/artifact.hpp"
#include "mln/error.hpp"
#include "syn/errors/artifact.hpp"

#include <filesystem>
#include <format>
#include <set>
#include <stdexcept>
#include <string_view>

namespace Syn::Compiler {
    namespace {
        void validate_name(std::string_view kind, std::string_view name) {
            if (name.empty())
                raise(Syn::Compiler::Errors::Artifact::Name::Empty, kind);
            if (name.find_first_of("\t\r\n") != std::string_view::npos)
                raise(Syn::Compiler::Errors::Artifact::Name::Invalid, kind, name);
        }

        void validate_file(std::string_view kind, const Mln::Artifact::File &file) {
            validate_name(kind, file.name);
            if (file.path.empty())
                raise(Syn::Compiler::Errors::Artifact::File::Path::Missing, kind, file.name);
            if (!std::filesystem::is_regular_file(file.path))
                raise(Syn::Compiler::Errors::Artifact::File::Missing, kind, file.name, file.path);
        }
    }

    void validate_artifact_plan(const Mln::Artifact::Plan &plan, bool require_entrypoints) {
        if (require_entrypoints && plan.entrypoints.empty())
            raise(Syn::Compiler::Errors::Artifact::Entrypoint::Missing);

        std::set<std::string> wasm_modules;
        std::set<std::string> native_modules;
        for (const Mln::Artifact::Module &module : plan.modules) {
            validate_file("runtime module", module.artifact);
            std::set<std::string> &names =
                module.runtime == Mln::Artifact::Runtime::Wasm ? wasm_modules : native_modules;
            if (!names.insert(module.artifact.name).second)
                raise(Syn::Compiler::Errors::Artifact::Module::Duplicated, module.artifact.name);
        }

        std::set<std::string> entrypoints;
        for (const Mln::Artifact::Entrypoint &entrypoint : plan.entrypoints) {
            validate_name("entrypoint", entrypoint.name);
            validate_name("entrypoint module", entrypoint.module);
            validate_name("entrypoint export", entrypoint.export_name);
            if (!entrypoints.insert(entrypoint.name).second)
                raise(Syn::Compiler::Errors::Artifact::Entrypoint::Duplicated, entrypoint.name);
            if (entrypoint.runtime == Mln::Artifact::Runtime::Wasm &&
                !wasm_modules.contains(entrypoint.module))
                raise(Syn::Compiler::Errors::Artifact::Entrypoint::Module::Missing, entrypoint.name,
                      entrypoint.module);
            if (entrypoint.runtime == Mln::Artifact::Runtime::Native && native_modules.empty())
                raise(Syn::Compiler::Errors::Artifact::Entrypoint::Module::Missing, entrypoint.name,
                      entrypoint.module);
        }

        for (const Mln::Artifact::Hook &hook : plan.hooks) {
            if (hook.path.empty())
                raise(Syn::Compiler::Errors::Artifact::Hook::Path::Empty);
            validate_name("modulo Wasm de hook", hook.module);
            validate_name("export Wasm de hook", hook.export_name);
            if (!wasm_modules.contains(hook.module))
                raise(Syn::Compiler::Errors::Artifact::Hook::Module::Missing, hook.path,
                      hook.module);
        }

        std::set<std::string> assets;
        for (const Mln::Artifact::File &asset : plan.assets) {
            validate_file("asset", asset);
            if (!assets.insert(asset.name).second)
                raise(Syn::Compiler::Errors::Artifact::Asset::Duplicated, asset.name);
        }
    }
}
