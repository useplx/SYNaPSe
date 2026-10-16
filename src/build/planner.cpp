#include "syn/build/planner.hpp"

#include "syn/compiler/api.hpp"
#include "syn/error.hpp"
#include "syn/system/hash.hpp"
#include "syn/toolchain/abi.hpp"
#include "syn/toolchain/sdk.hpp"
#include "syn/toolchain/symbols.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <variant>

namespace Syn::Build {
    namespace {
        bool is_weight_file(const std::filesystem::path &path) {
            const std::string extension = path.extension().string();
            return extension == ".safetensors" || extension == ".gguf" || extension == ".irpa";
        }

        std::vector<std::filesystem::path> weight_files(const std::filesystem::path &source) {
            if (std::filesystem::is_regular_file(source))
                return {source};
            std::vector<std::filesystem::path> result;
            std::error_code error;
            for (const auto &item : std::filesystem::directory_iterator(source, error))
                if (item.is_regular_file() && is_weight_file(item.path()))
                    result.push_back(item.path());
            // Um diretorio ilegivel devolve range vazio. Sem checar, o
            // checkpoint sumiria do fingerprint e o .mln sairia sem pesos.
            if (error)
                Errors::at(Error::File::Unreadable, "weights", source, error.message());
            std::sort(result.begin(), result.end());
            return result;
        }

        // Hash de conteudo para checkpoints de varios GB tornaria um build
        // sem mudancas mais caro que o packaging. Tamanho+mtime detecta a
        // substituicao/edicao local sem reler os pesos; snapshots git/HF sao
        // imutaveis e o lockfile continua sendo a identidade reproduzivel.
        std::string weight_signature(const std::vector<std::filesystem::path> &sources) {
            std::string result;
            for (const std::filesystem::path &source : sources)
                for (const std::filesystem::path &file : weight_files(source)) {
                    result +=
                        file.lexically_normal().string() + ":" +
                        std::to_string(std::filesystem::file_size(file)) + ":" +
                        std::to_string(
                            std::filesystem::last_write_time(file).time_since_epoch().count()) +
                        "\n";
                }
            return result;
        }

        Toolchain::Runtime runtime_of(const Config::Build::Package &build_package,
                                      const std::string &package) {
            if (build_package.runtime == "iree")
                return Toolchain::Runtime::Iree;
            if (build_package.runtime == "native")
                return Toolchain::Runtime::Native;
            if (build_package.runtime == "driver")
                return Toolchain::Runtime::Driver;
            if (build_package.runtime == "wasm")
                return Toolchain::Runtime::Wasm;
            Errors::at(Error::Runtime::Unknown, package + ".package." + build_package.name,
                       build_package.runtime);
        }

        Toolchain::Output output_of(Toolchain::Runtime runtime,
                                    const Config::Build::Package &package) {
            return runtime == Toolchain::Runtime::Wasm ? Toolchain::Output::WasmModule
                   : std::filesystem::path(package.artifact).extension() == ".a"
                       ? Toolchain::Output::StaticLibrary
                       : Toolchain::Output::SharedLibrary;
        }

        std::string artifact_name(const Config::Build::Package &build_package,
                                  const std::string &package, Toolchain::Runtime runtime) {
            if (runtime == Toolchain::Runtime::Wasm)
                return package + "-" + build_package.name + ".wasm";
            return "lib" + package + "-" + build_package.name + ".so";
        }

        // Descoberta por convencao quando o build_package nao lista sources. O
        // main.cpp e excluido de proposito: ele e o ponto de entrada de uma
        // ferramenta do package, e compila-lo dentro de um plugin ou de uma
        // orchestration criaria um segundo `main` no artefato.
        std::vector<std::filesystem::path>
        discover_sources(const Package::Resolved::Package &package,
                         const Config::Build::Package &build_package) {
            std::vector<std::filesystem::path> sources;
            const std::filesystem::path root = package.project.root();

            if (!build_package.sources.empty()) {
                for (const std::filesystem::path &entry : build_package.sources) {
                    const std::filesystem::path absolute =
                        entry.is_absolute() ? entry : root / entry;
                    std::error_code error;
                    if (std::filesystem::is_directory(absolute, error)) {
                        for (const auto &item :
                             std::filesystem::recursive_directory_iterator(absolute, error))
                            if (item.is_regular_file())
                                sources.push_back(item.path());
                        // Um walk interrompido devolve menos sources em
                        // silencio, e o build_package compila sem parte do codigo.
                        if (error)
                            Errors::at(Error::File::Unreadable,
                                       package.project.metadata().name + ".package." +
                                           build_package.name,
                                       absolute, error.message());
                    } else {
                        if (!std::filesystem::exists(absolute, error))
                            Errors::at(Error::Path::Missing,
                                       package.project.metadata().name + ".package." +
                                           build_package.name,
                                       entry);
                        sources.push_back(absolute);
                    }
                }
                std::sort(sources.begin(), sources.end());
                return sources;
            }

            const std::filesystem::path directory = root / package.project.layout().sources;
            std::error_code error;
            if (!std::filesystem::exists(directory, error))
                return sources;

            for (const auto &item :
                 std::filesystem::recursive_directory_iterator(directory, error)) {
                if (!item.is_regular_file())
                    continue;
                const std::filesystem::path &path = item.path();
                const std::string extension = path.extension().string();
                if (extension != ".cpp" && extension != ".cc" && extension != ".c")
                    continue;
                if (path.filename() == "main.cpp")
                    continue;
                sources.push_back(path);
            }
            if (error)
                Errors::at(Error::File::Unreadable,
                           package.project.metadata().name + ".package." + build_package.name,
                           directory, error.message());
            std::sort(sources.begin(), sources.end());
            return sources;
        }

        std::string expand_environment(std::string_view value, std::string_view origin);

        std::vector<std::filesystem::path> include_paths(
            const Package::Resolution &resolution, const Package::Resolved::Package &package,
            const Config::Build::Package &build_package, bool include_dependencies = true) {
            std::vector<std::filesystem::path> includes;
            const std::filesystem::path root = package.project.root();
            includes.push_back(root / package.project.layout().includes);

            for (const std::filesystem::path &declared : build_package.includes) {
                // GCC and Clang interpret an include beginning with '=' relative
                // to their active sysroot. Preserve that spelling so packages
                // can declare target system headers without embedding an SDK
                // installation path or accidentally reading host headers.
                const std::string value = expand_environment(
                    declared.string(), package.project.metadata().name + ".package." +
                                           build_package.name + ".includes");
                const std::filesystem::path entry(value);
                includes.push_back(entry.is_absolute() || value.starts_with('=') ? entry
                                                                                 : root / entry);
            }

            for (const std::filesystem::path &entry :
                 package.project.metadata().include_directories)
                includes.push_back(entry.is_absolute() ? entry : root / entry);

            // Includes publicos das dependencias diretas. Um package enxerga
            // o que declarou depender, e nada alem disso.
            if (include_dependencies)
                for (const Package::Dependency::Edge &edge : package.dependencies) {
                    const Package::Resolved::Package &dependency =
                        resolution.packages().at(edge.index);
                    const std::filesystem::path &base = dependency.project.root();
                    includes.push_back(base / dependency.project.layout().includes);
                    for (const std::filesystem::path &entry :
                         dependency.project.metadata().include_directories)
                        includes.push_back(entry.is_absolute() ? entry : base / entry);
                }
            return includes;
        }

        // Nome estavel do no: package + build_package. Colisao aqui seria dois
        // build_packages disputando o mesmo artefato.
        std::string node_name(const std::string &package,
                              const Config::Build::Package &build_package) {
            return package + ":" + build_package.name;
        }

        // O alvo resolvido de um `extends`. Agrupado sob Extends e nao sob
        // Package porque `Package` aqui dentro colidiria com o namespace que
        // resolve dependencias.
        namespace Extends {
            struct Target {
                    const Package::Resolved::Package *package;
                    const Config::Build::Package *build_package;
            };
        } // namespace Extends

        // Resolve um `project:package` de um `extends`. As checagens sao
        // deliberadamente estritas: uma referencia que nao resolve e um erro
        // de composicao, e falhar aqui e muito mais barato que descobrir no
        // link que faltou um objeto.
        Extends::Target dereference(const Package::Resolution &resolution,
                                    const Package::Resolved::Package &from,
                                    const Config::Build::Package &build_package,
                                    const std::string &reference) {
            const std::string origin =
                from.project.metadata().name + ".package." + build_package.name;
            const std::size_t separator = reference.find(':');
            if (separator == std::string::npos || separator == 0 ||
                separator + 1 == reference.size())
                Errors::at(Error::Extends::Format::Invalid, origin, reference);

            const std::string project_name = reference.substr(0, separator);
            const std::string target_package_name = reference.substr(separator + 1);

            // So se compoe a partir do que se declarou depender. Sem isso um
            // package entraria no artefato sem aparecer em [dependencies], e
            // o grafo de dependencias deixaria de descrever o build. O nome
            // aceito e o alias de [dependencies] ou o nome real do package --
            // normalmente o mesmo, mas o alias e o que o projeto escreveu.
            const Package::Resolved::Package *target = nullptr;
            if (project_name == from.project.metadata().name) {
                target = &from;
            } else {
                for (const Package::Dependency::Edge &edge : from.dependencies) {
                    const Package::Resolved::Package &candidate =
                        resolution.packages().at(edge.index);
                    if (edge.name == project_name ||
                        candidate.project.metadata().name == project_name) {
                        target = &candidate;
                        break;
                    }
                }
            }
            if (target == nullptr)
                Errors::at(Error::Extends::Project::Unknown, origin, reference, project_name);

            const auto found = target->project.build_packages().find(target_package_name);
            if (found == target->project.build_packages().end())
                Errors::at(Error::Extends::Package::Unknown, origin, reference, project_name);

            // Dominios diferentes tem ABI, toolchain e artefato diferentes:
            // absorver um source de host dentro de um modulo Wasm nao produz
            // nada que linque.
            if (found->second.runtime != build_package.runtime && !found->second.runtime_inferred &&
                !build_package.runtime_inferred)
                Errors::at(Error::Extends::Runtime::Mismatch, origin, reference,
                           found->second.runtime, build_package.runtime);

            return {target, &found->second};
        }

        // Um campo de [package.variables] com `define` chega ao codigo do package
        // como -D. Sem isso a configuracao so alcancaria os plugins de host,
        // que a recebem pela ABI -- a orchestration, que e compilada, ficaria
        // presa nos defaults do header e passaria a discordar do grafo sobre
        // coisas como o tamanho do KV cache. O projeto escreve o valor uma vez
        // em [packages.<nome>] e os dois lados enxergam o mesmo numero.
        std::vector<std::string> configuration_defines(const Package::Resolution &resolution,
                                                       const Package::Resolved::Package &package,
                                                       std::string_view facet) {
            std::vector<std::string> defines;
            const Config::Value *declaration = package.project.document().find("package");
            const Config::Value *schema = declaration ? declaration->find("variables") : nullptr;
            if (schema == nullptr || !schema->is_table())
                return defines;

            const auto &configurations = resolution.configurations();
            const auto found = configurations.find(package.project.metadata().name);
            if (found == configurations.end())
                return defines;

            for (const auto &[field, rule] : schema->table()) {
                const Config::Value *name = rule.find("define");
                if (name == nullptr)
                    continue;
                if (const Config::Value *facets = rule.find("facets")) {
                    if (!facets->is_array())
                        Errors::at(Error::Manifest::Field::Array::Expected,
                                   "package.variables." + field + ".facets");
                    bool selected = false;
                    for (const Config::Value &candidate : facets->array()) {
                        if (!std::holds_alternative<std::string>(candidate.data))
                            Errors::at(Error::Manifest::Field::Array::Element::String::Expected,
                                       "package.variables." + field + ".facets");
                        selected = selected || candidate.string() == facet;
                    }
                    if (!selected)
                        continue;
                }
                const Config::Value *value = found->second.find(field);
                if (value == nullptr)
                    continue;

                std::string text;
                if (std::holds_alternative<bool>(value->data))
                    text = value->boolean() ? "1" : "0";
                else if (std::holds_alternative<int64_t>(value->data))
                    text = std::to_string(value->integer());
                else if (std::holds_alternative<double>(value->data))
                    text = std::to_string(value->real());
                else
                    text = "\"" + std::string(value->string()) + "\"";
                defines.push_back(std::string(name->string()) + "=" + text);
            }
            return defines;
        }

        // Simbolos publicos agrupados pelo modulo Wasm que os implementa.
        // O nome do modulo e o id estavel `project:package`, nao o nome do
        // arquivo produzido por um toolchain especifico.
        std::map<std::string, std::vector<std::string>, std::less<>>
        manifest_exports(const Package::Resolution &resolution) {
            std::map<std::string, std::vector<std::string>, std::less<>> exports;
            for (const Config::Value &entrypoint : Compiler::Plan::entrypoints(resolution)) {
                const Config::Value *runtime = entrypoint.find("runtime");
                if (runtime != nullptr && runtime->string() != "wasm")
                    continue;
                const Config::Value *module = entrypoint.find("module");
                if (module == nullptr)
                    Errors::at(Error::Entrypoint::Module::Missing, "entrypoints.module");
                const Config::Value *symbol = entrypoint.find("export");
                exports[std::string(module->string())].push_back(
                    symbol ? std::string(symbol->string()) : "forward");
            }
            for (const Config::Value &hook : Compiler::Plan::hooks(resolution)) {
                const Config::Value *module = hook.find("module");
                if (module == nullptr)
                    continue;
                const Config::Value *symbol = hook.find("export");
                exports[std::string(module->string())].push_back(
                    symbol ? std::string(symbol->string()) : "forward");
            }
            return exports;
        }

        std::string symbol_identifier(std::string_view name) {
            std::string result;
            result.reserve(name.size());
            for (const unsigned char ch : name)
                result.push_back(std::isalnum(ch) ? static_cast<char>(ch) : '_');
            return result;
        }

        std::string implementation_symbol(std::string_view exported) {
            return "mln_package_" + symbol_identifier(exported);
        }

        namespace Generated {
            struct Glue {
                    std::filesystem::path source;
            };
        } // namespace Generated

        Generated::Glue generate_package_glue(const Package::Resolved::Package &package,
                                              const Config::Build::Package &build_package,
                                              const std::filesystem::path &build_directory) {
            const std::string &package_name = package.project.metadata().name;
            const std::string &version = package.project.metadata().version;
            const std::string graph_namespace = build_package.graph_namespace.empty()
                                                    ? package_name
                                                    : build_package.graph_namespace;

            std::ostringstream source;
            source << R"cpp(#include <syn/package.hpp>
#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <string_view>

namespace {
    void report(const Syn::Compiler::Package::Abi::Invocation* invocation,
                std::string_view message) {
        if (!invocation->error || invocation->error_capacity == 0) {
            return;
        }
        const std::size_t length =
            std::min(message.size(), invocation->error_capacity - 1);
        std::memcpy(invocation->error, message.data(), length);
        invocation->error[length] = '\0';
    }
    uint32_t abi_kind(const Syn::Compiler::Package::Object& object) {
        using Kind = Syn::Compiler::Package::Export::Kind;
        const Kind kind =
            std::holds_alternative<Syn::Compiler::Package::GraphFactory>(object) ? Kind::Graph
            : std::holds_alternative<Syn::Compiler::Hook*>(object) ? Kind::Hook
            : std::holds_alternative<Syn::Compiler::Codec::Compatibility*>(object)
                ? Kind::CodecCompatibility
            : std::holds_alternative<Mln::Sdk::Operation::Compatibility*>(object)
                ? Kind::OperationCompatibility
            : std::holds_alternative<Mln::Sdk::Lowering::Compatibility*>(object)
                ? Kind::LoweringCompatibility
            : std::holds_alternative<Mln::Sdk::Backend::Compatibility*>(object)
                ? Kind::BackendCompatibility
                : Kind::Codec;
        return static_cast<uint32_t>(kind);
    }
)cpp";

            for (const std::string &exported : build_package.exports) {
                const std::string id = symbol_identifier(exported);
                source << "    extern \"C\" const Syn::Compiler::Package::Object* "
                       << implementation_symbol(exported) << "();\n";

                source
                    << "    int32_t invoke_" << id
                    << "(const Syn::Compiler::Package::Abi::Invocation* invocation, void*) {\n"
                       "        auto* context = "
                       "static_cast<Syn::Compiler::Package::Invocation::Context*>(invocation->"
                       "context);\n"
                       "        try {\n"
                       "            const auto& object = *"
                    << implementation_symbol(exported)
                    << "();\n"
                       "            using Kind = Syn::Compiler::Package::Invocation::Kind;\n"
                       "            if (const auto* factory = "
                       "std::get_if<Syn::Compiler::Package::GraphFactory>(&object)) {\n"
                       "                if (context->kind == Kind::Describe) {\n"
                       "                    *static_cast<std::string*>(context->output) = \""
                    << graph_namespace
                    << "\";\n                    return 0;\n                }\n"
                       "                if (context->kind == Kind::Graph) {\n"
                       "                    *static_cast<std::unique_ptr<Mln::Sdk::Module>*>("
                       "context->output) = (*factory)();\n"
                       "                }\n"
                       "            } else if (const auto* hook = "
                       "std::get_if<Syn::Compiler::Hook*>(&object)) {\n"
                       "                if (context->kind == Kind::Hook) {\n"
                       "                    *static_cast<Syn::Compiler::Hook*>(context->output) = "
                       "**hook;\n"
                       "                }\n"
                       "            } else if (const auto* format = "
                       "std::get_if<Mln::Sdk::Codec::Format*>(&object)) {\n"
                       "                if (context->kind == Kind::Codec) {\n"
                       "                    "
                       "*static_cast<Mln::Sdk::Codec::Format*>(context->output) "
                       "= **format;\n"
                       "                }\n"
                       "            } else if (const auto* compatibility = "
                       "std::get_if<Syn::Compiler::Codec::Compatibility*>(&object)) {\n"
                       "                if (context->kind == Kind::CodecCompatibility) {\n"
                       "                    "
                       "*static_cast<Syn::Compiler::Codec::Compatibility**>(context->output) = "
                       "*compatibility;\n"
                       "                }\n"
                       "            } else if (const auto* compatibility = "
                       "std::get_if<Mln::Sdk::Operation::Compatibility*>(&object)) {\n"
                       "                if (context->kind == Kind::OperationCompatibility) {\n"
                       "                    "
                       "*static_cast<Mln::Sdk::Operation::Compatibility**>(context->output) "
                       "= *compatibility;\n"
                       "                }\n"
                       "            } else if (const auto* compatibility = "
                       "std::get_if<Mln::Sdk::Lowering::Compatibility*>(&object)) {\n"
                       "                if (context->kind == Kind::LoweringCompatibility) {\n"
                       "                    "
                       "*static_cast<Mln::Sdk::Lowering::Compatibility**>(context->output) "
                       "= *compatibility;\n"
                       "                }\n"
                       "            } else if (const auto* compatibility = "
                       "std::get_if<Mln::Sdk::Backend::Compatibility*>(&object)) {\n"
                       "                if (context->kind == Kind::BackendCompatibility) {\n"
                       "                    "
                       "*static_cast<Mln::Sdk::Backend::Compatibility**>(context->output) "
                       "= *compatibility;\n"
                       "                }\n"
                       "            }\n";
                source << "            return 0;\n"
                          "        } catch (const std::exception& error) {\n"
                          "            report(invocation, error.what());\n"
                          "            return 1;\n"
                          "        }\n    }\n";
            }

            source << "}\nextern \"C\" const Syn::Compiler::Package::Abi::Registration* "
                      "mln_package_register() {\n"
                      "    static const Syn::Compiler::Package::Abi::Export exports[] = {\n";
            for (const std::string &exported : build_package.exports) {
                source << "        {sizeof(Syn::Compiler::Package::Abi::Export), abi_kind(*"
                       << implementation_symbol(exported) << "())"
                       << ", \"" << exported << "\", invoke_" << symbol_identifier(exported)
                       << ", nullptr},\n";
            }
            source << "    };\n    static const Syn::Compiler::Package::Abi::Registration "
                      "registration{Syn::Compiler::Package::Abi::Version, "
                      "sizeof(Syn::Compiler::Package::Abi::Registration), \""
                   << package_name << "\", \"" << version
                   << "\", exports, sizeof(exports) / sizeof(exports[0])};\n"
                      "    return &registration;\n}\n";

            const std::filesystem::path path =
                build_directory / ".synps" / "generated" /
                (package_name + "-" + build_package.name + "-package.cpp");
            std::filesystem::create_directories(path.parent_path());
            const std::string contents = source.str();
            std::ifstream existing(path);
            const std::string previous((std::istreambuf_iterator<char>(existing)), {});
            if (previous != contents)
                std::ofstream(path) << contents;
            return {path};
        }

        // Tudo que os packages declararam exportar, como `package:nome`. O
        // manifest referencia esses nomes em `implementation`, e um erro de
        // digitacao ali so apareceria no meio da compilacao do grafo -- depois
        // de ja ter lido os pesos.
        std::set<std::string> declared_exports(const Package::Resolution &resolution) {
            std::set<std::string> exports;
            for (const Package::Resolved::Package &package : resolution.packages()) {
                const std::string &name = package.project.metadata().name;
                for (const auto &[package_name, build_package] : package.project.build_packages())
                    for (const std::string &item : build_package.exports)
                        exports.insert(name + ":" + item);
            }
            return exports;
        }

        template <typename T>
        void append_unique(std::vector<T> &target, const std::vector<T> &source) {
            for (const T &entry : source)
                if (std::find(target.begin(), target.end(), entry) == target.end())
                    target.push_back(entry);
        }

        std::string expand_environment(std::string_view value, std::string_view origin) {
            std::string result(value);
            std::size_t offset = 0;
            while ((offset = result.find("${", offset)) != std::string::npos) {
                const std::size_t end = result.find('}', offset + 2);
                if (end == std::string::npos)
                    Errors::at(Error::Environment::Reference::Unterminated, std::string(origin));
                const std::string name = result.substr(offset + 2, end - offset - 2);
                if (name.empty())
                    Errors::at(Error::Environment::Reference::Empty, std::string(origin));
                const char *replacement = std::getenv(name.c_str());
                if (replacement == nullptr || replacement[0] == '\0')
                    Errors::at(Error::Environment::Variable::Unset, std::string(origin), name);
                result.replace(offset, end - offset + 1, replacement);
                offset += std::char_traits<char>::length(replacement);
            }
            return result;
        }
    } // namespace

    Planner::Plan Planner::run(const Package::Resolution &resolution,
                               const Toolchain::Registry &toolchains, const Options &options) {
        Plan result;
        result.package =
            options.build_directory / (resolution.root().project.metadata().name + ".mln");

        std::vector<std::string> graph_package_nodes;
        std::vector<std::string> graph_implementations;
        std::vector<std::string> runtime_package_nodes;
        std::set<std::string> artifact_names;
        // O contrato do modulo Wasm vem dos entrypoints, nao do nome da
        // build_package nem da linguagem escolhida para compila-la.
        const auto public_exports = manifest_exports(resolution);

        // Uma build_package citada num `extends` nao e um artefato: ela e uma parte
        // de outro. Construi-la em separado gastaria um build inteiro (o Wasm
        // do decoder do Qwen, por exemplo) para produzir um arquivo que
        // ninguem consome, e no runtime Wasm ainda disputaria a unica vaga de
        // orchestration do .mln.
        std::set<std::string> absorbed;
        for (const Package::Resolved::Package &package : resolution.packages())
            for (const auto &[package_name, build_package] : package.project.build_packages())
                for (const std::string &reference : build_package.extends)
                    absorbed.insert(dereference(resolution, package, build_package, reference)
                                        .package->project.metadata()
                                        .name +
                                    ":" + reference.substr(reference.find(':') + 1));

        for (const Package::Resolved::Package &package : resolution.packages()) {
            const std::string &name = package.project.metadata().name;

            for (const auto &[package_name, build_package] : package.project.build_packages()) {
                if (absorbed.contains(name + ":" + package_name))
                    continue;
                const Toolchain::Runtime runtime = runtime_of(build_package, name);
                const Toolchain::Output output = output_of(runtime, build_package);

                Toolchain::Selection selection = toolchains.select(
                    runtime, build_package.language,
                    {build_package.toolchain,
                     std::string(resolution.root().project.toolchain_for(
                         Toolchain::runtime_name(runtime), build_package.language))});
                const Toolchain::Driver &driver = Toolchain::driver(build_package.language);
                driver.probe(selection.toolchain);

                const std::string file = build_package.artifact.empty()
                                             ? artifact_name(build_package, name, runtime)
                                             : build_package.artifact;
                if (!artifact_names.insert(file).second)
                    Errors::at(Error::Artifact::Name::Conflict, name + ".package." + package_name,
                               file);

                const std::filesystem::path artifact =
                    options.build_directory / Toolchain::runtime_name(runtime) / file;

                Toolchain::Request request;
                request.runtime = runtime;
                request.output = output;
                request.sources = discover_sources(package, build_package);
                std::vector<std::string> package_symbols;
                if (runtime == Toolchain::Runtime::Iree) {
                    const Generated::Glue glue =
                        generate_package_glue(package, build_package, options.build_directory);
                    request.sources.push_back(glue.source);
                    for (const std::string &exported : build_package.exports)
                        package_symbols.push_back(implementation_symbol(exported));
                }
                request.includes = include_paths(resolution, package, build_package,
                                                 build_package.extends.empty());
                request.libraries = build_package.libraries;
                append_unique(request.defines, build_package.defines);
                if (runtime == Toolchain::Runtime::Iree)
                    append_unique(request.defines,
                                  {"LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1"});
                append_unique(request.defines,
                              configuration_defines(resolution, package, build_package.name));
                for (const std::filesystem::path &declared : build_package.library_directories) {
                    const std::filesystem::path entry(
                        expand_environment(declared.string(), name + ".package." + package_name +
                                                                  ".library_directories"));
                    request.library_directories.push_back(
                        entry.is_absolute() ? entry : package.project.root() / entry);
                }
                // Absorve as build_packages citadas em `extends`. Os sources vem
                // resolvidos contra a raiz de CADA package de origem, entao o
                // projeto compoe sem escrever caminho nenhum -- e os defines
                // declarados aqui valem para tudo que entrou, que e como se
                // parametriza um package sem edita-lo.
                std::vector<std::string> extended_uses;
                for (const std::string &reference : build_package.extends) {
                    const Extends::Target target =
                        dereference(resolution, package, build_package, reference);
                    append_unique(request.sources,
                                  discover_sources(*target.package, *target.build_package));
                    append_unique(request.includes, include_paths(resolution, *target.package,
                                                                  *target.build_package));
                    append_unique(request.libraries, target.build_package->libraries);
                    append_unique(request.defines, target.build_package->defines);
                    append_unique(request.defines,
                                  configuration_defines(resolution, *target.package,
                                                        target.build_package->name));
                    for (const std::filesystem::path &entry :
                         target.build_package->library_directories)
                        append_unique(
                            request.library_directories,
                            {entry.is_absolute() ? entry : target.package->project.root() / entry});
                    for (const std::string &used : target.build_package->uses)
                        extended_uses.push_back(target.package->project.metadata().name + ":" +
                                                used);
                }

                request.artifact = artifact;
                request.working_directory = package.project.root();
                request.optimize = options.optimize;

                // `uses` liga artefatos, nao apenas a ordem do DAG. O
                // consumidor deve procurar a biblioteca no diretorio de
                // outputs normalizado pelo Synapse; apontar para target/ do
                // Cargo mistura triples e reutiliza archives do host num
                // cross-build.
                const auto add_used_artifact_directory = [&](const std::string &reference) {
                    const std::string qualified = reference.find(':') == std::string::npos
                                                      ? name + ":" + reference
                                                      : reference;
                    const Extends::Target target =
                        dereference(resolution, package, build_package, qualified);
                    request.library_directories.push_back(
                        options.build_directory /
                        Toolchain::runtime_name(runtime_of(
                            *target.build_package, target.package->project.metadata().name)));
                };
                for (const std::string &used : build_package.uses)
                    add_used_artifact_directory(used);
                for (const std::string &used : extended_uses)
                    add_used_artifact_directory(used);

                if (request.sources.empty())
                    Errors::at(Error::Build::Package::Without::Sources,
                               name + ".package." + package_name);

                Node node;
                node.name = node_name(name, build_package);
                for (const std::string &used : build_package.uses)
                    node.dependencies.push_back(name + ":" + used);
                // O `uses` de uma build_package absorvida continua valendo: o
                // artefato que ela precisava (um staticlib de Rust, por
                // exemplo) agora e insumo de quem a absorveu.
                for (const std::string &used : extended_uses)
                    node.dependencies.push_back(used);
                node.inputs = request.sources;
                // Wrappers de SDK (Yocto e outros cross environments) fazem
                // parte da identidade da toolchain. Hash apenas do path
                // reutilizaria o cache depois que o SDK fosse atualizado no
                // mesmo local.
                if (std::filesystem::is_regular_file(selection.toolchain.compiler))
                    node.inputs.push_back(selection.toolchain.compiler);
                // Headers do proprio component/package tambem sao source. O
                // cache antigo so via .cpp e podia reutilizar um plugin mesmo
                // depois de mudar seu codec em include/. Dependencias externas
                // continuam representadas pelos nos que as produzem.
                const std::filesystem::path public_headers =
                    package.project.root() / package.project.layout().includes;
                if (std::filesystem::exists(public_headers))
                    node.inputs.push_back(public_headers);
                const auto add_declared_headers = [&](const Package::Resolved::Package &owner,
                                                      const auto &directories) {
                    for (const std::filesystem::path &entry : directories) {
                        const std::filesystem::path headers =
                            entry.is_absolute() ? entry : owner.project.root() / entry;
                        if (std::filesystem::exists(headers) &&
                            std::ranges::find(node.inputs, headers) == node.inputs.end())
                            node.inputs.push_back(headers);
                    }
                };
                add_declared_headers(package, build_package.includes);
                add_declared_headers(package, package.project.metadata().include_directories);
                for (const Package::Dependency::Edge &edge : package.dependencies) {
                    const Package::Resolved::Package &dependency =
                        resolution.packages().at(edge.index);
                    const std::filesystem::path dependency_headers =
                        dependency.project.root() / dependency.project.layout().includes;
                    if (std::filesystem::exists(dependency_headers) &&
                        std::ranges::find(node.inputs, dependency_headers) == node.inputs.end())
                        node.inputs.push_back(dependency_headers);
                    add_declared_headers(dependency,
                                         dependency.project.metadata().include_directories);
                }
                for (const std::string &reference : build_package.extends) {
                    const Extends::Target target =
                        dereference(resolution, package, build_package, reference);
                    const std::filesystem::path headers =
                        target.package->project.root() / target.package->project.layout().includes;
                    if (std::filesystem::exists(headers))
                        node.inputs.push_back(headers);
                }
                if (build_package.language == "c" || build_package.language == "cpp" ||
                    build_package.language == "c++") {
                    // APIs publicas do MyeLiN e do Compiler tambem sao inputs.
                    // Uma mudanca de namespace ou assinatura precisa invalidar
                    // o package mesmo quando seus sources nao mudaram.
                    const std::filesystem::path myelin_headers =
                        Toolchain::Myelin::root() / "include" / "mln";
                    if (std::filesystem::exists(myelin_headers) &&
                        std::ranges::find(node.inputs, myelin_headers) == node.inputs.end())
                        node.inputs.push_back(myelin_headers);
                    // The C++ driver adds this include path implicitly. Its
                    // package interfaces affect plugin ABI even when sources
                    // and explicitly declared include paths are unchanged.
                    const std::filesystem::path compiler_headers =
                        Toolchain::Sdk::compiler() / "syn";
                    if (std::filesystem::exists(compiler_headers) &&
                        std::ranges::find(node.inputs, compiler_headers) == node.inputs.end())
                        node.inputs.push_back(compiler_headers);
                    // SDK sources are compiled into graph/Wasm plugins too.
                    if (Toolchain::Abi::requirements(runtime, output).mln_sdk_core)
                        node.inputs.push_back(Toolchain::Myelin::root() / "src/sdk");
                    if (runtime == Toolchain::Runtime::Iree)
                        for (const auto &header : Toolchain::Sdk::headers())
                            node.inputs.push_back(header);
                    for (const std::filesystem::path &include : request.includes)
                        for (std::string_view api : {"mln", "syn"}) {
                            const std::filesystem::path headers = include / api;
                            if (std::filesystem::exists(headers) &&
                                std::ranges::find(node.inputs, headers) == node.inputs.end())
                                node.inputs.push_back(headers);
                        }
                }
                node.outputs = {artifact};
                // Tudo que muda o artefato entra no fingerprint. Defines,
                // libraries e includes mudam o binario tanto quanto os
                // sources; deixa-los de fora faria o cache servir um
                // artefato velho sem avisar.
                node.configuration = {
                    std::string(Toolchain::runtime_name(runtime)), build_package.language,
                    selection.toolchain.compiler.string(),         selection.toolchain.triple,
                    options.optimize ? "release" : "debug",
                };
                const Toolchain::Abi::Requirements abi =
                    Toolchain::Abi::requirements(runtime, output);
                for (const std::string &define : abi.defines)
                    node.configuration.push_back("D" + define);
                if (runtime == Toolchain::Runtime::Iree)
                    node.configuration.push_back("DLLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING=1");
                for (const std::string &define : request.defines)
                    node.configuration.push_back("D" + define);
                for (const std::string &library : request.libraries)
                    node.configuration.push_back("l" + library);
                for (const std::filesystem::path &include : request.includes)
                    node.configuration.push_back("I" + include.string());
                for (const std::filesystem::path &directory : request.library_directories)
                    node.configuration.push_back("L" + directory.string());
                const std::string origin = name + ".package." + package_name;
                std::vector<std::string> required =
                    Toolchain::Abi::requirements(runtime, output).required_symbols;
                append_unique(required, package_symbols);
                if (runtime == Toolchain::Runtime::Wasm) {
                    // Num modulo Wasm o que o package lista em `exports` sao
                    // nomes de funcao de verdade; nos outros runtimes sao
                    // exports Myelin, registrados em runtime pela ABI e nao
                    // visiveis na tabela de simbolos.
                    append_unique(required, build_package.exports);
                    if (const auto manifest = public_exports.find(node_name(name, build_package));
                        manifest != public_exports.end())
                        append_unique(required, manifest->second);
                }
                node.action = [&driver, selection, request, required, origin] {
                    driver.build(selection.toolchain, request);
                    // Ler o artefato e o que substitui confiar no compilador.
                    // Falhar aqui custa um build de build_package; deixar passar
                    // custa o empacotamento inteiro e um .mln que o Runtime
                    // rejeita ao carregar.
                    Toolchain::Symbols::require(request.artifact, required, origin);
                };
                result.graph.add(std::move(node));
                if (runtime == Toolchain::Runtime::Iree)
                    graph_package_nodes.push_back(node_name(name, build_package));
                else
                    runtime_package_nodes.push_back(node_name(name, build_package));

                // O artefato entra no plano conforme o runtime: plugins de
                // host alimentam o compiler, providers e orchestration entram
                // no .mln.
                switch (runtime) {
                case Toolchain::Runtime::Iree:
                    result.artifacts.plugins.push_back(artifact);
                    if (!build_package.graph_namespace.empty())
                        for (const std::string &export_name : build_package.exports)
                            graph_implementations.push_back(name + ":" + export_name);
                    break;
                case Toolchain::Runtime::Native:
                    // Um archive estatico e insumo de build, nao provider:
                    // ele e linkado DENTRO de um .so e nao tem
                    // mln_native_module para o registry resolver.
                    // Empacota-lo somaria dezenas de MB de peso morto e
                    // faria o runtime tentar dlopen num .a.
                    if (artifact.extension() != ".a")
                        result.artifacts.modules.push_back(
                            {"native", node_name(name, build_package), artifact});
                    break;
                case Toolchain::Runtime::Driver:
                    if (artifact.extension() != ".a")
                        result.artifacts.modules.push_back(
                            {"driver", node_name(name, build_package), artifact});
                    break;
                case Toolchain::Runtime::Wasm:
                    result.artifacts.modules.push_back(
                        {"wasm", node_name(name, build_package), artifact});
                    break;
                }
            }
        }

        // O manifest referencia codigo pelo nome que os packages declararam em
        // `exports`. Conferir aqui e barato; deixar passar significa carregar
        // os pesos, comecar a compilar o grafo e so entao descobrir que
        // `qwen3.5:decodr` nao existe.
        {
            const Config::Value &document = resolution.root().project.document();
            const std::set<std::string> available = declared_exports(resolution);
            const auto check = [&](const Config::Value &reference, const std::string &origin) {
                std::string name{reference.string()};
                // Sem `:` a referencia nomeia o export homonimo do package.
                if (name.find(':') == std::string::npos)
                    name += ":" + name;
                if (!available.contains(name))
                    Errors::at(Error::Export::Unknown, origin, reference.string());
            };
            if (const Config::Value *myelin = document.find("myelin"))
                if (const Config::Value *formats = myelin->find("formats");
                    formats != nullptr && formats->is_array())
                    for (const Config::Value &format : formats->array())
                        check(format, "myelin.formats");
            // Os codecs das policies sao o ponto onde um nome errado sai mais
            // caro: o Compiler so chega neles depois de abrir o checkpoint.
            if (const Config::Value *policies = document.find("policies");
                policies != nullptr && policies->is_table())
                for (const auto &[path, policy] : policies->table()) {
                    const Config::Value *schemes = policy.find("schemes");
                    if (schemes == nullptr || !schemes->is_array())
                        continue;
                    for (const Config::Value &scheme : schemes->array())
                        if (const Config::Value *format = scheme.find("format"))
                            check(*format, "policies." + path);
                }
        }

        // Raiz dos pesos. E do projeto, nao do package: o mesmo modelo
        // portado serve a checkpoints diferentes (base, fine-tune), e a
        // escolha pertence a quem monta o .mln.
        std::vector<std::filesystem::path> weight_sources;
        if (const Config::Value *weights = resolution.root().project.document().find("weights")) {
            const Config::Value *path = weights->find("path");
            const std::filesystem::path root = resolution.root().project.root();
            const Config::Value *paths = weights->find("paths");
            if ((path == nullptr) == (paths == nullptr))
                Errors::at(Error::Weights::Path::Selection::Invalid, "weights");
            const auto add = [&](std::string_view value) {
                const std::filesystem::path entry{expand_environment(value, "weights.path")};
                weight_sources.push_back(entry.is_absolute() ? entry : root / entry);
            };
            if (path != nullptr)
                add(path->string());
            else {
                if (!paths->is_array() || paths->array().empty())
                    Errors::at(Error::Weights::Paths::Empty, "weights.paths");
                for (const Config::Value &entry : paths->array())
                    add(entry.string());
            }
            for (const std::filesystem::path &source : weight_sources) {
                std::error_code missing;
                if (std::filesystem::exists(source, missing))
                    continue;
                Errors::at(Error::Path::Missing, "weights.path", source);
            }

            result.artifacts.weights = weight_sources;
        }

        // Assets do projeto raiz, por convencao de layout.
        const std::filesystem::path assets =
            resolution.root().project.root() / resolution.root().project.layout().assets;
        std::error_code error;
        if (std::filesystem::exists(assets, error)) {
            std::error_code walk;
            for (const auto &item : std::filesystem::recursive_directory_iterator(assets, walk))
                if (item.is_regular_file())
                    result.artifacts.assets.emplace_back(
                        item.path().lexically_relative(assets).string(), item.path());
            // Assets que nao entram no pacote nao quebram o build: o .mln sai
            // sem o tokenizer e falha no carregamento, longe daqui.
            if (walk)
                Errors::at(Error::File::Unreadable, "layout.assets", assets, walk.message());
        }

        // Grafo, parametros e packaging sao nos diferentes. Estes caches sao
        // internos e nenhum deles e um .mln deployavel isoladamente.
        Compiler::Artifacts artifacts = result.artifacts;
        const std::filesystem::path output = result.package;
        const std::filesystem::path graph_cache =
            options.build_directory / ".myelin" / "graph.cache";
        const std::filesystem::path requirements_graph_cache =
            options.build_directory / ".myelin" / "requirements-graph.cache";
        const std::filesystem::path parameters_cache =
            options.build_directory / ".myelin" / "parameters.cache";
        const std::filesystem::path weight_requirements =
            options.build_directory / ".myelin" / "weights.requirements";
        const std::filesystem::path parameters_requirements =
            options.build_directory / ".myelin" / "parameters.requirements";
        const std::filesystem::path api = options.compiler_api;
        const Package::Resolution *resolved = &resolution;
        const std::optional<std::filesystem::path> tuning = Compiler::tuning_spec(*resolved);
        if (tuning && !std::filesystem::is_regular_file(*tuning))
            Errors::at(Error::Path::Missing, "myelin.tuning", *tuning);

        Node graph_node;
        graph_node.name = "myelin:graph";
        graph_node.dependencies = graph_package_nodes;
        graph_node.inputs = {api};
        if (tuning)
            graph_node.inputs.push_back(*tuning);
        graph_node.outputs = {weight_requirements};
        const Config::Value graph_plan =
            Compiler::Plan::graph(*resolved, artifacts, weight_requirements);
        graph_node.configuration = {api.string(), Config::json(graph_plan)};
        graph_node.action = [resolved, artifacts, requirements_graph_cache, weight_requirements,
                             api] {
            std::error_code failure;
            std::filesystem::create_directories(requirements_graph_cache.parent_path(), failure);
            if (failure)
                Errors::raise(Error::Directory::Not::Created,
                              requirements_graph_cache.parent_path(), failure.message());
            // This phase discovers the weight contract only. Executable
            // serialization happens in myelin:final-graph after parameters
            // exist, which is required by backends embedding constants.
            Compiler::Api::load(api).compile(
                Compiler::Plan::graph(*resolved, artifacts, weight_requirements),
                requirements_graph_cache);
        };
        result.graph.add(std::move(graph_node));

        Node parameters_node;
        parameters_node.name = "myelin:parameters";
        parameters_node.order_only_dependencies = graph_package_nodes;
        parameters_node.order_only_dependencies.push_back("myelin:graph");
        parameters_node.inputs = {api, weight_requirements};
        for (const std::filesystem::path &plugin : artifacts.plugins)
            parameters_node.inputs.push_back(plugin);
        parameters_node.outputs = {parameters_cache, parameters_requirements};
        parameters_node.configuration = {api.string(),
                                         Config::json(Compiler::weights_fingerprint(*resolved)),
                                         weight_signature(weight_sources)};
        parameters_node.action = [resolved, artifacts, weight_requirements, parameters_cache,
                                  parameters_requirements, api, options] {
            if (std::filesystem::exists(parameters_cache) &&
                std::filesystem::exists(parameters_requirements) &&
                System::hash_file(weight_requirements) ==
                    System::hash_file(parameters_requirements))
                return;
            Compiler::Api::load(api).compile(
                Compiler::Plan::weights(*resolved, artifacts, weight_requirements, options.jobs,
                                        options.weight_conversion_memory_bytes),
                parameters_cache);
            std::filesystem::copy_file(weight_requirements, parameters_requirements,
                                       std::filesystem::copy_options::overwrite_existing);
        };
        result.graph.add(std::move(parameters_node));

        std::vector<std::pair<std::string, std::filesystem::path>> graph_shards;
        if (graph_implementations.empty()) {
            graph_shards.emplace_back("model", graph_cache);
            Node final_graph_node;
            final_graph_node.name = "myelin:final-graph";
            final_graph_node.dependencies = graph_package_nodes;
            final_graph_node.dependencies.push_back("myelin:parameters");
            final_graph_node.inputs = {api, parameters_cache};
            if (tuning)
                final_graph_node.inputs.push_back(*tuning);
            final_graph_node.outputs = {graph_cache};
            const Config::Value final_graph_plan =
                Compiler::Plan::graph(*resolved, artifacts, {}, parameters_cache);
            final_graph_node.configuration = {api.string(), Config::json(final_graph_plan)};
            final_graph_node.action = [resolved, artifacts, graph_cache, parameters_cache, api] {
                Compiler::Api::load(api).compile(
                    Compiler::Plan::graph(*resolved, artifacts, {}, parameters_cache), graph_cache);
            };
            result.graph.add(std::move(final_graph_node));
        } else {
            for (const std::string &implementation : graph_implementations) {
                const std::string shard = symbol_identifier(implementation);
                const std::filesystem::path shard_cache =
                    options.build_directory / ".myelin" / ("graph." + shard + ".cache");
                graph_shards.emplace_back(shard, shard_cache);
                Node node;
                node.name = "myelin:graph:" + shard;
                node.dependencies = graph_package_nodes;
                node.dependencies.push_back("myelin:parameters");
                node.inputs = {api, parameters_cache};
                if (tuning)
                    node.inputs.push_back(*tuning);
                node.outputs = {shard_cache};
                const Config::Value plan = Compiler::Plan::graph(*resolved, artifacts, {},
                                                                 parameters_cache, implementation);
                node.configuration = {api.string(), Config::json(plan)};
                node.action = [resolved, artifacts, shard_cache, parameters_cache, api,
                               implementation] {
                    Compiler::Api::load(api).compile(Compiler::Plan::graph(*resolved, artifacts, {},
                                                                           parameters_cache,
                                                                           implementation),
                                                     shard_cache);
                };
                result.graph.add(std::move(node));
            }

            Node merge;
            merge.name = "myelin:final-graph";
            for (const auto &[name, path] : graph_shards) {
                merge.dependencies.push_back("myelin:graph:" + name);
                merge.inputs.push_back(path);
            }
            merge.inputs.push_back(api);
            merge.outputs = {graph_cache};
            merge.configuration = {api.string(),
                                   Config::json(Compiler::Plan::merge_graphs(graph_shards))};
            merge.action = [graph_shards, graph_cache, api] {
                Compiler::Api::load(api).compile(Compiler::Plan::merge_graphs(graph_shards),
                                                 graph_cache);
            };
            result.graph.add(std::move(merge));
        }

        Node package_node;
        package_node.name = "package";
        package_node.dependencies = runtime_package_nodes;
        package_node.dependencies.push_back("myelin:final-graph");
        package_node.dependencies.push_back("myelin:parameters");
        package_node.inputs = {api};
        package_node.outputs = {output};
        // Assets entram como INPUT, nao so como configuracao: trocar o
        // conteudo de um asset muda o .mln, e sem isso o no ficaria em cache
        // com o artefato velho -- falha silenciosa, o pior tipo aqui.
        //
        // Os pesos ficam de fora de proposito: sao dezenas de GB e o hash
        // custaria mais que o proprio empacotamento. O caminho deles entra na
        // configuracao, entao trocar de checkpoint ainda invalida.
        for (const auto &[name, path] : artifacts.assets)
            package_node.inputs.push_back(path);
        package_node.configuration = {output.string(), api.string(), graph_cache.string(),
                                      parameters_cache.string(),
                                      Config::json(Compiler::Plan::package(
                                          *resolved, artifacts, graph_cache, parameters_cache))};
        for (const Compiler::Runtime::Module &module : artifacts.modules) {
            package_node.inputs.push_back(module.path);
            package_node.configuration.push_back(module.runtime + ":" + module.name + ":" +
                                                 module.path.string());
        }
        package_node.action = [resolved, artifacts, output, graph_cache, parameters_cache, api] {
            std::error_code failure;
            std::filesystem::create_directories(output.parent_path(), failure);
            if (failure)
                Errors::raise(Error::Directory::Not::Created, output.parent_path(),
                              failure.message());
            const Config::Value plan =
                Compiler::Plan::package(*resolved, artifacts, graph_cache, parameters_cache);
            Compiler::Api::load(api).compile(plan, output);
        };
        result.graph.add(std::move(package_node));

        return result;
    }
} // namespace Syn::Build
