#include "syn/error.hpp"

#include <print>

namespace Syn {

    std::string_view Errors::pattern(Error::Code error) {
        switch (error) {
        case Error::Value::Table::Expected:
            return "expected table, received {}";
        case Error::Value::Array::Expected:
            return "expected array, received {}";
        case Error::Value::String::Expected:
            return "expected string, received {}";
        case Error::Value::Integer::Expected:
            return "expected integer, received {}";
        case Error::Value::Number::Expected:
            return "expected number, received {}";
        case Error::Value::Boolean::Expected:
            return "expected boolean, received {}";
        case Error::Required::Missing:
            return "required {} is missing";
        case Error::Section::Renamed:
            return "`{}` was replaced by `{}`";
        case Error::Toml::Invalid:
            return "invalid TOML: {}";
        case Error::Path::Missing:
            return "`{}` does not exist";
        case Error::File::Write::Failed:
            return "could not write `{}`";
        case Error::Variable::Out::Of::Bounds:
            return "value is {} the configured {}";

        case Error::Package::Manifest::Missing:
            return "package does not contain synps.toml";
        case Error::Dependency::Cycle:
            return "dependency cycle: {}";
        case Error::Capability::Unprovided:
            return "package `{}` requires capability `{}`, but no package "
                   "provides it";
        case Error::Capability::Ambiguous:
            return "capability `{}` has multiple providers; select one in "
                   "[bindings]";
        case Error::Capability::Not::Provided:
            return "selected package `{}` does not provide this capability";
        case Error::Registry::Unavailable:
            return "registry dependency `{}` cannot be resolved because no "
                   "registry is configured";
        case Error::Configuration::Unknown::Dependency:
            return "configuration refers to an unknown dependency";

        case Error::Myelin::Root::Unresolved:
            return "invalid Myelin root: `{}`; configure MLN_ROOT or "
                   "rebuild SYNaPSe with -DMLN_ROOT=<path>";
        case Error::Language::Without::Driver:
            return "language `{}` has no driver; known languages: {}";
        case Error::Toolchain::Not::Declared:
            return "toolchain `{}` is not declared in [toolchains]";
        case Error::Toolchain::Language::Mismatch:
            return "toolchain `{}` uses language `{}`, but the component uses "
                   "`{}`";
        case Error::Toolchain::Unavailable:
            return "no toolchain for `{}` on runtime `{}`; declare one in "
                   "[toolchains] or install the compiler";
        case Error::Toolchain::Compiler::Missing:
            return "{} toolchain has no `compiler`";
        case Error::Compiler::Unresponsive:
            return "compiler `{}` did not respond to `{}`";
        case Error::Output::Unsupported:
            return "{} driver cannot produce this artifact for runtime `{}`";
        case Error::Sources::Missing:
            return "build package {} has no {}";
        case Error::Artifact::Not::Produced:
            return "cargo did not produce `{}`";
        case Error::Go::Requires::Tinygo:
            return "Go orchestration requires tinygo: standard `go` emits "
                   "Wasm with its own runtime, which is incompatible with the Myelin ABI";
        case Error::Toolchain::Ambiguous:
            return "more than one toolchain is declared for `{}`; select one "
                   "in the component";
        case Error::Entrypoint::Missing:
            return "a .mln artifact requires at least one executable entrypoint; "
                   "declare [entrypoint].package";
        case Error::Entrypoint::Dependency::Missing:
            return "package `{}` was selected as the entry but declares no entrypoint";
        case Error::Entrypoint::Dependency::Unknown:
            return "entrypoint references unknown dependency `{}`";
        case Error::Entrypoint::Path::Empty:
            return "entrypoint `{}` has an empty path";
        case Error::Entrypoint::Path::Unknown:
            return "package `{}` publishes no contract for entry path `{}`";
        case Error::Entrypoint::Duplicated:
            return "entrypoint `{}` is declared twice";
        case Error::Entrypoint::Export::Conflict:
            return "two entrypoints use the same symbol `{}` in module `{}`";
        case Error::Name::Duplicated:
            return "{}: duplicate name `{}`";
        case Error::Entrypoint::Module::Unknown:
            return "entrypoint references missing Wasm module `{}`";
        case Error::Entrypoint::Module::Missing:
            return "entrypoint has no `module`; use `project:package`";
        case Error::Hook::Function::Format::Invalid:
            return "hook function `{}` must use the `project:package` format";
        case Error::Hook::Function::Missing:
            return "hook at `{}` must declare `function`";
        case Error::Hook::Module::Unknown:
            return "hook references missing Wasm module `{}`";
        case Error::Hook::Package::Unknown:
            return "hook references unknown package `{}`";
        case Error::Compiler::Plan::Manifest::Array::Expected:
            return "compiler plan manifest section must be an array";
        case Error::Compiler::Plan::Options::Array::Expected:
            return "compiler plan option must be an array";

        case Error::Runtime::Unknown:
            return "unknown runtime `{}`; Myelin supports iree, native, and wasm";
        case Error::Artifact::Name::Conflict:
            return "two packages produce `{}`";
        case Error::Build::Package::Without::Sources:
            return "package has no sources";
        case Error::Extends::Format::Invalid:
            return "extends `{}` does not use the `project:package` format";
        case Error::Extends::Project::Unknown:
            return "extends `{}`: `{}` is not listed in [dependencies]";
        case Error::Extends::Package::Unknown:
            return "extends `{}`: project `{}` does not declare that package";
        case Error::Extends::Runtime::Mismatch:
            return "extends `{}`: runtime `{}` does not match `{}`";
        case Error::Exports::Missing:
            return "`{}` does not export the symbols resolved by the runtime: {}";
        case Error::Export::Unknown:
            return "`{}` is not exported by any package in the resolved projects";

        case Error::Target::Exists:
            return "`{}` already exists";
        case Error::Dependency::Exists:
            return "`{}` is already a dependency";
        case Error::Dependency::Source::Required:
            return "no remote registry is configured; provide --path or --git";
        case Error::Dependency::Unknown:
            return "`{}` is not a dependency";
        case Error::Build::Directory::Not::Removed:
            return "could not remove `{}`: {}";
        case Error::Cli::Usage:
            return "use: {}";
        case Error::Option::Requires::Value:
            return "option `--{}` requires a value";
        case Error::Option::Unknown:
            return "unknown option `--{}`; see `synps help {}`";

        case Error::Weights::Path::Selection::Invalid:
            return "[weights] must declare exactly one of `path` or `paths`";
        case Error::Environment::Variable::Unset:
            return "environment variable `{}` is not set";
        case Error::Directory::Not::Created:
            return "could not create `{}`: {}";
        case Error::Weights::Paths::Empty:
            return "weights.paths must be a non-empty array";
        case Error::Environment::Reference::Unterminated:
            return "unterminated environment reference";
        case Error::Environment::Reference::Empty:
            return "empty environment reference";

        case Error::Node::Name::Empty:
            return "build node name is empty";
        case Error::Node::Duplicated:
            return "duplicate build node";
        case Error::Graph::Cycle:
            return "cycle in build graph";
        case Error::Node::Dependency::Unknown:
            return "unknown build dependency";
        case Error::Node::Input::Missing:
            return "build input does not exist";
        case Error::Node::Output::Missing:
            return "build action did not produce declared output";
        case Error::Node::Without::Action:
            return "build node has no action";

        case Error::Compiler::Api::Unloadable:
            return "cannot load Myelin Compiler API: {}";
        case Error::Compiler::Api::Symbol::Missing:
            return "library does not export the Myelin Compiler API: {}";
        case Error::Compiler::Api::Incompatible:
            return "incompatible Myelin Compiler API v1";
        case Error::Compilation::Failed:
            return "Myelin compilation failed: {}";
        case Error::Value::Unsupported:
            return "TOML date/time values are not supported in Synapse "
                   "metadata";
        case Error::Manifest::Not::Found:
            return "synps.toml not found";
        case Error::Dependency::Source::Invalid:
            return "expected dependency table or version string";
        case Error::Dependency::Source::Ambiguous:
            return "expected exactly one source: path, git, or version";
        case Error::Build::Domain::Unknown:
            return "unknown build domain; expected host, native, or wasm";
        case Error::Build::Domain::Empty:
            return "expected `toolchain` or a language mapping";
        case Error::Build::Domain::Duplicated:
            return "build domain was configured more than once";
        case Error::Manifest::Field::String::Expected:
            return "manifest field must be a string";
        case Error::Manifest::Field::Array::Expected:
            return "manifest field must be an array";
        case Error::Manifest::Field::Array::Element::String::Expected:
            return "manifest array element must be a string";
        case Error::Manifest::Table::Expected:
            return "manifest section must be a table";
        case Error::Toolchain::Table::Expected:
            return "toolchain declaration must be a table";
        case Error::Build::Domain::Table::Expected:
            return "build domain declaration must be a table";
        case Error::Build::Toolchain::Name::Expected:
            return "build toolchain selection must be a string";
        case Error::Build::Toolchain::Unknown:
            return "build selects unknown toolchain `{}`";
        case Error::Package::Table::Expected:
            return "build package declaration must be a table";
        case Error::Binding::Package::Name::Expected:
            return "capability binding must name a package with a string";
        case Error::Schema::Type::Unknown:
            return "unknown schema type `{}`";
        case Error::Schema::Variable::Unknown:
            return "package does not declare variable `{}`";
        case Error::Schema::Value::Type::Mismatch:
            return "schema expected {}, received {}";
        case Error::Schema::Type::Missing:
            return "schema field is missing its type";
        case Error::Variables::Undeclared:
            return "package does not declare configurable fields";
        case Error::Variable::Out::Of::Range:
            return "expected uint32, received value outside its range";
        case Error::Lockfile::Version::Unsupported:
            return "unsupported lockfile version";
        case Error::Lockfile::Package::Array::Expected:
            return "lockfile package array is missing or invalid";
        case Error::Lockfile::Package::Table::Expected:
            return "lockfile package entry must be a table";
        case Error::Lockfile::Dependency::Table::Expected:
            return "lockfile dependency entry must be a table";
        case Error::Lockfile::Field::String::Expected:
            return "lockfile field must be a string";

        case Error::Process::Pipe::Failed:
            return "cannot create process pipe: {}";
        case Error::Process::Executable::Empty:
            return "process executable is empty";
        case Error::Process::Start::Failed:
            return "cannot start process: {}";
        case Error::Process::Wait::Failed:
            return "cannot wait for process `{}`: {}";
        case Error::Process::Failed:
            return "{} failed with exit code {}{}";
        case Error::File::Unreadable:
            return "could not read `{}`: {}";
        }
        return "unknown error";
    }

    std::string Errors::code(Error::Code error) {
        return std::format("E{}", std::to_underlying(error));
    }

    std::string Errors::vcompose(Error::Code error, std::format_args arguments) {
        const std::string_view text = pattern(error);
        try {
            return std::vformat(text, arguments);
        } catch (const std::format_error &) {

            return std::string(text);
        }
    }

    void Errors::dispatch(Error::Code error, std::string location, std::string message) {
        throw Diagnostic(error, std::move(message), std::move(location));
    }

    void Errors::report(const Diagnostic &diagnostic, std::FILE *stream) {
        std::println(stream, "{}", diagnostic.what());
    }

    namespace {
        std::string format(Error::Code error, const std::string &message,
                           const std::string &location) {
            return location.empty()
                       ? std::format("error[{}]: {}", Errors::code(error), message)
                       : std::format("error[{}]: {}: {}", Errors::code(error), location, message);
        }
    }

    Diagnostic::Diagnostic(Error::Code error, std::string message, std::string location)
        : std::runtime_error(format(error, message, location)), error_(error),
          message_(std::move(message)), location_(std::move(location)) {}

    std::string Diagnostic::code() const { return Errors::code(error_); }
}
