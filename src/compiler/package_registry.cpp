#include "syn/package_registry.hpp"
#include "mln/error.hpp"
#include "syn/errors/package.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <format>
#include <stdexcept>
#include <utility>

namespace Syn::Compiler {
    namespace {
        void close_library(void *library) {
            if (library)
                dlclose(library);
        }

        Package::Export::Kind checked_kind(uint32_t kind, std::string_view id) {
            switch (kind) {
            case mln_package_export_graph:
                return Package::Export::Kind::Graph;
            case mln_package_export_codec:
                return Package::Export::Kind::Codec;
            case mln_package_export_hook:
                return Package::Export::Kind::Hook;
            case mln_package_export_codec_compatibility:
                return Package::Export::Kind::CodecCompatibility;
            case mln_package_export_operation_compatibility:
                return Package::Export::Kind::OperationCompatibility;
            case mln_package_export_lowering_compatibility:
                return Package::Export::Kind::LoweringCompatibility;
            case mln_package_export_backend_compatibility:
                return Package::Export::Kind::BackendCompatibility;
            default:
                raise(Errors::Package::Registry::Export::Kind::Unknown, id, kind);
            }
        }
    }

    void Package::Registry::load(std::string_view path) {
        const std::string path_string(path);
        Library library(dlopen(path_string.c_str(), RTLD_NOW | RTLD_LOCAL), close_library);
        if (!library)
            raise(Errors::Package::Registry::Library::Load::Failed, path, dlerror());

        dlerror();
        void *symbol = dlsym(library.get(), "mln_package_register");
        if (const char *error = dlerror())
            raise(Errors::Package::Registry::Library::Registration::Symbol::Missing, path, error);

        const auto function = reinterpret_cast<Package::Abi::Register>(symbol);
        const Package::Abi::Registration *registration = function();
        if (!registration)
            raise(Errors::Package::Registry::Registration::Missing, path);
        if (registration->abi_version != Package::Abi::Version)
            raise(Errors::Package::Registry::Registration::Abi::Mismatch, path,
                  registration->abi_version, Package::Abi::Version);
        if (registration->struct_size < sizeof(*registration))
            raise(Errors::Package::Registry::Registration::Truncated, path);
        if (!registration->name || registration->name[0] == '\0' || !registration->version ||
            registration->version[0] == '\0')
            raise(Errors::Package::Registry::Registration::Identity::Missing, path);
        if (registration->export_count != 0 && !registration->exports)
            raise(Errors::Package::Registry::Registration::Exports::Missing, path);

        std::vector<Package::Export> pending;
        pending.reserve(registration->export_count);
        for (size_t index = 0; index < registration->export_count; ++index) {
            const Package::Abi::Export &item = registration->exports[index];
            if (item.struct_size < sizeof(item) || !item.name || item.name[0] == '\0' ||
                !item.invoke)
                raise(Errors::Package::Registry::Export::Invalid, index, path);

            Package::Export value{
                registration->name,
                registration->version,
                item.name,
                checked_kind(item.kind, std::format("{}:{}", registration->name, item.name)),
                item.invoke,
                item.user_data,
            };
            const std::string id = value.id();
            if (exports_.contains(id) ||
                std::any_of(pending.begin(), pending.end(),
                            [&](const Package::Export &other) { return other.id() == id; }))
                raise(Errors::Package::Registry::Export::Duplicated, id);
            pending.push_back(std::move(value));
        }

        for (Package::Export &item : pending)
            exports_.emplace(item.id(), std::move(item));
        libraries_.push_back(std::move(library));
    }

    const Package::Export *Package::Registry::resolve(Package::Export::Kind kind,
                                                      std::string_view id) const {
        std::string key(id);
        if (id.find(':') == std::string_view::npos) {
            key += ':';
            key.append(id);
        }
        const auto found = exports_.find(key);
        if (found == exports_.end() || found->second.kind != kind)
            return nullptr;
        return &found->second;
    }

    std::vector<Package::Export> Package::Registry::exports() const {
        std::vector<Package::Export> result;
        result.reserve(exports_.size());
        for (const auto &[id, item] : exports_)
            result.push_back(item);
        std::sort(result.begin(), result.end(),
                  [](const Package::Export &left, const Package::Export &right) {
                      return left.id() < right.id();
                  });
        return result;
    }

    void Package::Registry::pin() {
        static auto *const pinned = new std::vector<Library>;
        pinned->reserve(pinned->size() + libraries_.size());
        for (Library &library : libraries_)
            pinned->push_back(std::move(library));
        libraries_.clear();
    }

    Package::Registry::~Registry() {
        exports_.clear();

        while (!libraries_.empty())
            libraries_.pop_back();
    }
}
