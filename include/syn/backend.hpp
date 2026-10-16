#pragma once

#include "mln/sdk/backend/compatibility.hpp"

#include <mutex>
#include <span>
#include <string>

namespace Syn::Compiler {
    class BackendResources {
        public:
            BackendResources(std::string_view identity,
                             std::span<const Mln::Sdk::Backend::Resource> resources);
            BackendResources(const BackendResources &) = delete;
            BackendResources &operator=(const BackendResources &) = delete;
            ~BackendResources();

        private:
            bool owner_ = false;
    };

    void register_backends(std::span<Mln::Sdk::Backend::Compatibility *const> compatibilities);

    std::string runtime_device_uri(const Mln::Sdk::Backend &backend);
}
