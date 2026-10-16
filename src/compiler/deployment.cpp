#include "syn/deployment.hpp"
#include "mln/error.hpp"
#include "syn/errors/graph.hpp"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#include <format>
#include <mlir/IR/BuiltinAttributes.h>

#include <algorithm>
#include <string>
#include <vector>

namespace Syn::Compiler {
    namespace {
        std::string device_uri(std::string_view device, int ordinal) {
            if (device == "local")
                return "local-task://";
            if (device == "vulkan") {

                if (ordinal != 0) {
                    raise(Errors::Graph::Deployment::Vulkan::Ordinal::Unsupported);
                }
                return "vulkan://";
            }

            if (ordinal == 0)
                return std::format("{}://", device);
            return std::format("{}://{}", device, ordinal);
        }
    }

    std::string runtime_device_uri(const Mln::Sdk::Backend &backend) {
        const auto target = std::ranges::find_if(
            backend.flags, [](const auto &flag) { return flag.first == "iree-hal-target-device"; });
        if (target == backend.flags.end() || target->second.starts_with('#'))
            return device_uri(backend.device, backend.ordinal);

        std::string result;
        std::size_t begin = 0;
        while (begin <= target->second.size()) {
            const std::size_t separator = target->second.find(',', begin);
            const std::string_view device(
                target->second.data() + begin,
                (separator == std::string::npos ? target->second.size() : separator) - begin);
            if (!result.empty())
                result += ',';
            result += device_uri(device, backend.ordinal);
            if (separator == std::string::npos)
                break;
            begin = separator + 1;
        }
        return result;
    }

    namespace {
        bool same_device(const Mln::Sdk::Backend &left, const Mln::Sdk::Backend &right) {
            return left.device == right.device && left.ordinal == right.ordinal;
        }
    }

    std::vector<Mln::Sdk::Backend> deployment_backends(const Manifest &manifest) {
        std::vector<Mln::Sdk::Backend> backends;
        for (const auto &[path, police] : manifest.policies) {
            (void)path;
            if (!police.backend)
                continue;
            auto existing = std::find_if(backends.begin(), backends.end(),
                                         [&](const Mln::Sdk::Backend &backend) {
                                             return same_device(backend, *police.backend);
                                         });
            if (existing == backends.end()) {
                backends.push_back(*police.backend);
                continue;
            }
            for (const auto &flag : police.backend->flags) {
                if (std::find(existing->flags.begin(), existing->flags.end(), flag) ==
                    existing->flags.end()) {
                    existing->flags.push_back(flag);
                }
            }
        }
        return backends;
    }

    void apply_backend(mlir::Operation *operation, const Manifest &manifest, const Police &police) {
        if (!operation || !police.backend)
            return;
        mlir::MLIRContext *context = operation->getContext();
        const std::vector<Mln::Sdk::Backend> backends = deployment_backends(manifest);
        const auto selected = std::ranges::find_if(backends, [&](const Mln::Sdk::Backend &backend) {
            return same_device(backend, *police.backend);
        });
        if (selected == backends.end())
            raise(Errors::Graph::Deployment::Backend::Missing);
        const auto index = static_cast<std::size_t>(selected - backends.begin());
        const mlir::StringAttr device =
            mlir::StringAttr::get(context, std::format("device_{}", index));
        operation->setAttr(
            "stream.affinity",
            mlir::iree_compiler::IREE::HAL::DevicePromiseAttr::get(context, device, -1));
        operation->setAttr(
            "myelin.backend",
            mlir::StringAttr::get(
                context, std::format("{}:{}", police.backend->device, police.backend->ordinal)));
    }
}
