#include "syn/package/resolver.hpp"

#include "syn/error.hpp"

#include <algorithm>

namespace Syn::Package {
    std::optional<Resolution::Entrypoint> Resolution::entrypoint() const {
        const std::string &reference = root().project.metadata().entrypoint;
        if (reference.empty())
            return std::nullopt;

        if (reference == root().project.metadata().name)
            return Entrypoint{root_, reference};

        const auto dependency =
            std::ranges::find(root().dependencies, reference, &Dependency::Edge::name);
        if (dependency == root().dependencies.end())
            Errors::at(Error::Entrypoint::Dependency::Unknown, "entrypoint.package", reference);

        const auto &selected = packages_.at(dependency->index);
        return Entrypoint{dependency->index, selected.project.metadata().name};
    }
}
