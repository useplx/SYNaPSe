#include "syn/package.hpp"

#include <memory>
#include <stdexcept>

namespace {
    class Graph final : public Mln::Sdk::Module {};

    constexpr auto factory = [] { return std::make_unique<Graph>(); };
}

int main() {
    using namespace Syn::Compiler::Package;
    const Object *object = published_object<Graph, factory>();
    const auto *create = std::get_if<GraphFactory>(object);
    if (!create)
        throw std::runtime_error("graph export was not published as a factory");

    auto first = (*create)();
    auto second = (*create)();
    if (!first || !second || first.get() == second.get())
        throw std::runtime_error("graph factory shared a module instance");

    first->name = "mutated";
    if (!second->name.empty())
        throw std::runtime_error("graph factory shared mutable module state");
    return 0;
}
