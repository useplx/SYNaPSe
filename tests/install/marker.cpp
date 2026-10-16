#include <mln/ir.hpp>
#include <syn/package.hpp>

#include <stdexcept>

class Identity final : public Mln::Sdk::Module {
    public:
        Identity() { name = "model"; }
        std::vector<Mln::Sdk::Tensor> graph(std::span<const Mln::Sdk::Tensor> inputs) override {
            return {inputs[0]};
        }
        std::vector<std::vector<Mln::Sdk::Tensor>> specializations() const override {
            return {{Mln::Sdk::Tensor({4}, Mln::DType::UInt8)}};
        }
};

class Marker final : public Mln::Sdk::Lowering::Compatibility {
    public:
        std::string_view source() const noexcept override { return Mln::Ir::Graph::Format; }
        std::string_view target() const noexcept override { return "install-check"; }
        bool applies(const Mln::Sdk::Backend &) const override { return false; }

        bool lower(Mln::Ir::Module module, const Context &) const override {
            module->setAttr("install.checked",
                            Mln::Ir::StringAttribute::get(module.getContext(), "yes"));
            return true;
        }
};

class Unused final : public Mln::Sdk::Codec {
    public:
        Unused() { name = "unused"; }
        Mln::Sdk::Tensor decode(Mln::Sdk::Operations &, const Mln::Sdk::Weight &,
                                Mln::DType) override {
            throw std::logic_error("the installation fixture has no weights");
        }
};

PUBLISH(model, Identity{})
PUBLISH(marker, Marker{})
PUBLISH(unused, Mln::Sdk::Codec::Format(Unused{}))
