#include <syn/package.hpp>

#include <stdexcept>

class Unused final : public Mln::Sdk::Codec {
    public:
        Unused() { name = "unused"; }
        Mln::Sdk::Tensor decode(Mln::Sdk::Operations &, const Mln::Sdk::Weight &,
                                Mln::DType) override {
            throw std::logic_error("the installation graph has no weights");
        }
};

PUBLISH(unused, Mln::Sdk::Codec::Format(Unused{}))
