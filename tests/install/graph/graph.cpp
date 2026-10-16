#include <mln/sdk/functional/add.hpp>
#include <syn/package.hpp>

class Double final : public Mln::Sdk::Module {
    public:
        Double() { name = "model"; }
        std::vector<Mln::Sdk::Tensor> graph(std::span<const Mln::Sdk::Tensor> inputs) override {
            return {operations.insert<Mln::Sdk::Functional::Add>(inputs[0], inputs[0])};
        }
        std::vector<std::vector<Mln::Sdk::Tensor>> specializations() const override {
            return {{Mln::Sdk::Tensor({4}, Mln::DType::UInt8)}};
        }
};

PUBLISH(model, Double{})
