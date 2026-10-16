#include <mln/sdk/native/entrypoint.hpp>

class Echo final : public Mln::Sdk::Orchestration {
    public:
        Mln::Buffer forward(Mln::Runtime::Session &, Mln::Buffer input) override { return input; }
};

ENTRYPOINT(Echo)
