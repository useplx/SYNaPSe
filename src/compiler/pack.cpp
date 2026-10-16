#include "mln/error.hpp"
#include "mln/weight.hpp"
#include "syn/weights_pack.hpp"

#include <print>
#include <string_view>

int main(int argc, char **argv) {
    if (argc != 4 || std::string_view(argv[1]) != "--inspect") {
        std::println(stderr,
                     "usage: {} --inspect <weights.(safetensors|gguf|irpa)> <weights.index>",
                     argv[0]);
        return 2;
    }
    try {
        const auto descriptors = Syn::Compiler::Weight::read(argv[2]);
        Mln::Weight::write_binary_file(
            argv[3],
            Mln::Weight::encode_weight_index(Mln::Weight::Index::Kind::Descriptors, descriptors));
    } catch (const Mln::Diagnostic &diagnostic) {
        Mln::report(diagnostic);
        return 1;
    } catch (const std::exception &error) {
        std::println(stderr, "error: {}", error.what());
        return 1;
    }
    return 0;
}
