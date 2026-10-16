#include "test.hpp"

#include <cstdio>
#include <exception>
#include <print>

int main() {
    std::size_t failures = 0;
    for (const auto &test : Syn::Testing::tests()) {
        try {
            test.run();
            std::println("PASS {}", test.name);
        } catch (const std::exception &error) {
            ++failures;
            std::println(stderr, "FAIL {}: {}", test.name, error.what());
        }
    }
    std::println("{}/{} tests passed", Syn::Testing::tests().size() - failures,
                 Syn::Testing::tests().size());
    return failures == 0 ? 0 : 1;
}
