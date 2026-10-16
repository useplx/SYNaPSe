#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace Syn::Testing {
    struct Test {
            std::string name;
            std::function<void()> run;
    };

    inline std::vector<Test> &tests() {
        static std::vector<Test> values;
        return values;
    }

    struct Registration {
            Registration(std::string name, std::function<void()> run) {
                tests().push_back({std::move(name), std::move(run)});
            }
    };

    inline void require(bool condition, std::string_view message) {
        if (!condition)
            throw std::runtime_error(std::string(message));
    }
}

#define SYNPS_TEST(name)                                                                           \
    static void name();                                                                            \
    static ::Syn::Testing::Registration name##_registration(#name, name);                          \
    static void name()
