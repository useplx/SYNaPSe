#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace Syn::Config {
    struct Value {
            using Array = std::vector<Value>;
            using Table = std::map<std::string, Value, std::less<>>;
            using Data =
                std::variant<std::monostate, bool, int64_t, double, std::string, Array, Table>;

            Data data;

            Value() = default;
            template <typename T> Value(T value) : data(std::move(value)) {}

            bool is_table() const noexcept { return std::holds_alternative<Table>(data); }
            bool is_array() const noexcept { return std::holds_alternative<Array>(data); }
            const Table &table() const;
            const Array &array() const;
            std::string_view string() const;
            int64_t integer() const;
            double real() const;
            bool boolean() const;

            const Value *find(std::string_view key) const;
            Value *find(std::string_view key);
            const Value &require(std::string_view key, std::string_view location = {}) const;
    };

    std::string type_name(const Value &value);
    std::string json(const Value &value);
}
