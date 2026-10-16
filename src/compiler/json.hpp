#include "mln/error.hpp"
#include "syn/errors/json.hpp"
#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <format>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace Syn::Compiler::Json {
    struct Value;
    using Object = std::map<std::string, Value>;
    using Array = std::vector<Value>;

    struct Value {
            using Data = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;
            Data data;

            const Object &object() const { return std::get<Object>(data); }
            const Array &array() const { return std::get<Array>(data); }
            const std::string &string() const { return std::get<std::string>(data); }
            bool boolean() const { return std::get<bool>(data); }
            double number() const { return std::get<double>(data); }
            int64_t integer() const {
                const double value = number();
                if (!std::isfinite(value) || std::trunc(value) != value ||
                    value < static_cast<double>(INT64_MIN) ||
                    value > static_cast<double>(INT64_MAX))
                    raise(Syn::Compiler::Errors::Json::Integer::Expected);
                return static_cast<int64_t>(value);
            }
            bool null() const { return std::holds_alternative<std::nullptr_t>(data); }
            bool contains(std::string_view key) const {
                return object().contains(std::string(key));
            }
            const Value &at(std::string_view key) const {
                const auto found = object().find(std::string(key));
                if (found == object().end())
                    raise(Syn::Compiler::Errors::Json::Field::Missing, key);
                return found->second;
            }
    };

    class Parser {
        public:
            explicit Parser(std::string_view text) : text_(text) {}

            Value parse() {
                Value value = parse_value();
                whitespace();
                if (!text_.empty())
                    fail("trailing data");
                return value;
            }

        private:
            static constexpr std::size_t max_depth = 64;

            std::string_view text_;
            std::size_t offset_ = 0;
            std::size_t depth_ = 0;

            [[noreturn]] void fail(std::string_view message) const {
                raise(Syn::Compiler::Errors::Json::Malformed, message, offset_);
            }

            char take() {
                if (text_.empty())
                    fail("unexpected end");
                const char value = text_.front();
                text_.remove_prefix(1);
                ++offset_;
                return value;
            }

            bool consume(char value) {
                if (!text_.empty() && text_.front() == value) {
                    take();
                    return true;
                }
                return false;
            }

            void whitespace() {
                while (!text_.empty() && (text_.front() == ' ' || text_.front() == '\t' ||
                                          text_.front() == '\r' || text_.front() == '\n'))
                    take();
            }

            Value parse_value() {
                whitespace();
                if (text_.empty())
                    fail("expected value");
                switch (text_.front()) {
                case '{':
                case '[': {
                    if (++depth_ > max_depth)
                        fail("nesting too deep");
                    Value value =
                        text_.front() == '{' ? Value{parse_object()} : Value{parse_array()};
                    --depth_;
                    return value;
                }
                case '"':
                    return Value{parse_string()};
                case 't':
                    literal("true");
                    return Value{true};
                case 'f':
                    literal("false");
                    return Value{false};
                case 'n':
                    literal("null");
                    return Value{nullptr};
                default:
                    return Value{parse_number()};
                }
            }

            Object parse_object() {
                take();
                whitespace();
                Object result;
                if (consume('}'))
                    return result;
                while (true) {
                    whitespace();
                    if (text_.empty() || text_.front() != '"')
                        fail("expected object key");
                    std::string key = parse_string();
                    whitespace();
                    if (!consume(':'))
                        fail("expected ':'");
                    if (!result.emplace(std::move(key), parse_value()).second)
                        fail("duplicate object key");
                    whitespace();
                    if (consume('}'))
                        return result;
                    if (!consume(','))
                        fail("expected ',' or '}'");
                }
            }

            Array parse_array() {
                take();
                whitespace();
                Array result;
                if (consume(']'))
                    return result;
                while (true) {
                    result.push_back(parse_value());
                    whitespace();
                    if (consume(']'))
                        return result;
                    if (!consume(','))
                        fail("expected ',' or ']'");
                }
            }

            static void append_utf8(std::string &output, uint32_t value) {
                if (value <= 0x7f)
                    output.push_back(static_cast<char>(value));
                else if (value <= 0x7ff) {
                    output.push_back(static_cast<char>(0xc0 | (value >> 6)));
                    output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
                } else if (value <= 0xffff) {
                    output.push_back(static_cast<char>(0xe0 | (value >> 12)));
                    output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
                    output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
                } else {
                    output.push_back(static_cast<char>(0xf0 | (value >> 18)));
                    output.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
                    output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
                    output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
                }
            }

            uint32_t hex4() {
                uint32_t result = 0;
                for (int i = 0; i < 4; ++i) {
                    const char value = take();
                    result <<= 4;
                    if (value >= '0' && value <= '9')
                        result |= value - '0';
                    else if (value >= 'a' && value <= 'f')
                        result |= value - 'a' + 10;
                    else if (value >= 'A' && value <= 'F')
                        result |= value - 'A' + 10;
                    else
                        fail("invalid unicode escape");
                }
                return result;
            }

            std::string parse_string() {
                if (take() != '"')
                    fail("expected string");
                std::string result;
                while (true) {
                    const char value = take();
                    if (value == '"')
                        return result;
                    if (static_cast<unsigned char>(value) < 0x20)
                        fail("control character in string");
                    if (value != '\\') {
                        result.push_back(value);
                        continue;
                    }
                    switch (take()) {
                    case '"':
                        result.push_back('"');
                        break;
                    case '\\':
                        result.push_back('\\');
                        break;
                    case '/':
                        result.push_back('/');
                        break;
                    case 'b':
                        result.push_back('\b');
                        break;
                    case 'f':
                        result.push_back('\f');
                        break;
                    case 'n':
                        result.push_back('\n');
                        break;
                    case 'r':
                        result.push_back('\r');
                        break;
                    case 't':
                        result.push_back('\t');
                        break;
                    case 'u': {
                        uint32_t codepoint = hex4();
                        if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                            if (take() != '\\' || take() != 'u')
                                fail("missing low surrogate");
                            const uint32_t low = hex4();
                            if (low < 0xdc00 || low > 0xdfff)
                                fail("invalid low surrogate");
                            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                        }
                        append_utf8(result, codepoint);
                        break;
                    }
                    default:
                        fail("invalid string escape");
                    }
                }
            }

            double parse_number() {
                const char *begin = text_.data();
                const char *end = begin + text_.size();
                double result = 0;
                const auto parsed = std::from_chars(begin, end, result);
                if (parsed.ec != std::errc{} || parsed.ptr == begin)
                    fail("invalid number");
                const std::size_t consumed = parsed.ptr - begin;
                text_.remove_prefix(consumed);
                offset_ += consumed;
                return result;
            }

            void literal(std::string_view expected) {
                if (!text_.starts_with(expected))
                    fail("invalid literal");
                text_.remove_prefix(expected.size());
                offset_ += expected.size();
            }
    };

    inline Value parse(std::string_view text) { return Parser(text).parse(); }

    inline void append_string(std::string &output, std::string_view value) {
        static constexpr char hex[] = "0123456789abcdef";
        output.push_back('"');
        for (const unsigned char character : value) {
            switch (character) {
            case '"':
                output += "\\\"";
                break;
            case '\\':
                output += "\\\\";
                break;
            case '\b':
                output += "\\b";
                break;
            case '\f':
                output += "\\f";
                break;
            case '\n':
                output += "\\n";
                break;
            case '\r':
                output += "\\r";
                break;
            case '\t':
                output += "\\t";
                break;
            default:
                if (character < 0x20) {
                    output += "\\u00";
                    output.push_back(hex[character >> 4]);
                    output.push_back(hex[character & 0xf]);
                } else
                    output.push_back(static_cast<char>(character));
            }
        }
        output.push_back('"');
    }

    inline void append(std::string &output, const Value &value) {
        if (value.null()) {
            output += "null";
            return;
        }
        if (const bool *boolean = std::get_if<bool>(&value.data)) {
            output += *boolean ? "true" : "false";
            return;
        }
        if (const double *number = std::get_if<double>(&value.data)) {
            std::ostringstream stream;
            stream.precision(17);
            stream << *number;
            output += stream.str();
            return;
        }
        if (const auto *string = std::get_if<std::string>(&value.data)) {
            append_string(output, *string);
            return;
        }
        if (const Array *array = std::get_if<Array>(&value.data)) {
            output.push_back('[');
            for (std::size_t i = 0; i < array->size(); ++i) {
                if (i)
                    output.push_back(',');
                append(output, (*array)[i]);
            }
            output.push_back(']');
            return;
        }
        output.push_back('{');
        std::size_t index = 0;
        for (const auto &[key, item] : std::get<Object>(value.data)) {
            if (index++)
                output.push_back(',');
            append_string(output, key);
            output.push_back(':');
            append(output, item);
        }
        output.push_back('}');
    }

    inline std::string stringify(const Value &value) {
        std::string result;
        append(result, value);
        return result;
    }
}
