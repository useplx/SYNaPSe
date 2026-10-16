#include "syn/config/value.hpp"
#include "syn/error.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace Syn::Config {
    const Value::Table &Value::table() const {
        if (const auto *value = std::get_if<Table>(&data))
            return *value;
        Errors::raise(Error::Value::Table::Expected, type_name(*this));
    }

    const Value::Array &Value::array() const {
        if (const auto *value = std::get_if<Array>(&data))
            return *value;
        Errors::raise(Error::Value::Array::Expected, type_name(*this));
    }

    std::string_view Value::string() const {
        if (const auto *value = std::get_if<std::string>(&data))
            return *value;
        Errors::raise(Error::Value::String::Expected, type_name(*this));
    }

    int64_t Value::integer() const {
        if (const auto *value = std::get_if<int64_t>(&data))
            return *value;
        Errors::raise(Error::Value::Integer::Expected, type_name(*this));
    }

    double Value::real() const {
        if (const auto *value = std::get_if<double>(&data))
            return *value;
        if (const auto *value = std::get_if<int64_t>(&data))
            return static_cast<double>(*value);
        Errors::raise(Error::Value::Number::Expected, type_name(*this));
    }

    bool Value::boolean() const {
        if (const auto *value = std::get_if<bool>(&data))
            return *value;
        Errors::raise(Error::Value::Boolean::Expected, type_name(*this));
    }

    const Value *Value::find(std::string_view key) const {
        const auto *values = std::get_if<Table>(&data);
        if (!values)
            return nullptr;
        const auto found = values->find(key);
        return found == values->end() ? nullptr : &found->second;
    }

    Value *Value::find(std::string_view key) {
        auto *values = std::get_if<Table>(&data);
        if (!values)
            return nullptr;
        const auto found = values->find(key);
        return found == values->end() ? nullptr : &found->second;
    }

    const Value &Value::require(std::string_view key, std::string_view location) const {
        if (const Value *value = find(key))
            return *value;
        Errors::at(Error::Required::Missing,
                   location.empty() ? std::string(key)
                                    : std::string(location) + "." + std::string(key),
                   "field");
    }

    std::string type_name(const Value &value) {
        switch (value.data.index()) {
        case 0:
            return "null";
        case 1:
            return "boolean";
        case 2:
            return "integer";
        case 3:
            return "float";
        case 4:
            return "string";
        case 5:
            return "array";
        case 6:
            return "table";
        default:
            return "unknown";
        }
    }

    namespace {
        void append_string(std::string &output, std::string_view value) {
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
                        output.push_back(hex[character & 15]);
                    } else
                        output.push_back(static_cast<char>(character));
                }
            }
            output.push_back('"');
        }

        void append_json(std::string &output, const Value &value) {
            if (std::holds_alternative<std::monostate>(value.data)) {
                output += "null";
            } else if (const bool *flag = std::get_if<bool>(&value.data)) {
                output += *flag ? "true" : "false";
            } else if (const int64_t *number = std::get_if<int64_t>(&value.data)) {
                output += std::to_string(*number);
            } else if (const double *real = std::get_if<double>(&value.data)) {
                std::ostringstream stream;
                stream << std::setprecision(17) << *real;
                output += stream.str();
            } else if (const auto *text = std::get_if<std::string>(&value.data)) {
                append_string(output, *text);
            } else if (const auto *items = std::get_if<Value::Array>(&value.data)) {
                output.push_back('[');
                for (std::size_t index = 0; index < items->size(); ++index) {
                    if (index)
                        output.push_back(',');
                    append_json(output, (*items)[index]);
                }
                output.push_back(']');
            } else {
                output.push_back('{');
                std::size_t index = 0;
                for (const auto &[key, entry] : std::get<Value::Table>(value.data)) {
                    if (index++)
                        output.push_back(',');
                    append_string(output, key);
                    output.push_back(':');
                    append_json(output, entry);
                }
                output.push_back('}');
            }
        }
    }

    std::string json(const Value &value) {
        std::string result;
        append_json(result, value);
        return result;
    }
}
