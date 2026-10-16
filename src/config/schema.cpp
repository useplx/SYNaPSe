#include "syn/config/schema.hpp"

#include "syn/error.hpp"

#include <limits>

namespace Syn::Config {
    namespace {
        bool matches(std::string_view type, const Value &value) {
            if (type == "string")
                return std::holds_alternative<std::string>(value.data);
            if (type == "boolean" || type == "bool")
                return std::holds_alternative<bool>(value.data);
            if (type == "integer" || type == "int64" || type == "uint32")
                return std::holds_alternative<int64_t>(value.data);
            if (type == "float" || type == "number")
                return std::holds_alternative<double>(value.data) ||
                       std::holds_alternative<int64_t>(value.data);
            if (type == "array")
                return value.is_array();
            if (type == "table")
                return value.is_table();
            Errors::raise(Error::Schema::Type::Unknown, type);
        }

        void validate_range(const Value &value, std::string_view type, const Value::Table &rule,
                            const std::string &location) {
            if (type == "uint32") {
                const int64_t number = value.integer();
                if (number < 0 || number > std::numeric_limits<uint32_t>::max())
                    Errors::at(Error::Variable::Out::Of::Range, location);
            }
            if (type != "integer" && type != "int64" && type != "uint32" && type != "float" &&
                type != "number")
                return;
            const double number = value.real();
            if (const auto minimum = rule.find("min");
                minimum != rule.end() && number < minimum->second.real())
                Errors::at(Error::Variable::Out::Of::Bounds, location, "below", "minimum");
            if (const auto maximum = rule.find("max");
                maximum != rule.end() && number > maximum->second.real())
                Errors::at(Error::Variable::Out::Of::Bounds, location, "above", "maximum");
        }
    }

    Value Schema::resolve(const Project &package, const Value *overrides,
                          std::string_view reference) {
        const Value *declaration = package.document().find("package");
        const Value *schema = declaration ? declaration->find("variables") : nullptr;
        if (!schema) {
            if (overrides && !overrides->table().empty())
                Errors::at(Error::Variables::Undeclared, "packages." + std::string(reference));
            return Value::Table{};
        }
        const Value::Table &rules = schema->table();
        const Value::Table empty;
        const Value::Table &provided = overrides ? overrides->table() : empty;
        for (const auto &[name, value] : provided)
            if (!rules.contains(name))
                Errors::at(Error::Schema::Variable::Unknown, std::string(reference) + "." + name,
                           name);

        Value::Table resolved;
        for (const auto &[name, rule_value] : rules) {
            const Value::Table &rule = rule_value.table();
            const auto type_item = rule.find("type");
            if (type_item == rule.end())
                Errors::at(Error::Schema::Type::Missing, "package.variables." + name);
            const std::string_view type = type_item->second.string();
            const auto supplied = provided.find(name);
            const Value *selected = supplied == provided.end() ? nullptr : &supplied->second;
            if (!selected) {
                if (const auto default_value = rule.find("default"); default_value != rule.end())
                    selected = &default_value->second;
            }
            const bool required = rule.contains("required") && rule.at("required").boolean();
            if (!selected) {
                if (required)
                    Errors::at(Error::Required::Missing, std::string(reference) + "." + name,
                               "package configuration");
                continue;
            }
            if (!matches(type, *selected))
                Errors::at(Error::Schema::Value::Type::Mismatch,
                           std::string(reference) + "." + name, type, type_name(*selected));
            validate_range(*selected, type, rule, std::string(reference) + "." + name);
            resolved.emplace(name, *selected);
        }
        return resolved;
    }
}
