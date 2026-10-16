#include "json.hpp"
#include "mln/error.hpp"

#include <stdexcept>
#include <string>

namespace {
    using namespace Syn::Compiler;

    void rejects(const std::string &text, const char *what) {
        try {
            Json::parse(text);
        } catch (const Mln::Diagnostic &) {
            return;
        }
        throw std::runtime_error(std::string("parser accepted ") + what);
    }
}

int main() {
    const Json::Value value =
        Json::parse(R"({"name":"q6k","rank":3,"shape":[1,2,3],"quantized":true,"scale":null})");
    if (value.at("name").string() != "q6k")
        throw std::runtime_error("incorrect string");
    if (value.at("rank").integer() != 3)
        throw std::runtime_error("incorrect integer");
    if (value.at("shape").array().size() != 3)
        throw std::runtime_error("incorrect array");
    if (!value.at("quantized").boolean())
        throw std::runtime_error("incorrect boolean");
    if (!value.at("scale").null())
        throw std::runtime_error("incorrect null");

    rejects(std::string(4096, '[') + std::string(4096, ']'), "aninhamento fundo");
    rejects(std::string(4096, '{'), "aninhamento fundo truncado");

    rejects("{\"a\":1} lixo", "dados no fim");
    rejects("{\"a\":1,\"a\":2}", "chave duplicada");
    rejects("{\"a\":}", "missing value");
    rejects("[1,2", "array truncado");

    Json::parse(std::string(60, '[') + std::string(60, ']'));
    return 0;
}
