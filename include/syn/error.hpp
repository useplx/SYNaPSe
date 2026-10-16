#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Syn {
    namespace Error {
        enum class Code : std::uint16_t {

            E101 = 101,
            E102 = 102,
            E103 = 103,
            E104 = 104,
            E105 = 105,
            E106 = 106,
            E107 = 107,

            E120 = 120,
            E121 = 121,
            E122 = 122,
            E123 = 123,
            E124 = 124,
            E125 = 125,
            E126 = 126,
            E127 = 127,
            E128 = 128,
            E129 = 129,
            E130 = 130,
            E131 = 131,

            E141 = 141,
            E142 = 142,
            E143 = 143,
            E144 = 144,
            E145 = 145,
            E147 = 147,
            E149 = 149,
            E190 = 190,
            E191 = 191,
            E192 = 192,
            E193 = 193,
            E194 = 194,
            E195 = 195,
            E196 = 196,
            E197 = 197,

            E150 = 150,
            E151 = 151,
            E152 = 152,
            E153 = 153,
            E154 = 154,
            E156 = 156,
            E157 = 157,
            E158 = 158,
            E159 = 159,
            E160 = 160,
            E161 = 161,

            E162 = 162,
            E163 = 163,
            E164 = 164,
            E165 = 165,
            E167 = 167,
            E168 = 168,
            E169 = 169,
            E180 = 180,

            E170 = 170,
            E172 = 172,
            E174 = 174,
            E175 = 175,
            E177 = 177,
            E178 = 178,

            E201 = 201,
            E202 = 202,
            E203 = 203,
            E204 = 204,
            E205 = 205,
            E206 = 206,
            E207 = 207,

            E301 = 301,
            E302 = 302,
            E303 = 303,
            E304 = 304,

            E402 = 402,
            E403 = 403,
            E404 = 404,
            E405 = 405,
            E406 = 406,
            E408 = 408,
            E410 = 410,
            E412 = 412,
            E413 = 413,
            E414 = 414,
            E415 = 415,
            E416 = 416,
            E417 = 417,
            E418 = 418,
            E419 = 419,
            E420 = 420,
            E421 = 421,
            E422 = 422,
            E423 = 423,
            E424 = 424,
            E425 = 425,
            E426 = 426,
            E427 = 427,
            E428 = 428,
            E429 = 429,
            E430 = 430,
            E431 = 431,
            E432 = 432,
            E433 = 433,
            E434 = 434,
            E441 = 441,
            E442 = 442,
            E446 = 446,
            E447 = 447,
            E448 = 448,

            E501 = 501,
            E502 = 502,
            E503 = 503,
            E504 = 504,
            E505 = 505,

            E601 = 601,
            E602 = 602,
            E603 = 603,
            E604 = 604,
            E605 = 605,
            E606 = 606,
        };

        namespace Package {
            namespace Manifest {
                inline constexpr Code Missing = Code::E101;
            }
        }
        namespace Dependency {
            inline constexpr Code Cycle = Code::E102;
            inline constexpr Code Exists = Code::E163;
            namespace Source {
                inline constexpr Code Required = Code::E164;
            }
            inline constexpr Code Unknown = Code::E165;
            namespace Source {
                inline constexpr Code Invalid = Code::E415;
                inline constexpr Code Ambiguous = Code::E416;
            }
            namespace Subdir {
                inline constexpr Code Invalid = Code::E448;
            }
        }
        namespace Capability {
            inline constexpr Code Unprovided = Code::E103;
            inline constexpr Code Ambiguous = Code::E104;
            namespace Not {
                inline constexpr Code Provided = Code::E105;
            }
        }
        namespace Registry {
            inline constexpr Code Unavailable = Code::E106;
        }
        namespace Configuration {
            namespace Unknown {
                inline constexpr Code Dependency = Code::E107;
            }
        }
        namespace Myelin {
            namespace Root {
                inline constexpr Code Unresolved = Code::E120;
            }
        }
        namespace Language {
            namespace Without {
                inline constexpr Code Driver = Code::E121;
            }
        }
        namespace Toolchain {
            namespace Not {
                inline constexpr Code Declared = Code::E122;
            }
            namespace Language {
                inline constexpr Code Mismatch = Code::E123;
            }
            inline constexpr Code Unavailable = Code::E124;
            namespace Compiler {
                inline constexpr Code Missing = Code::E125;
            }
            inline constexpr Code Ambiguous = Code::E131;
        }
        namespace Compiler {
            inline constexpr Code Unresponsive = Code::E126;
            namespace Api {
                inline constexpr Code Unloadable = Code::E301;
                namespace Symbol {
                    inline constexpr Code Missing = Code::E302;
                }
                inline constexpr Code Incompatible = Code::E303;
            }
        }
        namespace Output {
            inline constexpr Code Unsupported = Code::E127;
        }
        namespace Sources {
            inline constexpr Code Missing = Code::E128;
        }
        namespace Artifact {
            namespace Not {
                inline constexpr Code Produced = Code::E129;
            }
            namespace Name {
                inline constexpr Code Conflict = Code::E152;
            }
        }
        namespace Go {
            namespace Requires {
                inline constexpr Code Tinygo = Code::E130;
            }
        }

        namespace Entrypoint {
            namespace Dependency {
                inline constexpr Code Missing = Code::E190;
                inline constexpr Code Unknown = Code::E191;
            }
            namespace Path {
                inline constexpr Code Empty = Code::E192;
                inline constexpr Code Unknown = Code::E193;
            }
            inline constexpr Code Missing = Code::E141;
            inline constexpr Code Duplicated = Code::E142;
            namespace Export {
                inline constexpr Code Conflict = Code::E143;
            }
            namespace Module {
                inline constexpr Code Unknown = Code::E145;
            }
            namespace Module {
                inline constexpr Code Missing = Code::E147;
            }
        }
        namespace Hook {
            namespace Function::Format {
                inline constexpr Code Invalid = Code::E194;
            }
            namespace Function {
                inline constexpr Code Missing = Code::E195;
            }
            namespace Module {
                inline constexpr Code Unknown = Code::E196;
            }
            namespace Package {
                inline constexpr Code Unknown = Code::E197;
            }
        }
        namespace Name {
            inline constexpr Code Duplicated = Code::E144;
        }
        namespace Compiler::Plan {
            namespace Manifest::Array {
                inline constexpr Code Expected = Code::E149;
            }
            namespace Options::Array {
                inline constexpr Code Expected = Code::E154;
            }
        }
        namespace Runtime {
            inline constexpr Code Unknown = Code::E150;
        }
        namespace Path {
            inline constexpr Code Missing = Code::E151;
        }
        namespace Build {
            namespace Package::Without {
                inline constexpr Code Sources = Code::E153;
            }
            namespace Directory {
                namespace Not {
                    inline constexpr Code Removed = Code::E167;
                }
            }
            namespace Domain {
                inline constexpr Code Unknown = Code::E417;
                inline constexpr Code Empty = Code::E418;
                inline constexpr Code Duplicated = Code::E419;
            }
        }
        namespace Extends {
            namespace Format {
                inline constexpr Code Invalid = Code::E156;
            }
            namespace Project {
                inline constexpr Code Unknown = Code::E157;
            }
            namespace Package {
                inline constexpr Code Unknown = Code::E158;
            }
            namespace Runtime {
                inline constexpr Code Mismatch = Code::E159;
            }
        }
        namespace Exports {
            inline constexpr Code Missing = Code::E160;
        }
        namespace Export {
            inline constexpr Code Unknown = Code::E161;
        }
        namespace Target {
            inline constexpr Code Exists = Code::E162;
        }
        namespace Cli {
            inline constexpr Code Usage = Code::E168;
        }
        namespace Option {
            namespace Requires {
                inline constexpr Code Value = Code::E169;
            }
            inline constexpr Code Unknown = Code::E180;
        }
        namespace Weights {
            namespace Path::Selection {
                inline constexpr Code Invalid = Code::E170;
            }
            namespace Paths {
                inline constexpr Code Empty = Code::E175;
            }
        }

        namespace Environment {
            namespace Variable {
                inline constexpr Code Unset = Code::E172;
            }
            namespace Reference {
                inline constexpr Code Unterminated = Code::E177;
            }
            namespace Reference {
                inline constexpr Code Empty = Code::E178;
            }
        }

        namespace Directory {
            namespace Not {
                inline constexpr Code Created = Code::E174;
            }
        }
        namespace Node {
            namespace Name {
                inline constexpr Code Empty = Code::E201;
            }
            inline constexpr Code Duplicated = Code::E202;
            namespace Dependency {
                inline constexpr Code Unknown = Code::E204;
            }
            namespace Input {
                inline constexpr Code Missing = Code::E205;
            }
            namespace Output {
                inline constexpr Code Missing = Code::E206;
            }
            namespace Without {
                inline constexpr Code Action = Code::E207;
            }
        }
        namespace Graph {
            inline constexpr Code Cycle = Code::E203;
        }
        namespace Compilation {
            inline constexpr Code Failed = Code::E304;
        }
        namespace Value {
            inline constexpr Code Unsupported = Code::E412;
            namespace Table {
                inline constexpr Code Expected = Code::E402;
            }
            namespace Array {
                inline constexpr Code Expected = Code::E403;
            }
            namespace String {
                inline constexpr Code Expected = Code::E404;
            }
            namespace Integer {
                inline constexpr Code Expected = Code::E405;
            }
            namespace Number {
                inline constexpr Code Expected = Code::E406;
            }
            namespace Boolean {
                inline constexpr Code Expected = Code::E408;
            }
        }

        namespace Required {
            inline constexpr Code Missing = Code::E410;
        }
        namespace Manifest {
            namespace Not {
                inline constexpr Code Found = Code::E414;
            }
        }
        namespace Toml {
            inline constexpr Code Invalid = Code::E413;
        }
        namespace Section {
            inline constexpr Code Renamed = Code::E430;
        }

        namespace Schema {
            namespace Type {
                inline constexpr Code Missing = Code::E441;
                inline constexpr Code Unknown = Code::E432;
            }
            namespace Variable {
                inline constexpr Code Unknown = Code::E433;
            }
            namespace Value::Type {
                inline constexpr Code Mismatch = Code::E434;
            }
        }
        namespace Manifest::Field {
            namespace String {
                inline constexpr Code Expected = Code::E420;
            }
            namespace Array {
                inline constexpr Code Expected = Code::E421;
                namespace Element::String {
                    inline constexpr Code Expected = Code::E422;
                }
            }
        }
        namespace Manifest::Table {
            inline constexpr Code Expected = Code::E424;
        }
        namespace Toolchain::Table {
            inline constexpr Code Expected = Code::E425;
        }
        namespace Build {
            namespace Domain::Table {
                inline constexpr Code Expected = Code::E426;
            }
            namespace Toolchain {
                namespace Name {
                    inline constexpr Code Expected = Code::E427;
                }
                inline constexpr Code Unknown = Code::E428;
            }
        }
        namespace Package::Table {
            inline constexpr Code Expected = Code::E429;
        }
        namespace Binding::Package::Name {
            inline constexpr Code Expected = Code::E431;
        }
        namespace Variables {
            inline constexpr Code Undeclared = Code::E442;
        }
        namespace Variable {
            namespace Out::Of {
                inline constexpr Code Range = Code::E446;
            }
            namespace Out::Of {
                inline constexpr Code Bounds = Code::E447;
            }
        }
        namespace Lockfile {
            namespace Package {
                namespace Array {
                    inline constexpr Code Expected = Code::E501;
                }
                namespace Table {
                    inline constexpr Code Expected = Code::E503;
                }
            }
            namespace Dependency::Table {
                inline constexpr Code Expected = Code::E504;
            }
            namespace Field::String {
                inline constexpr Code Expected = Code::E505;
            }
            namespace Version {
                inline constexpr Code Unsupported = Code::E502;
            }
        }
        namespace Process {
            namespace Pipe {
                inline constexpr Code Failed = Code::E601;
            }
            namespace Executable {
                inline constexpr Code Empty = Code::E602;
            }
            namespace Start {
                inline constexpr Code Failed = Code::E603;
            }
            namespace Wait {
                inline constexpr Code Failed = Code::E604;
            }
            inline constexpr Code Failed = Code::E605;
        }
        namespace File {
            inline constexpr Code Unreadable = Code::E606;
            namespace Write {
                inline constexpr Code Failed = Code::E423;
            }
        }
    }

    class Diagnostic final : public std::runtime_error {
        public:
            Diagnostic(Error::Code error, std::string message, std::string location);

            [[nodiscard]] Error::Code error() const noexcept { return error_; }
            [[nodiscard]] std::string code() const;
            [[nodiscard]] const std::string &message() const noexcept { return message_; }
            [[nodiscard]] const std::string &location() const noexcept { return location_; }

        private:
            Error::Code error_;
            std::string message_;
            std::string location_;
    };

    class Errors {
        public:
            template <typename... Arguments>
            [[noreturn]] static void raise(Error::Code error, const Arguments &...arguments) {
                dispatch(error, std::string{}, compose(error, arguments...));
            }

            template <typename... Arguments>
            [[noreturn]] static void at(Error::Code error, std::string location,
                                        const Arguments &...arguments) {
                dispatch(error, std::move(location), compose(error, arguments...));
            }

            [[nodiscard]] static std::string code(Error::Code error);
            [[nodiscard]] static std::string_view pattern(Error::Code error);

            static void report(const Diagnostic &diagnostic, std::FILE *stream = stderr);

        private:
            [[noreturn]] static void dispatch(Error::Code error, std::string location,
                                              std::string message);

            [[nodiscard]] static std::string vcompose(Error::Code error,
                                                      std::format_args arguments);

            template <typename... Arguments>
            [[nodiscard]] static std::string compose(Error::Code error,
                                                     const Arguments &...arguments) {
                return std::apply(
                    [error](const auto &...values) {
                        return vcompose(error, std::make_format_args(values...));
                    },
                    std::tuple{text(arguments)...});
            }

            static std::string text(std::string value) { return value; }
            static std::string text(const char *value) {
                return value ? std::string(value) : std::string{};
            }
            static std::string text(std::string_view value) { return std::string(value); }
            static std::string text(const std::filesystem::path &value) { return value.string(); }
            template <typename Number, typename = std::enable_if_t<std::is_arithmetic_v<Number>>>
            static std::string text(Number value) {
                return std::to_string(value);
            }
    };
}
