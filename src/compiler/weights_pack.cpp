#include "syn/weights_pack.hpp"
#include "mln/error.hpp"
#include "syn/errors/weight.hpp"

#include "iree/base/api.h"
#include "iree/base/internal/json.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/io/formats/parser_registry.h"
#include "iree/io/parameter_index.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

namespace Syn::Compiler {
    namespace Weight {
        namespace {
            namespace Iree::File {
                struct Deleter {
                        void operator()(iree_io_file_handle_t *value) const noexcept {
                            iree_io_file_handle_release(value);
                        }
                };
                using Handle = std::unique_ptr<iree_io_file_handle_t, Deleter>;
            }

            namespace Iree::Index {
                struct Deleter {
                        void operator()(iree_io_parameter_index_t *value) const noexcept {
                            iree_io_parameter_index_release(value);
                        }
                };
                using Handle = std::unique_ptr<iree_io_parameter_index_t, Deleter>;
            }

            void check(iree_status_t status, const Mln::Error &error) {
                if (iree_status_is_ok(status)) {
                    return;
                }

                iree_allocator_t allocator = iree_allocator_system();
                char *message = nullptr;
                iree_host_size_t length = 0;
                iree_status_to_string(status, &allocator, &message, &length);
                std::string text = message ? std::string(message, length) : "IREE error";
                iree_allocator_free(allocator, message);
                iree_status_ignore(status);

                raise(error, text);
            }

            class Mapping {
                public:
                    explicit Mapping(const std::string &path)
                        : file_(::open(path.c_str(), O_RDONLY | O_CLOEXEC)) {

                        if (!file_) {
                            raise(Syn::Compiler::Errors::Weight::File::Open::Failed, path,
                                  std::strerror(errno));
                        }

                        struct stat info{};

                        if (fstat(file_.get(), &info) != 0 || info.st_size <= 0) {
                            const std::string error = std::strerror(errno);
                            raise(Syn::Compiler::Errors::Weight::File::Size::Invalid, path, error);
                        }

                        size_ = static_cast<std::size_t>(info.st_size);
                        void *mapping =
                            mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, file_.get(), 0);

                        if (mapping == MAP_FAILED) {
                            const std::string error = std::strerror(errno);
                            raise(Syn::Compiler::Errors::Weight::File::Mapping::Failed, path,
                                  error);
                        }

                        region_ =
                            Region(static_cast<const uint8_t *>(mapping), Region::Deleter{size_});
                    }

                    Mapping(const Mapping &) = delete;
                    Mapping &operator=(const Mapping &) = delete;

                    std::span<const uint8_t> bytes() const { return {region_.get(), size_}; }

                private:
                    struct File {
                            explicit File(int value) noexcept : descriptor(value) {}
                            ~File() {
                                if (descriptor >= 0)
                                    ::close(descriptor);
                            }
                            File(const File &) = delete;
                            File &operator=(const File &) = delete;
                            [[nodiscard]] explicit operator bool() const noexcept {
                                return descriptor >= 0;
                            }
                            [[nodiscard]] int get() const noexcept { return descriptor; }

                            int descriptor = -1;
                    };

                    struct Region {
                            struct Deleter {
                                    std::size_t size = 0;

                                    void operator()(const uint8_t *data) const noexcept {
                                        if (data)
                                            munmap(const_cast<uint8_t *>(data), size);
                                    }
                            };

                            Region() noexcept : value(nullptr, Deleter{}) {}
                            Region(const uint8_t *data, Deleter deleter) noexcept
                                : value(data, deleter) {}
                            [[nodiscard]] const uint8_t *get() const noexcept {
                                return value.get();
                            }

                            std::unique_ptr<const uint8_t, Deleter> value;
                    };

                    File file_;
                    Region region_;
                    std::size_t size_ = 0;
            };

            struct State {
                    std::vector<Mln::Weight::Descriptor> *entries;
            };

            iree_status_t append_shape_dimension(void *user_data, iree_host_size_t,
                                                 iree_string_view_t value) {
                auto &shape = *static_cast<std::vector<int64_t> *>(user_data);
                int64_t dimension = 0;
                IREE_RETURN_IF_ERROR(iree_json_parse_int64(value, &dimension));

                if (dimension <= 0) {
                    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                            "shape contains a non-positive dimension");
                }

                shape.push_back(dimension);

                return iree_ok_status();
            }

            iree_status_t append_offset(void *user_data, iree_host_size_t,
                                        iree_string_view_t value) {
                auto &offsets = *static_cast<std::vector<uint64_t> *>(user_data);
                uint64_t offset = 0;
                IREE_RETURN_IF_ERROR(iree_json_parse_uint64(value, &offset));
                offsets.push_back(offset);

                return iree_ok_status();
            }

            iree_status_t append_safetensors_entry(void *user_data, iree_string_view_t key,
                                                   iree_string_view_t value) {
                if (iree_string_view_equal(key, IREE_SV("__metadata__"))) {
                    return iree_ok_status();
                }

                iree_string_view_t dtype{};
                iree_string_view_t shape_json{};
                iree_string_view_t offsets_json{};
                IREE_RETURN_IF_ERROR(
                    iree_json_lookup_object_value(value, IREE_SV("dtype"), &dtype));

                IREE_RETURN_IF_ERROR(
                    iree_json_lookup_object_value(value, IREE_SV("shape"), &shape_json));

                IREE_RETURN_IF_ERROR(
                    iree_json_lookup_object_value(value, IREE_SV("data_offsets"), &offsets_json));

                Mln::Weight::Descriptor descriptor;
                descriptor.name.assign(key.data, key.size);
                descriptor.storage.assign(dtype.data, dtype.size);
                descriptor.quant = descriptor.storage;

                IREE_RETURN_IF_ERROR(iree_json_enumerate_array(shape_json, append_shape_dimension,
                                                               &descriptor.shape));

                std::vector<uint64_t> offsets;
                IREE_RETURN_IF_ERROR(
                    iree_json_enumerate_array(offsets_json, append_offset, &offsets));

                if (offsets.size() != 2 || offsets[1] < offsets[0]) {
                    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid data_offsets");
                }

                descriptor.byte_length = offsets[1] - offsets[0];

                static_cast<State *>(user_data)->entries->push_back(std::move(descriptor));

                return iree_ok_status();
            }

            std::vector<Mln::Weight::Descriptor>
            read_safetensors_descriptors(const std::string &path) {
                const Mapping file(path);
                const std::span<const uint8_t> bytes = file.bytes();

                if (bytes.size() < 8) {
                    raise(Syn::Compiler::Errors::Weight::Safetensors::Header::Missing);
                }

                uint64_t header_length = 0;

                for (unsigned i = 0; i < 8; ++i) {
                    header_length |= static_cast<uint64_t>(bytes[i]) << (i * 8);
                }

                if (header_length > bytes.size() - 8) {
                    raise(Syn::Compiler::Errors::Weight::Safetensors::Header::Truncated);
                }

                std::vector<Mln::Weight::Descriptor> entries;

                const iree_string_view_t header =
                    iree_make_string_view(reinterpret_cast<const char *>(bytes.data() + 8),
                                          static_cast<iree_host_size_t>(header_length));

                State state{&entries};
                check(iree_json_enumerate_object(header, append_safetensors_entry, &state),
                      Syn::Compiler::Errors::Weight::Iree::Safetensors::Header::Parse::Failed);

                return entries;
            }

            std::vector<std::string> parameter_files(const std::string &path) {
                std::error_code error;

                if (std::filesystem::is_regular_file(path, error)) {
                    return {path};
                }

                if (error) {
                    raise(Syn::Compiler::Errors::Weight::Checkpoint::Inspect::Failed, path,
                          error.message());
                }

                if (!std::filesystem::is_directory(path, error) || error) {
                    raise(Syn::Compiler::Errors::Weight::Checkpoint::Type::Invalid, path);
                }

                std::vector<std::string> files;

                for (const auto &entry : std::filesystem::directory_iterator(path)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                        files.push_back(entry.path().string());
                    }
                }

                std::sort(files.begin(), files.end());

                if (files.empty()) {
                    raise(Syn::Compiler::Errors::Weight::Checkpoint::Shards::Missing, path);
                }

                return files;
            }

            class Reader {
                public:
                    explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

                    uint8_t u8() { return integer<uint8_t>(); }

                    uint16_t u16() { return integer<uint16_t>(); }

                    uint32_t u32() { return integer<uint32_t>(); }

                    uint64_t u64() { return integer<uint64_t>(); }

                    std::string string() {
                        const uint64_t length = u64();

                        if (length > bytes_.size() - offset_) {
                            raise(Syn::Compiler::Errors::Weight::Gguf::String::Truncated);
                        }

                        std::string value(reinterpret_cast<const char *>(bytes_.data() + offset_),
                                          static_cast<std::size_t>(length));

                        offset_ += static_cast<std::size_t>(length);

                        return value;
                    }

                    void skip(std::size_t length) {
                        require(length);
                        offset_ += length;
                    }

                private:
                    template <typename T> T integer() {
                        require(sizeof(T));
                        T value = 0;

                        for (unsigned i = 0; i < sizeof(T); ++i) {
                            value |= static_cast<T>(bytes_[offset_++]) << (i * 8);
                        }

                        return value;
                    }

                    void require(std::size_t length) const {
                        if (length > bytes_.size() - offset_) {
                            raise(Syn::Compiler::Errors::Weight::Gguf::File::Truncated);
                        }
                    }

                    std::span<const uint8_t> bytes_;
                    std::size_t offset_ = 0;
            };

            void skip_gguf_value(Reader &reader, uint32_t type) {
                switch (type) {
                case 0:
                case 1:
                case 7:
                    reader.skip(1);
                    return;

                case 2:
                case 3:
                    reader.skip(2);
                    return;

                case 4:
                case 5:
                case 6:
                    reader.skip(4);
                    return;
                case 8:
                    (void)reader.string();
                    return;

                case 9: {
                    const uint32_t element_type = reader.u32();
                    const uint64_t count = reader.u64();

                    if (count > 100'000'000) {
                        raise(Syn::Compiler::Errors::Weight::Gguf::Metadata::Array::Too::Large);
                    }

                    for (uint64_t i = 0; i < count; ++i) {
                        skip_gguf_value(reader, element_type);
                    }

                    return;
                }

                case 10:
                case 11:
                case 12:
                    reader.skip(8);
                    return;

                default:
                    raise(Syn::Compiler::Errors::Weight::Gguf::Metadata::Type::Unknown, type);
                }
            }

            std::string gguf_quant(uint32_t file_type) {
                static constexpr const char *names[] = {
                    "F32",    "F16",    "Q4_0",   "Q4_1",   nullptr,  nullptr,  nullptr,
                    "Q8_0",   "Q5_0",   "Q5_1",   "Q2_K",   "Q3_K_S", "Q3_K_M", "Q3_K_L",
                    "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K"};

                if (file_type < std::size(names) && names[file_type]) {
                    return names[file_type];
                }

                return std::format("GGUF_FILE_TYPE_{}", file_type);
            }

            std::string_view required_gguf_storage(std::string_view quant) {
                if (quant == "F32")
                    return "F32";
                if (quant == "F16")
                    return "F16";
                if (quant == "Q2_K")
                    return "Q2_K";

                if (quant.rfind("Q3_K_", 0) == 0) {
                    return "Q3_K";
                }

                if (quant.rfind("Q4_K_", 0) == 0) {
                    return "Q4_K";
                }

                if (quant.rfind("Q5_K_", 0) == 0) {
                    return "Q5_K";
                }

                if (quant == "Q6_K")
                    return "Q6_K";
                if (quant == "Q4_0")
                    return "Q4_0";
                if (quant == "Q4_1")
                    return "Q4_1";
                if (quant == "Q5_0")
                    return "Q5_0";
                if (quant == "Q5_1")
                    return "Q5_1";
                if (quant == "Q8_0")
                    return "Q8_0";

                return {};
            }

            struct Storage {
                    const char *name;
                    uint32_t block_elements;
                    uint32_t block_bytes;
            };

            Storage ggml_storage(uint32_t type) {
                static constexpr Storage storage[] = {
                    {"F32", 1, 4},      {"F16", 1, 2},      {"Q4_0", 32, 18},   {"Q4_1", 32, 20},
                    {nullptr, 0, 0},    {nullptr, 0, 0},    {"Q5_0", 32, 22},   {"Q5_1", 32, 24},
                    {"Q8_0", 32, 34},   {"Q8_1", 32, 40},   {"Q2_K", 256, 84},  {"Q3_K", 256, 110},
                    {"Q4_K", 256, 144}, {"Q5_K", 256, 176}, {"Q6_K", 256, 210}, {"Q8_K", 256, 292},
                    {"I8", 1, 1},       {"I16", 1, 2},      {"I32", 1, 4},
                };

                if (type >= std::size(storage) || !storage[type].name) {
                    raise(Syn::Compiler::Errors::Weight::Gguf::Storage::Unsupported, type);
                }

                return storage[type];
            }

            std::vector<Mln::Weight::Descriptor> read_gguf_descriptors(const std::string &path) {
                const Mapping file(path);
                Reader reader(file.bytes());

                if (reader.u32() != 0x46554747) {
                    raise(Syn::Compiler::Errors::Weight::Gguf::Magic::Invalid);
                }

                const uint32_t version = reader.u32();

                if (version < 2 || version > 3) {
                    raise(Syn::Compiler::Errors::Weight::Gguf::Version::Unsupported, version);
                }

                const uint64_t tensor_count = reader.u64();
                const uint64_t metadata_count = reader.u64();

                if (tensor_count > 10'000'000 || metadata_count > 10'000'000) {
                    raise(Syn::Compiler::Errors::Weight::Gguf::Entry::Count::Invalid);
                }

                uint32_t file_type = std::numeric_limits<uint32_t>::max();

                for (uint64_t i = 0; i < metadata_count; ++i) {
                    const std::string key = reader.string();
                    const uint32_t type = reader.u32();

                    if (key == "general.file_type" && type == 4) {
                        file_type = reader.u32();
                    } else {
                        skip_gguf_value(reader, type);
                    }
                }

                std::vector<Mln::Weight::Descriptor> entries;
                entries.reserve(static_cast<std::size_t>(tensor_count));

                for (uint64_t i = 0; i < tensor_count; ++i) {
                    Mln::Weight::Descriptor descriptor;
                    descriptor.name = reader.string();
                    const uint32_t rank = reader.u32();

                    if (rank > 64) {
                        raise(Syn::Compiler::Errors::Weight::Gguf::Rank::Invalid);
                    }

                    descriptor.shape.resize(rank);
                    uint64_t element_count = 1;

                    for (uint32_t j = 0; j < rank; ++j) {
                        const uint64_t dimension = reader.u64();

                        if (dimension == 0 ||
                            dimension >
                                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
                            element_count > std::numeric_limits<uint64_t>::max() / dimension) {
                            raise(Syn::Compiler::Errors::Weight::Gguf::Shape::Invalid,
                                  descriptor.name);
                        }

                        descriptor.shape[rank - j - 1] = static_cast<int64_t>(dimension);
                        element_count *= dimension;
                    }

                    const Storage storage = ggml_storage(reader.u32());
                    (void)reader.u64();

                    if (element_count % storage.block_elements != 0) {
                        raise(Syn::Compiler::Errors::Weight::Gguf::Block::Alignment::Invalid,
                              descriptor.name);
                    }

                    const uint64_t blocks = element_count / storage.block_elements;

                    if (blocks > std::numeric_limits<uint64_t>::max() / storage.block_bytes) {
                        raise(Syn::Compiler::Errors::Weight::Gguf::Tensor::Too::Large);
                    }

                    descriptor.byte_length = blocks * storage.block_bytes;
                    descriptor.storage = storage.name;
                    descriptor.quant = descriptor.storage;
                    entries.push_back(std::move(descriptor));
                }

                if (file_type != std::numeric_limits<uint32_t>::max()) {
                    const std::string quant = gguf_quant(file_type);
                    const std::string_view required = required_gguf_storage(quant);

                    if (!required.empty() &&
                        std::none_of(entries.begin(), entries.end(),
                                     [&](const Mln::Weight::Descriptor &entry) {
                                         return entry.storage == required;
                                     })) {
                        raise(Syn::Compiler::Errors::Weight::Gguf::Declared::Storage::Missing,
                              quant, required);
                    }
                }

                return entries;
            }

            std::vector<Mln::Weight::Descriptor> read_irpa_descriptors(const std::string &path) {
                const iree_allocator_t allocator = iree_allocator_system();
                iree_io_file_handle_t *file_value = nullptr;

                check(iree_io_file_handle_open(
                          IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS,
                          iree_make_cstring_view(path.c_str()), allocator, &file_value),
                      Syn::Compiler::Errors::Weight::Iree::Archive::File::Open::Failed);
                Iree::File::Handle file(file_value);

                iree_io_parameter_index_t *index_value = nullptr;
                check(iree_io_parameter_index_create(allocator, &index_value),
                      Syn::Compiler::Errors::Weight::Iree::Source::Index::Create::Failed);
                Iree::Index::Handle index(index_value);
                check(iree_io_parse_file_index(iree_make_cstring_view(path.c_str()), file.get(),
                                               index.get(), allocator),
                      Syn::Compiler::Errors::Weight::Iree::Archive::Index::Parse::Failed);
                std::vector<Mln::Weight::Descriptor> entries;
                entries.reserve(iree_io_parameter_index_count(index.get()));

                for (iree_host_size_t i = 0; i < iree_io_parameter_index_count(index.get()); ++i) {
                    const iree_io_parameter_index_entry_t *entry = nullptr;
                    check(iree_io_parameter_index_get(index.get(), i, &entry),
                          Syn::Compiler::Errors::Weight::Iree::Index::Entry::Read::Failed);

                    if (entry->metadata.data_length == 0) {
                        raise(Syn::Compiler::Errors::Weight::Irpa::Metadata::Missing,
                              std::string_view(entry->key.data, entry->key.size));
                    }

                    auto descriptor = Mln::Weight::decode_weight_metadata(std::span<const uint8_t>(
                        entry->metadata.data, entry->metadata.data_length));

                    const std::string key(entry->key.data, entry->key.size);

                    if (descriptor.name != key || descriptor.byte_length != entry->length) {
                        raise(Syn::Compiler::Errors::Weight::Irpa::Metadata::Mismatch, key);
                    }

                    entries.push_back(std::move(descriptor));
                }

                return entries;
            }

            iree_status_t open_archive_memory(void *user_data,
                                              iree_io_physical_offset_t archive_offset,
                                              iree_io_physical_size_t archive_length,
                                              iree_io_file_handle_t **out_file_handle) {
                auto &bytes = *static_cast<std::vector<uint8_t> *>(user_data);

                if (archive_offset > std::numeric_limits<std::size_t>::max() ||
                    archive_length > std::numeric_limits<std::size_t>::max() -
                                         static_cast<std::size_t>(archive_offset)) {
                    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "parameter archive "
                                                                      "grande demais");
                }

                bytes.resize(static_cast<std::size_t>(archive_offset + archive_length));

                return iree_io_file_handle_wrap_host_allocation(
                    IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
                    iree_make_byte_span(bytes.data(), bytes.size()),
                    iree_io_file_handle_release_callback_null(), iree_allocator_system(),
                    out_file_handle);
            }

            uint64_t descriptor_element_count(const Mln::Weight::Descriptor &descriptor) {
                uint64_t count = 1;

                for (int64_t dimension : descriptor.shape) {
                    if (dimension <= 0 || count > std::numeric_limits<uint64_t>::max() /
                                                      static_cast<uint64_t>(dimension)) {
                        raise(Syn::Compiler::Errors::Weight::Descriptor::Shape::Invalid,
                              descriptor.name);
                    }

                    count *= static_cast<uint64_t>(dimension);
                }

                return count;
            }
        }

        std::vector<Mln::Weight::Descriptor> read(std::string_view path) {
            const std::string path_string(path);
            if (std::filesystem::is_directory(path_string)) {
                std::vector<Mln::Weight::Descriptor> descriptors;
                std::unordered_set<std::string> names;

                for (const std::string &file : parameter_files(path_string)) {
                    auto shard = read_safetensors_descriptors(file);

                    for (auto &descriptor : shard) {
                        if (!names.insert(descriptor.name).second) {
                            raise(Syn::Compiler::Errors::Weight::Duplicate::Shards,
                                  descriptor.name);
                        }

                        descriptors.push_back(std::move(descriptor));
                    }
                }

                return descriptors;
            }

            const std::string extension = std::filesystem::path(path_string).extension().string();

            if (extension == ".safetensors") {
                return read_safetensors_descriptors(path_string);
            }

            if (extension == ".gguf") {
                return read_gguf_descriptors(path_string);
            }

            if (extension == ".irpa") {
                return read_irpa_descriptors(path_string);
            }

            raise(Syn::Compiler::Errors::Weight::Metadata::Typed::Unsupported, extension);
        }

        Conversion::Result Conversion::empty() {
            const iree_allocator_t allocator = iree_allocator_system();
            iree_io_parameter_index_t *source_raw = nullptr;
            check(iree_io_parameter_index_create(allocator, &source_raw),
                  Syn::Compiler::Errors::Weight::Iree::Source::Index::Create::Failed);
            Iree::Index::Handle source(source_raw);

            iree_io_parameter_index_t *target_raw = nullptr;
            check(iree_io_parameter_index_create(allocator, &target_raw),
                  Syn::Compiler::Errors::Weight::Iree::Target::Index::Create::Failed);
            Iree::Index::Handle target(target_raw);

            Conversion::Result converted;
            iree_io_parameter_archive_file_open_callback_t callback{open_archive_memory,
                                                                    &converted.data};
            check(
                iree_io_build_parameter_archive(source.get(), target.get(), callback, 0, allocator),
                Syn::Compiler::Errors::Weight::Iree::Empty::Archive::Build::Failed);
            converted.statistics.output_bytes = converted.data.size();
            return converted;
        }

        Conversion::Result Conversion::inspect(std::span<const uint8_t> archive) {
            const iree_allocator_t allocator = iree_allocator_system();
            iree_io_file_handle_t *file_raw = nullptr;
            check(iree_io_file_handle_wrap_host_allocation(
                      IREE_IO_FILE_ACCESS_READ,
                      iree_make_byte_span(const_cast<uint8_t *>(archive.data()), archive.size()),
                      iree_io_file_handle_release_callback_null(), allocator, &file_raw),
                  Syn::Compiler::Errors::Weight::Iree::Archive::File::Open::Failed);
            Iree::File::Handle file(file_raw);

            iree_io_parameter_index_t *index_raw = nullptr;
            check(iree_io_parameter_index_create(allocator, &index_raw),
                  Syn::Compiler::Errors::Weight::Iree::Source::Index::Create::Failed);
            Iree::Index::Handle index(index_raw);
            check(iree_io_parse_file_index(iree_make_cstring_view("parameters.irpa"), file.get(),
                                           index.get(), allocator),
                  Syn::Compiler::Errors::Weight::Iree::Archive::Index::Parse::Failed);

            Conversion::Result result;
            result.resources.reserve(iree_io_parameter_index_count(index.get()));
            result.descriptors.reserve(iree_io_parameter_index_count(index.get()));
            for (iree_host_size_t i = 0; i < iree_io_parameter_index_count(index.get()); ++i) {
                const iree_io_parameter_index_entry_t *entry = nullptr;
                check(iree_io_parameter_index_get(index.get(), i, &entry),
                      Syn::Compiler::Errors::Weight::Iree::Index::Entry::Read::Failed);
                if (entry->type != IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE)
                    continue;
                const std::string name(entry->key.data, entry->key.size);
                if (entry->metadata.data_length == 0)
                    raise(Syn::Compiler::Errors::Weight::Irpa::Metadata::Missing, name);
                auto descriptor = Mln::Weight::decode_weight_metadata(
                    std::span<const uint8_t>(entry->metadata.data, entry->metadata.data_length));
                if (descriptor.name != name || descriptor.byte_length != entry->length)
                    raise(Syn::Compiler::Errors::Weight::Irpa::Metadata::Mismatch, name);
                result.descriptors.push_back(std::move(descriptor));
                result.resources.push_back({name, entry->storage.file.offset, entry->length});
            }
            return result;
        }

        Conversion::Result Conversion::convert(
            std::string_view path, const std::vector<Mln::Weight::Descriptor> &descriptors,
            const std::vector<Mln::Weight::Descriptor> &requirements, const Quant::Lookup &quants,
            const Scheme::Lookup &schemes, const Residency::Lookup &residency) {
            const std::string path_string(path);
            const iree_allocator_t allocator = iree_allocator_system();
            std::unordered_map<std::string, const Mln::Weight::Descriptor *> by_name;

            for (const auto &descriptor : descriptors) {
                if (!by_name.emplace(descriptor.name, &descriptor).second) {
                    raise(Syn::Compiler::Errors::Weight::Duplicate::Catalog, descriptor.name);
                }
            }

            std::unordered_map<std::string, std::vector<const Mln::Weight::Descriptor *>>
                targets_by_source;

            for (const auto &requirement : requirements) {
                const std::string &source_name =
                    requirement.storage.empty() ? requirement.name : requirement.storage;

                targets_by_source[source_name].push_back(&requirement);
            }

            struct Record {
                    std::string name;
                    Mln::Weight::Descriptor descriptor;
                    uint64_t offset;
            };
            std::vector<Record> records;
            std::vector<uint8_t> transformed_bytes;
            std::unordered_set<std::string> seen;
            Conversion::Statistics statistics;
            records.reserve(requirements.size());

            for (const std::string &source_path : parameter_files(path_string)) {
                const Mapping source_mapping(source_path);
                const std::span<const uint8_t> source_bytes = source_mapping.bytes();
                iree_io_file_handle_t *source_file_raw = nullptr;

                check(iree_io_file_handle_open(
                          IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_RANDOM_ACCESS,
                          iree_make_cstring_view(source_path.c_str()), allocator, &source_file_raw),
                      Syn::Compiler::Errors::Weight::Iree::Archive::File::Open::Failed);
                Iree::File::Handle source_file(source_file_raw);

                iree_io_parameter_index_t *source_index_raw = nullptr;
                check(iree_io_parameter_index_create(allocator, &source_index_raw),
                      Syn::Compiler::Errors::Weight::Iree::Source::Index::Create::Failed);
                Iree::Index::Handle source_index(source_index_raw);
                check(iree_io_parse_file_index(iree_make_cstring_view(source_path.c_str()),
                                               source_file.get(), source_index.get(), allocator),
                      Syn::Compiler::Errors::Weight::Iree::Archive::Index::Parse::Failed);

                for (iree_host_size_t i = 0;

                     i < iree_io_parameter_index_count(source_index.get());

                     ++i) {
                    const iree_io_parameter_index_entry_t *entry = nullptr;
                    check(iree_io_parameter_index_get(source_index.get(), i, &entry),
                          Syn::Compiler::Errors::Weight::Iree::Index::Entry::Read::Failed);
                    const std::string name(entry->key.data, entry->key.size);
                    const auto target_found = targets_by_source.find(name);

                    if (target_found == targets_by_source.end()) {
                        continue;
                    }

                    if (!seen.insert(name).second) {
                        raise(Syn::Compiler::Errors::Weight::Duplicate::Shards, name);
                    }

                    const auto source_found = by_name.find(name);

                    if (source_found == by_name.end()) {
                        raise(Syn::Compiler::Errors::Weight::Irpa::Descriptor::Missing, name);
                    }

                    const auto &source_descriptor = *source_found->second;

                    if (entry->length != source_descriptor.byte_length) {
                        raise(Syn::Compiler::Errors::Weight::Irpa::Descriptor::Size::Mismatch,
                              name);
                    }

                    statistics.source_bytes += entry->length;
                    std::vector<uint8_t> owned_raw;
                    std::span<const uint8_t> raw;

                    if (entry->type == IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE) {
                        const uint64_t offset = entry->storage.file.offset;

                        if (offset > source_bytes.size() ||
                            entry->length >
                                source_bytes.size() - static_cast<std::size_t>(offset)) {
                            raise(Syn::Compiler::Errors::Weight::Storage::Outside::File, name);
                        }

                        raw = source_bytes.subspan(static_cast<std::size_t>(offset),
                                                   static_cast<std::size_t>(entry->length));
                    } else if (entry->type == IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT) {
                        const uint8_t length = entry->storage.splat.pattern_length;

                        if (length == 0) {
                            raise(Syn::Compiler::Errors::Weight::Storage::Splat::Empty, name);
                        }

                        owned_raw.resize(entry->length);

                        for (std::size_t j = 0; j < owned_raw.size(); ++j) {
                            owned_raw[j] = entry->storage.splat.pattern[j % length];
                        }

                        raw = owned_raw;
                    } else {
                        raise(Syn::Compiler::Errors::Weight::Storage::Kind::Unknown, name);
                    }

                    const auto &targets = target_found->second;

                    const std::string &source_storage = source_descriptor.storage.empty()
                                                            ? source_descriptor.quant
                                                            : source_descriptor.storage;

                    const bool composite = targets.size() != 1 || targets.front()->name != name;

                    if (!composite) {
                        if (targets.front()->shape != source_descriptor.shape) {
                            raise(Syn::Compiler::Errors::Weight::Plan::Shape::Mismatch, name);
                        }
                    }

                    if (composite) {
                        const std::string &target_quant = targets.front()->quant;

                        if (std::any_of(targets.begin(), targets.end(), [&](const auto *target) {
                                return target->quant != target_quant;
                            })) {
                            raise(Syn::Compiler::Errors::Weight::Plan::Component::Codec::Mismatch,
                                  name);
                        }

                        const auto &target_scheme = schemes(name);

                        auto *target_codec =
                            dynamic_cast<Mln::Sdk::Codec *>(&target_scheme.format.get());

                        if (!target_codec || target_codec->storage_name() != target_quant) {
                            raise(Syn::Compiler::Errors::Weight::Plan::Scheme::Mismatch, name);
                        }

                        const Mln::Sdk::Weight logical_weight{
                            .name = name,
                            .shape = source_descriptor.shape,
                            .value = std::nullopt,
                        };

                        std::vector<float> values = quants(source_storage).decode(raw);

                        if (values.size() != descriptor_element_count(source_descriptor)) {
                            raise(Syn::Compiler::Errors::Weight::Codec::Element::Count::Mismatch,
                                  name);
                        }

                        const auto components =
                            target_codec->encode(values, logical_weight, target_scheme);

                        if (components.size() != targets.size()) {
                            raise(Syn::Compiler::Errors::Weight::Codec::Component::Count::Mismatch,
                                  name);
                        }

                        for (const auto *target : targets) {
                            const auto component = std::find_if(
                                components.begin(), components.end(), [&](const auto &candidate) {
                                    return name + candidate.suffix == target->name;
                                });

                            if (component == components.end()) {
                                raise(Syn::Compiler::Errors::Weight::Codec::Component::Missing,
                                      target->name);
                            }

                            Mln::Weight::Descriptor output = source_descriptor;
                            output.name = target->name;
                            output.quant = target_quant;
                            output.storage = target_quant;
                            output.shape = target->shape;
                            output.byte_length = component->bytes.size();
                            output.residency =
                                residency ? residency(target->name) : target_scheme.residency;
                            const uint64_t offset = transformed_bytes.size();

                            transformed_bytes.insert(transformed_bytes.end(),
                                                     component->bytes.begin(),
                                                     component->bytes.end());

                            records.push_back(Record{target->name, std::move(output), offset});
                            seen.insert(target->name);
                        }

                        ++statistics.converted_parameters;

                        continue;
                    }

                    const auto &target = *targets.front();
                    Mln::Weight::Descriptor output = source_descriptor;
                    const auto &target_scheme = schemes(target.name);
                    output.residency = residency ? residency(target.name) : target_scheme.residency;

                    if (source_storage != target.quant) {
                        ++statistics.converted_parameters;
                        std::vector<float> values = quants(source_storage).decode(raw);

                        if (values.size() != descriptor_element_count(source_descriptor)) {
                            raise(Syn::Compiler::Errors::Weight::Codec::Element::Count::Mismatch,
                                  name);
                        }

                        auto *target_codec =
                            dynamic_cast<Mln::Sdk::Codec *>(&target_scheme.format.get());

                        if (!target_codec || target_codec->storage_name() != target.quant) {
                            raise(Syn::Compiler::Errors::Weight::Plan::Scheme::Mismatch, name);
                        }

                        owned_raw = target_codec->encode(values, target_scheme);
                        raw = owned_raw;
                    } else {
                        ++statistics.passthrough_parameters;
                    }

                    output.quant = target.quant;
                    output.storage = target.quant;
                    output.byte_length = raw.size();
                    const uint64_t offset = transformed_bytes.size();
                    transformed_bytes.insert(transformed_bytes.end(), raw.begin(), raw.end());
                    records.push_back(Record{name, std::move(output), offset});
                    seen.insert(target.name);
                }
            }

            for (const auto &requirement : requirements) {
                if (!seen.contains(requirement.name)) {
                    raise(Syn::Compiler::Errors::Weight::Irpa::Required::Weight::Missing,
                          requirement.name);
                }
            }

            iree_io_parameter_index_t *annotated_index_raw = nullptr;
            check(iree_io_parameter_index_create(allocator, &annotated_index_raw),
                  Syn::Compiler::Errors::Weight::Iree::Annotated::Index::Create::Failed);
            Iree::Index::Handle annotated_index(annotated_index_raw);
            check(iree_io_parameter_index_reserve(annotated_index.get(), records.size()),
                  Syn::Compiler::Errors::Weight::Iree::Annotated::Index::Reserve::Failed);

            iree_io_file_handle_t *transformed_file_raw = nullptr;
            check(iree_io_file_handle_wrap_host_allocation(
                      IREE_IO_FILE_ACCESS_READ,
                      iree_make_byte_span(transformed_bytes.data(), transformed_bytes.size()),
                      iree_io_file_handle_release_callback_null(), allocator,
                      &transformed_file_raw),
                  Syn::Compiler::Errors::Weight::Iree::Transformed::Buffer::Wrap::Failed);
            Iree::File::Handle transformed_file(transformed_file_raw);

            for (const Record &record : records) {
                const std::vector<uint8_t> metadata =
                    Mln::Weight::encode_weight_metadata(record.descriptor);

                iree_io_parameter_index_entry_t annotated{};
                annotated.key = iree_make_string_view(record.name.data(), record.name.size());
                annotated.metadata = iree_make_const_byte_span(metadata.data(), metadata.size());
                annotated.length = record.descriptor.byte_length;
                annotated.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
                annotated.storage.file.handle = transformed_file.get();
                annotated.storage.file.offset = record.offset;
                check(iree_io_parameter_index_add(annotated_index.get(), &annotated),
                      Syn::Compiler::Errors::Weight::Iree::Annotated::Entry::Add::Failed);
            }

            iree_io_parameter_index_t *target_index_raw = nullptr;
            check(iree_io_parameter_index_create(allocator, &target_index_raw),
                  Syn::Compiler::Errors::Weight::Iree::Target::Index::Create::Failed);
            Iree::Index::Handle target_index(target_index_raw);

            Conversion::Result converted;
            converted.statistics = statistics;
            converted.descriptors.reserve(records.size());

            for (const Record &record : records) {
                converted.descriptors.push_back(record.descriptor);
            }

            iree_io_parameter_archive_file_open_callback_t callback{open_archive_memory,
                                                                    &converted.data};

            check(iree_io_build_parameter_archive(annotated_index.get(), target_index.get(),
                                                  callback, 0, allocator),
                  Syn::Compiler::Errors::Weight::Iree::Converted::Archive::Build::Failed);

            converted.resources.reserve(iree_io_parameter_index_count(target_index.get()));
            for (iree_host_size_t index = 0;
                 index < iree_io_parameter_index_count(target_index.get()); ++index) {
                const iree_io_parameter_index_entry_t *entry = nullptr;
                check(iree_io_parameter_index_get(target_index.get(), index, &entry),
                      Syn::Compiler::Errors::Weight::Iree::Index::Entry::Read::Failed);
                if (entry->type != IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE)
                    continue;
                converted.resources.push_back({
                    .name = std::string(entry->key.data, entry->key.size),
                    .offset = entry->storage.file.offset,
                    .length = entry->length,
                });
            }

            converted.statistics.source_parameters = descriptors.size();
            converted.statistics.output_parameters = converted.descriptors.size();
            converted.statistics.output_bytes = converted.data.size();

            return converted;
        }
    }
}
