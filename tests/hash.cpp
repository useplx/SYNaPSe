#include "test.hpp"

#include "syn/system/hash.hpp"

SYNPS_TEST(computes_sha256) {
    Syn::Testing::require(Syn::System::sha256("abc") ==
                              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                          "SHA-256 implementation returned an invalid digest");
}
