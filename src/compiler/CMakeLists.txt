add_compile_options(-Wall -Wextra -Wshadow -Wno-missing-field-initializers)

add_library(syn_weights_pack STATIC weights_pack.cpp)
target_include_directories(syn_weights_pack PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${MLN_ROOT}/include
)
target_include_directories(syn_weights_pack SYSTEM PRIVATE
  ${IREE_ROOT}/runtime/src
  ${CMAKE_BINARY_DIR}/third_party/iree/runtime/src
)
target_link_libraries(syn_weights_pack PUBLIC
  iree_base_internal_json
  iree_io_formats_irpa_irpa
  iree_io_formats_parser_registry
  iree_io_parameter_index
)
target_compile_features(syn_weights_pack PUBLIC cxx_std_23)

add_library(syn_compiler STATIC
  compiler.cpp
  artifact.cpp
  report.cpp
  manifest.cpp
  deployment.cpp
  dispatch_affinity.cpp
  backend.cpp
  vm_compatibility.cpp
  operations.cpp
  compatibility.cpp
  lowering.cpp
  functional.cpp
  functional/const.cpp
  functional/convert.cpp
  functional/bitcastconvert.cpp
  functional/reshape.cpp
  functional/broadcast.cpp
  functional/concatenate.cpp
  functional/matmul.cpp
  functional/attention.cpp
  functional/causal_conv1d.cpp
  functional/conv2d.cpp
  functional/maxpool2d.cpp
  functional/slice.cpp
  functional/upsample.cpp
  functional/transpose.cpp
  functional/reducesum.cpp
  functional/reducemax.cpp
  functional/gather.cpp
    functional/gated_delta_rule.cpp
    functional/gated_delta_chunk.cpp
  functional/iota.cpp
  functional/dynamicslice.cpp
  functional/dynamicupdateslice.cpp
  functional/equal.cpp
  functional/select.cpp
  functional/clamp.cpp
  functional/reverse.cpp
  functional/softplus.cpp
  functional/map.cpp
  functional/loop.cpp
)
set_source_files_properties(backend.cpp dispatch_affinity.cpp vm_compatibility.cpp PROPERTIES
  COMPILE_OPTIONS "-fno-rtti"
)

add_library(syn_package_registry STATIC
  package_registry.cpp
)
target_include_directories(syn_package_registry PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${MLN_ROOT}/include
)
target_link_libraries(syn_package_registry PUBLIC dl)
target_compile_features(syn_package_registry PUBLIC cxx_std_23)

add_library(syn_compiler_api SHARED compiler_api.cpp)
target_link_libraries(syn_compiler_api PRIVATE
  syn_compiler
  syn_package_registry
)
target_compile_features(syn_compiler_api PRIVATE cxx_std_23)
target_compile_definitions(syn_compiler_api PRIVATE SYN_COMPILER_API_BUILD=1)
set_target_properties(syn_compiler_api PROPERTIES
  VERSION ${SYNPS_VERSION}
  SOVERSION 1
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)
if(UNIX AND NOT APPLE)
  # LLVM/IREE are linked statically into the compiler service. The version
  # script exposes only the graph-host symbols consumed by package plugins.
  target_link_options(syn_compiler_api PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/compiler_api.map"
  )
  set_property(TARGET syn_compiler_api APPEND PROPERTY
    LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/compiler_api.map")
endif()
set_target_properties(syn_compiler syn_package_registry syn_weights_pack PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(syn_compiler PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${MLN_ROOT}/include
)

target_include_directories(syn_compiler SYSTEM PUBLIC
  ${IREE_ROOT}/third_party/llvm-project/mlir/include
  ${IREE_ROOT}/third_party/llvm-project/llvm/include
  ${IREE_ROOT}/third_party/stablehlo
  ${IREE_ROOT}/compiler/src
  ${CMAKE_BINARY_DIR}/third_party/iree/llvm-project/tools/mlir/include
  ${CMAKE_BINARY_DIR}/third_party/iree/llvm-project/include
  ${CMAKE_BINARY_DIR}/third_party/iree/llvm-external-projects/stablehlo
  ${CMAKE_BINARY_DIR}/third_party/iree/compiler/src
)

target_include_directories(syn_compiler SYSTEM PRIVATE
  ${IREE_ROOT}/compiler/bindings/c
)

set(MLN_IREE_COMPILER_LIBRARY "" CACHE FILEPATH
  "Optional path to a matching vendored libIREECompiler shared library")
if(MLN_IREE_COMPILER_LIBRARY)
  add_library(mln_iree_compiler_shared SHARED IMPORTED)
  set_target_properties(mln_iree_compiler_shared PROPERTIES
    IMPORTED_LOCATION "${MLN_IREE_COMPILER_LIBRARY}"
  )
  set(MLN_IREE_COMPILER_TARGET mln_iree_compiler_shared)
else()
  # Backend packages extend IREE's process-wide target registry. The shared C
  # API intentionally hides its C++ registry, so an independently linked
  # TargetBackend would register into a different copy. Keep the official IREE
  # static embedding boundary here so imported backends and the C API share the
  # same registry instance.
  set(MLN_IREE_COMPILER_TARGET iree_compiler_API_StaticImpl)
endif()

target_link_libraries(syn_compiler PUBLIC
  mln_sdk
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRPass
  StablehloOps
  StablehloPasses
  iree_compiler_Dialect_Util_IR_IR
  iree_compiler_Dialect_Flow_IR_IR
  iree_compiler_Dialect_Stream_IR_IR
  iree_compiler_Dialect_HAL_IR_IR
  iree_compiler_Dialect_HAL_IR_HALDialect
  iree_compiler_Dialect_HAL_Target_Target
  iree_compiler_Dialect_TensorExt_IR_IR
  iree_compiler_Dialect_LinalgExt_IR_IR
  ${MLN_IREE_COMPILER_TARGET}
)

target_link_libraries(syn_compiler PUBLIC
  syn_weights_pack
  mln_package
)

target_compile_features(syn_compiler PUBLIC cxx_std_23)

if(SYN_BUILD_TESTS AND EXISTS "${CMAKE_SOURCE_DIR}/tests/compiler")
  add_executable(syn_lowering_test "${CMAKE_SOURCE_DIR}/tests/compiler/lowering_test.cpp")
  target_link_libraries(syn_lowering_test PRIVATE syn_compiler)
  add_test(NAME syn_lowering COMMAND syn_lowering_test)
  add_executable(syn_report_test "${CMAKE_SOURCE_DIR}/tests/compiler/report_test.cpp")
  target_link_libraries(syn_report_test PRIVATE syn_compiler)
  add_test(NAME syn_report COMMAND syn_report_test)
  set_tests_properties(syn_report PROPERTIES TIMEOUT 10)
  add_executable(syn_compatibility_test
    "${CMAKE_SOURCE_DIR}/tests/compiler/compatibility_test.cpp")
  target_link_libraries(syn_compatibility_test PRIVATE syn_compiler)
  add_test(NAME syn_compatibility COMMAND syn_compatibility_test)
  set_tests_properties(syn_compatibility PROPERTIES TIMEOUT 10)
  add_executable(syn_parallel_compile_test
    "${CMAKE_SOURCE_DIR}/tests/compiler/parallel_compile_test.cpp")
  target_link_libraries(syn_parallel_compile_test PRIVATE syn_compiler)
  add_test(NAME syn_parallel_compile COMMAND syn_parallel_compile_test
    "${CMAKE_CURRENT_BINARY_DIR}/test-parallel-compile")
  set_tests_properties(syn_parallel_compile PROPERTIES TIMEOUT 30)
  add_executable(syn_graph_factory_test
    "${CMAKE_SOURCE_DIR}/tests/compiler/graph_factory_test.cpp")
  target_link_libraries(syn_graph_factory_test PRIVATE syn_compiler)
  add_test(NAME syn_graph_factory COMMAND syn_graph_factory_test)
  add_executable(syn_compiler_abi_test
    "${CMAKE_SOURCE_DIR}/tests/compiler/compiler_abi_test.cpp")
  target_include_directories(syn_compiler_abi_test PRIVATE "${CMAKE_SOURCE_DIR}/include")
  target_link_libraries(syn_compiler_abi_test PRIVATE syn_compiler_api)
  add_test(NAME syn_compiler_abi COMMAND syn_compiler_abi_test)
  if(SYN_ENABLE_VULKAN)
    add_executable(syn_multi_device_test
      "${CMAKE_SOURCE_DIR}/tests/compiler/multi_device_test.cpp")
    target_link_libraries(syn_multi_device_test PRIVATE syn_compiler)
    add_test(NAME syn_multi_device COMMAND syn_multi_device_test
      "${CMAKE_CURRENT_BINARY_DIR}/test-multi-device")
    set_tests_properties(syn_multi_device PROPERTIES TIMEOUT 30)
  endif()
  if(SYN_ENABLE_VULKAN AND
     EXISTS "${MLN_ROOT}/../Compatibilities/Vulkan/src/vulkan.cpp" AND
     EXISTS "${MLN_ROOT}/../Compatibilities/Adreno643/src/adreno.cpp" AND
     EXISTS "${MLN_ROOT}/../Compatibilities/LLVM/src/llvm.cpp")
    add_executable(syn_delta_test "${CMAKE_SOURCE_DIR}/tests/compiler/delta_test.cpp"
      "${MLN_ROOT}/../Compatibilities/Vulkan/src/vulkan.cpp"
      "${MLN_ROOT}/../Compatibilities/Adreno643/src/adreno.cpp"
      "${MLN_ROOT}/../Compatibilities/LLVM/src/llvm.cpp")
    target_link_libraries(syn_delta_test PRIVATE syn_compiler)
    target_include_directories(syn_delta_test PRIVATE
      "${MLN_ROOT}/../Quantizations/F32/include"
      "${MLN_ROOT}/../Quantizations/.shared/include")
    add_test(NAME syn_delta COMMAND syn_delta_test "${CMAKE_BINARY_DIR}/delta-validation")
    add_test(NAME syn_compatibility_selection COMMAND syn_delta_test --selection)
    set_tests_properties(syn_compatibility_selection PROPERTIES TIMEOUT 10)
    set_tests_properties(syn_delta PROPERTIES TIMEOUT 60)
  endif()
  add_executable(mln_quant_test "${CMAKE_SOURCE_DIR}/tests/compiler/quant_test.cpp")
target_link_libraries(mln_quant_test PRIVATE syn_compiler)
target_compile_features(mln_quant_test PRIVATE cxx_std_23)

set(PLEX_QUANTIZATIONS_ROOT "${MLN_ROOT}/../Quantizations" CACHE PATH
    "Optional path to the PLeX Quantizations source tree")
if(EXISTS "${PLEX_QUANTIZATIONS_ROOT}/tests/real_quant_graph_test.cpp")
  add_executable(mln_real_quant_graph_test
    "${PLEX_QUANTIZATIONS_ROOT}/tests/real_quant_graph_test.cpp"
  )
  target_include_directories(mln_real_quant_graph_test PRIVATE
    "${PLEX_QUANTIZATIONS_ROOT}/.shared/include"
    "${PLEX_QUANTIZATIONS_ROOT}/Q4K/include"
    "${PLEX_QUANTIZATIONS_ROOT}/Q5K/include"
    "${PLEX_QUANTIZATIONS_ROOT}/Q6K/include"
    "${PLEX_QUANTIZATIONS_ROOT}/Q8/include"
  )
  target_link_libraries(mln_real_quant_graph_test PRIVATE syn_compiler)
  target_compile_features(mln_real_quant_graph_test PRIVATE cxx_std_23)
  add_executable(mln_tiled_matmul_test
    "${PLEX_QUANTIZATIONS_ROOT}/tests/tiled_matmul_test.cpp")
  target_include_directories(mln_tiled_matmul_test PRIVATE
    "${PLEX_QUANTIZATIONS_ROOT}/.shared/include"
    "${PLEX_QUANTIZATIONS_ROOT}/Q4K/include"
    "${PLEX_QUANTIZATIONS_ROOT}/Q8/include")
  target_link_libraries(mln_tiled_matmul_test PRIVATE syn_compiler
    iree::compiler::Codegen::Dialect::GPU::TargetUtils::KnownTargets
    iree::compiler::Codegen::Dialect::GPU::IR::IREEGPUDialect
    iree::compiler::Codegen::Utils)
  target_compile_features(mln_tiled_matmul_test PRIVATE cxx_std_23)
  add_test(NAME mln_tiled_matmul COMMAND mln_tiled_matmul_test
    "${CMAKE_CURRENT_BINARY_DIR}/test-tiled-matmul")
  set_tests_properties(mln_tiled_matmul PROPERTIES TIMEOUT 60)
  if(SYN_ENABLE_VULKAN)
    add_test(NAME mln_tiled_matmul_adreno COMMAND mln_tiled_matmul_test
      "${CMAKE_CURRENT_BINARY_DIR}/test-tiled-adreno" adreno-643 6144 128)
    set_tests_properties(mln_tiled_matmul_adreno PROPERTIES TIMEOUT 60)
  endif()
endif()

add_executable(mln_ops_test "${CMAKE_SOURCE_DIR}/tests/compiler/ops_test.cpp")
target_link_libraries(mln_ops_test PRIVATE syn_compiler)
target_compile_features(mln_ops_test PRIVATE cxx_std_23)

add_executable(mln_replace_vmfb_test
  "${CMAKE_SOURCE_DIR}/tests/compiler/replace_vmfb.cpp")
target_link_libraries(mln_replace_vmfb_test PRIVATE mln_package)
target_compile_features(mln_replace_vmfb_test PRIVATE cxx_std_23)

add_executable(mln_artifact_test
  "${CMAKE_SOURCE_DIR}/tests/compiler/artifact_test.cpp"
  artifact.cpp
)
target_include_directories(mln_artifact_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${MLN_ROOT}/include
)
target_compile_features(mln_artifact_test PRIVATE cxx_std_23)

add_library(mln_package_fixture SHARED "${CMAKE_SOURCE_DIR}/tests/compiler/package_fixture.cpp")
target_include_directories(
    mln_package_fixture
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${MLN_ROOT}/include
)
target_compile_features(mln_package_fixture PRIVATE cxx_std_23)

add_executable(mln_json_test "${CMAKE_SOURCE_DIR}/tests/compiler/json_test.cpp")
target_include_directories(mln_json_test PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MLN_ROOT}/include
)
target_compile_features(mln_json_test PRIVATE cxx_std_23)

add_executable(syn_package_registry_test
  "${CMAKE_SOURCE_DIR}/tests/compiler/package_registry_test.cpp")
target_link_libraries(syn_package_registry_test PRIVATE syn_package_registry)
target_compile_features(syn_package_registry_test PRIVATE cxx_std_23)

add_test(NAME mln_quant_compile COMMAND mln_quant_test
  ${CMAKE_CURRENT_BINARY_DIR}/test-quant.vmfb
  ${CMAKE_CURRENT_BINARY_DIR}/test-quant.devices
  ${CMAKE_CURRENT_BINARY_DIR}/test-quant.weights)
if(TARGET mln_real_quant_graph_test)
  add_test(NAME mln_real_quant_graph COMMAND mln_real_quant_graph_test
    ${CMAKE_CURRENT_BINARY_DIR}/test-real-quant.vmfb
    ${CMAKE_CURRENT_BINARY_DIR}/test-real-quant.devices
    ${CMAKE_CURRENT_BINARY_DIR}/test-real-quant.weights
    ${CMAKE_CURRENT_BINARY_DIR}/test-real-quant.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/test-real-quant-executables)
endif()
add_test(NAME mln_ops_compile COMMAND mln_ops_test
  ${CMAKE_CURRENT_BINARY_DIR}/test-ops.vmfb
  ${CMAKE_CURRENT_BINARY_DIR}/test-ops.devices)
add_test(NAME mln_artifact_plan COMMAND mln_artifact_test
  ${CMAKE_CURRENT_BINARY_DIR}/artifact-test)
add_test(NAME mln_json COMMAND mln_json_test)
add_test(NAME syn_package_registry COMMAND syn_package_registry_test
  $<TARGET_FILE:mln_package_fixture>)
endif()
