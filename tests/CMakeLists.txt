# The suite runs with lit, found in the LLVM installation this project is
# configured against; -DLLVM_EXTERNAL_LIT overrides where to look.
#
# LLVM's own discovery looks only in <install-prefix>/bin, which a packaged LLVM
# does not populate: Debian and Ubuntu ship lit as a python tree under the
# package's build/ directory. Every spelling a distribution is known to use is
# tried before giving up. lit that cannot run is a configure error, not a
# skipped test target.
get_llvm_lit_path(_rvsdg_lit_dir _rvsdg_lit_file ALLOW_EXTERNAL)
set(_rvsdg_lit "${_rvsdg_lit_dir}/${_rvsdg_lit_file}")

if(NOT EXISTS "${_rvsdg_lit}")
  foreach(candidate
      "${LLVM_INSTALL_PREFIX}/bin/llvm-lit"
      "${LLVM_INSTALL_PREFIX}/build/utils/lit/lit.py"
      "${LLVM_INSTALL_PREFIX}/build/utils/lit/llvm-lit.py")
    if(EXISTS "${candidate}")
      set(_rvsdg_lit "${candidate}")
      break()
    endif()
  endforeach()
endif()

# lit is also installed on its own, by pip, which is how the JLM tree's CI
# provides it. That puts it on the PATH rather than under an LLVM prefix, so the
# search above cannot reach it.
if(NOT EXISTS "${_rvsdg_lit}")
  find_program(_rvsdg_lit NAMES lit llvm-lit)
endif()

if(NOT EXISTS "${_rvsdg_lit}")
  message(FATAL_ERROR
    "lit was not found, and the dialect's regression tests need it.\n"
    "Looked in ${LLVM_INSTALL_PREFIX}/bin, in the LLVM package's build/utils/lit,\n"
    "and for lit on the PATH. Either install an LLVM tools package (llvm-18-tools\n"
    "or pip install lit~=18), or reconfigure with -DLLVM_EXTERNAL_LIT=<path-to-lit.py>.")
endif()

# lit.py is a module rather than an executable, so it needs an interpreter; a
# shipped llvm-lit has its own shebang and runs directly.
if(_rvsdg_lit MATCHES "\\.py$")
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(RVSDG_LIT_COMMAND "${Python3_EXECUTABLE}" "${_rvsdg_lit}")
else()
  set(RVSDG_LIT_COMMAND "${_rvsdg_lit}")
endif()
message(STATUS "Dialect regression tests use lit at ${_rvsdg_lit}")

# Where the driver under test lands, per LLVM_RUNTIME_OUTPUT_INTDIR in the root
# CMakeLists.
set(RVSDG_TOOLS_DIR "${LLVM_RUNTIME_OUTPUT_INTDIR}")

# The suite's sources stay in the checkout; only the site config is generated
# per build tree.
set(RVSDG_TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Built through ${RVSDG_LIT_COMMAND} rather than add_lit_testsuite(): LLVM's
# helper assembles its own command line from variables a standalone project has
# not defined, and looks for lit in the one place an apt-installed LLVM does not
# put it. The dependency edge is kept, so the check target always runs against a
# current build.
add_custom_target(check-rvsdg-dialect
  COMMAND ${RVSDG_LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR} ${LLVM_LIT_ARGS}
  COMMENT "Running the RVSDG dialect regression tests"
  DEPENDS
    rvsdg-opt
  USES_TERMINAL
)

set_target_properties(check-rvsdg-dialect PROPERTIES FOLDER "Tests")

# Register the same lit run with CTest so `ctest` and `ninja test` are the entry
# points, rather than only the project's own check target. This does not build
# anything — ctest assumes the tree is configured and the driver is current.
add_test(NAME dialect-regression-tests
  COMMAND ${RVSDG_LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR} ${LLVM_LIT_ARGS})
set_tests_properties(dialect-regression-tests PROPERTIES
  FAIL_REGULAR_EXPRESSION "Failing Tests|Segfaulting Tests"
  TIMEOUT 300
  LABELS "regression")
