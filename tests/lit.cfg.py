# -*- Python -*- -*-

# Configuration for the dialect's lit test suite.
#
# Values that depend on the build (tool directories, the location of this
# source tree) arrive through lit.site.cfg.py, which CMake generates from
# lit.site.cfg.py.in. Everything that is the same for every build lives here.

import os
import subprocess

import lit.formats
import lit.llvm

# Builds lit's LLVM support layer, which then discovers FileCheck and friends
# from config.llvm_tools_dir. initialize() rebinds lit.llvm.llvm_config from
# the module of that name to the LLVMConfig instance, so anything that uses it
# has to be read afterwards.
lit.llvm.initialize(lit_config, config)
from lit.llvm import llvm_config

# lit prints this name per test file.
config.name = "RVSDG"

# ShTest executes each RUN line with a real shell when one is available, which
# is what the redirections and pipelines below need.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

config.suffixes = [".mlir"]

config.excludes = ["Inputs"]

config.test_source_root = os.path.dirname(__file__)

# rvsdg-opt is the only driver these tests need. It is built into the runtime
# output directory of this project, not of LLVM, so it is substituted
# explicitly rather than discovered through the PATH.
rvsdg_opt = os.path.join(config.rvsdg_tools_dir, "rvsdg-opt")
if not os.path.exists(rvsdg_opt):
    raise ValueError(f"Missing required test tool: {rvsdg_opt}")
config.substitutions.append(("%rvsdg-opt", rvsdg_opt))

# FileCheck belongs to the LLVM installation this dialect was configured
# against. lit's default substitutions name it with a bare word-boundary
# pattern, because in an LLVM build it sits on the PATH and tests call it by
# name; fed '%FileCheck', that pattern replaces just the tool name and strands
# the percent sign. Register the %-prefixed form these tests write first, so it
# wins, and note that the path it expands to is immune to the bare pattern that
# follows: that one refuses a match preceded by a slash.
filecheck = os.path.join(config.llvm_tools_dir, "FileCheck")
if not os.path.exists(filecheck):
    raise ValueError(f"Missing required test tool: {filecheck}")
config.substitutions.append(("%FileCheck", filecheck))

# %python and the std::error_code messages.
llvm_config.use_default_substitutions()

# Two of these tests exercise code that is not in every build of the dialect, so
# they can only run against a driver that has it. Rather than name a commit or a
# branch, which goes stale the first time it is wrong, the config asks the built
# driver. Inputs/ holds one probe file per thing being asked about; they are not
# tests, and are excluded from collection by config.excludes.
#
# A feature that is not added here makes every test REQUIRING it UNSUPPORTED, so
# the suite still says what was not run instead of failing.
inputs_dir = os.path.join(os.path.dirname(__file__), "Inputs")


def driver_supports(op_name, probe):
    """True when the driver knows op_name.

    The probe parses a program that is not valid for that op, so the driver is
    expected to fail either way. What separates the two cases is which error it
    reaches first: an op that is not registered fails while the parser looks the
    name up, before the operands are ever examined.
    """
    probe_path = os.path.join(inputs_dir, probe)
    if not os.path.exists(probe_path):
        raise ValueError(f"Missing feature probe: {probe_path}")
    result = subprocess.run(
        [rvsdg_opt, probe_path], capture_output=True, text=True
    )
    return f"unregistered operation '{op_name}'" not in (result.stderr + result.stdout)


if driver_supports("jlm.memset", "memset_probe.mlir"):
    config.available_features.add("jlm-memset")

# phi_result is a different question, so the probe is a different shape: this
# program is valid only if phiResult accepts the function-type references a
# lambda node produces, so the driver either accepts the file or does not, and
# the exit status is the whole answer.
phi_probe = os.path.join(inputs_dir, "phi_probe.mlir")
if os.path.exists(phi_probe) and subprocess.run(
    [rvsdg_opt, phi_probe], capture_output=True
).returncode == 0:
    config.available_features.add("phi-result-typed-by-reference")
