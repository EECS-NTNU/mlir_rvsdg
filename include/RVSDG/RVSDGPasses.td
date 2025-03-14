include "mlir/Pass/PassBase.td"
include "RVSDG/RVSDGOps.td"

def RVSDG_PrintExportPass: PassBase<"rvsdg-print-export", "::mlir::Pass"> {
    let summary = "Print RVSDG in export format";
    let description = [{
        This pass prints the RVSDG in export format. Proof of concept for future
        pass development.
    }];

    let dependentDialects = ["mlir::rvsdg::RVSDGDialect"];
}

def RVSDG_ImportPolygeistPass: PassBase<"rvsdg-import-polygeist", "::mlir::Pass"> {
    let summary = "Import Polygeist to RVSDG";
    let description = [{
        This pass imports Polygeist to RVSDG.
    }];

    let dependentDialects = [
        "mlir::rvsdg::RVSDGDialect",
        "func::FuncDialect",
        "LLVM::LLVMDialect",
        "memref::MemRefDialect",
        "arith::ArithDialect",
        "scf::SCFDialect",
        "polygeist::PolygeistDialect",
    ];
}