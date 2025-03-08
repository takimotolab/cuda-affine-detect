#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/InitAllDialects.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <polygeist/Dialect.h>

#include <iostream>
#include <string>

int main(int argc, char **argv) {
  // コマンドライン引数を確認
  if (argc < 2) {
    std::cerr << "Usage: caf <file-path>" << std::endl;
    return 1;
  }
  std::string inputFilename(argv[1]);

  // 入力ファイルを開く
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return 1;
  }

  // MLIRContextの初期化
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<
    mlir::affine::AffineDialect,
    mlir::arith::ArithDialect,
    mlir::async::AsyncDialect,
    mlir::cf::ControlFlowDialect,
    mlir::DLTIDialect,
    mlir::func::FuncDialect,
    mlir::gpu::GPUDialect,
    mlir::index::IndexDialect,
    mlir::linalg::LinalgDialect,
    mlir::LLVM::LLVMDialect,
    mlir::math::MathDialect,
    mlir::memref::MemRefDialect,
    mlir::NVVM::NVVMDialect,
    mlir::omp::OpenMPDialect,
    mlir::polygeist::PolygeistDialect,
    mlir::pdl::PDLDialect,
    mlir::quant::QuantizationDialect,
    mlir::scf::SCFDialect,
    mlir::shape::ShapeDialect,
    mlir::sparse_tensor::SparseTensorDialect,
    mlir::spirv::SPIRVDialect,
    mlir::tensor::TensorDialect,
    mlir::tosa::TosaDialect,
    mlir::transform::TransformDialect,
    mlir::vector::VectorDialect
  >();
  context.appendDialectRegistry(registry);

  // パース
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file: " << inputFilename << "\n";
    return 1;
  }

  // 各関数について走査
  module->walk([](mlir::func::FuncOp funcOp) {
    const auto name = funcOp.getName().str();
    if (!name.starts_with("__device_stub__")) {
      return;
    }

    bool hasAffine = false;

    funcOp.walk([&hasAffine](mlir::Operation *op) {
      if (llvm::isa<
        mlir::affine::AffineForOp,
        mlir::affine::AffineIfOp,
        mlir::affine::AffineLoadOp, 
        mlir::affine::AffineStoreOp,
        mlir::affine::AffineApplyOp
      >(op)) {
        hasAffine = true;
      }
    });

    if (hasAffine) {
      std::cout << name << std::endl;
    }
  });

  return 0;
}
