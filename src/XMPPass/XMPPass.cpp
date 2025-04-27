#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
// Use LLVM-17 location for Dominators:
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/AssumptionCache.h" // For AssumptionCacheTracker

#include <algorithm>
#include <cxxabi.h>
#include <math.h>
#include <queue>
#include <string>
#include <set>
#include <vector>

using namespace llvm;

bool DEBUG_MODE = false;

std::vector<BasicBlock*> nmodBasicBlocks = {};
std::vector<std::string> visitedFunctions = {};
std::vector<Loop*> visitedLoops = {};

static cl::opt<unsigned> ci_loopBodyUnrollSize("loop_body_size",
    cl::desc("Loop body unroll size in number of LLVM instructions"),
    cl::value_desc("unroll size"), cl::init(0));
static cl::opt<int> ci_enableInstrumentation("c_instrument",
    cl::desc("Enable instrumentation"),
    cl::value_desc("enable instrumentation"), cl::init(1));
static cl::opt<int> ci_modifiedSubLoops("modified_subloops",
    cl::desc("Number of modified subloops"),
    cl::value_desc("modified subloops"), cl::init(200));
static cl::opt<int> ci_disableBoundedLoops("disable_bounded_loops",
    cl::desc("Disable bounded loops"),
    cl::value_desc("disable bounded loops"), cl::init(0));

namespace {

  struct XMPPass : public ModulePass {
    static char ID;
    long loopCount = 0;
    long loopBodyUnrollSize = 0;
    int enableInstrumentation = 1;
    long modifiedSubLoops = 0;
    int disableBoundedLoops = 1;

    std::set<Function*> annotFuncs;

    XMPPass() : ModulePass(ID) {}

    void getAnalysisUsage(AnalysisUsage& AU) const override {
      AU.addRequired<ScalarEvolutionWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
      AU.addRequired<PostDominatorTreeWrapperPass>();
      AU.addRequired<CallGraphWrapperPass>();
      AU.addPreserved<CallGraphWrapperPass>();
      // In LLVM 17, use AssumptionCacheTracker (which requires a Function argument).
      AU.addRequired<AssumptionCacheTracker>();
    }

    virtual bool doInitialization(Module& M) override {
      getAnnotatedFunctions(&M);
      return false;
    }

    void getAnnotatedFunctions(Module* M) {
      for (Module::global_iterator I = M->global_begin(), E = M->global_end(); I != E; ++I) {
        if (I->getName() == "llvm.global.annotations") {
          ConstantArray* CA = dyn_cast<ConstantArray>(I->getOperand(0));
          for (auto OI = CA->op_begin(); OI != CA->op_end(); ++OI) {
            ConstantStruct* CS = dyn_cast<ConstantStruct>(OI->get());
            Function* FUNC = dyn_cast<Function>(CS->getOperand(0)->getOperand(0));
            GlobalVariable* AnnotationGL = dyn_cast<GlobalVariable>(CS->getOperand(1)->getOperand(0));
            StringRef annotation = dyn_cast<ConstantDataArray>(AnnotationGL->getInitializer())->getAsCString();
            if (annotation == "xmp_skip") {
              annotFuncs.insert(FUNC);
            }
          }
        }
      }
    }

    virtual bool runOnModule(Module& M) override {
      loopBodyUnrollSize = ci_loopBodyUnrollSize;
      enableInstrumentation = ci_enableInstrumentation;
      modifiedSubLoops = ci_modifiedSubLoops;
      disableBoundedLoops = ci_disableBoundedLoops;

      errs() << "========== Starting the analysis of the module ========== \n";
      errs() << "> Module Name: " << M.getName() << "\n";
      errs() << "> Enable instrumentation: " << enableInstrumentation << "\n";
      errs() << "> Size of each loop after unrolling: " << loopBodyUnrollSize << "\n";
      errs() << "> Modified subloops: " << modifiedSubLoops << "\n";
      errs() << "> Disable bounded loops: " << disableBoundedLoops << "\n";

      if(enableInstrumentation) {
        std::vector<Loop*> loops;

        for (auto& F : M) {
          if (shouldInstrumentFunc(F)) {
            errs() << "> Skipping function: '" << F.getName()
                   << "' because it has been annotated with xmp_skip attribute.\n";
            continue;
          }
          if (F.isDeclaration()) {
            errs() << "> This is declaration, skip " << F.getName() << "\n";
            continue;
          }

          std::string funcName = F.getName().str();
          std::string demangledFuncName = funcName;

          int status;
          char* demangled = abi::__cxa_demangle(demangledFuncName.c_str(), 0, 0, &status);
          if (status == 0) {
            demangledFuncName = demangled;
          }

          if (std::find(visitedFunctions.begin(), visitedFunctions.end(), demangledFuncName) != visitedFunctions.end())
            continue;

          visitedFunctions.push_back(demangledFuncName);

          LoopInfo& LI = getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
          auto& SE = getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
          DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();

          for (Loop* loop : LI) {
            int subLoopCounter = 0;
            std::queue<Loop*> worklist;
            worklist.push(loop);

            while (!worklist.empty()) {
              Loop* currentLoop = worklist.front();

              // Calculate unroll count.
              int cnt = loopBodyUnrollSize / estimateLoopBodySize(currentLoop, LI);
              // For LLVM 17, default-construct ULO then set its fields.
              UnrollLoopOptions ULO;
              ULO.Count = cnt;
              ULO.UnrollRemainder = true;
              // Remove ULO.AllowPartialUnroll since it does not exist.
              ULO.AllowExpensiveTripCount = true;

              // Use AssumptionCacheTracker: pass the function as argument.
              AssumptionCache* AC = &getAnalysis<AssumptionCacheTracker>(F).getAssumptionCache(F);

              // UnrollLoop in LLVM 17 takes 10 arguments.
              LoopUnrollResult result = UnrollLoop(currentLoop,
                                                   ULO,
                                                   &LI,
                                                   &SE,
                                                   &DT,
                                                   AC,
                                                   nullptr,   // TTI: use nullptr
                                                   nullptr,   // ORE: use nullptr
                                                   true,      // PreserveLCSSA
                                                   nullptr);  // Loop** out
              if (result == LoopUnrollResult::PartiallyUnrolled) {
                errs() << "Loop partially unrolled \n";
              } else if (result == LoopUnrollResult::FullyUnrolled) {
                errs() << "Loop fully unrolled \n";
              } else {
                errs() << "Loop not unrolled \n";
              }

              worklist.pop();

              instrumentLoop(currentLoop, F, M, LI, demangledFuncName);

              if (modifiedSubLoops != 0) {
                for (Loop::iterator SL = currentLoop->begin(), SLEnd = currentLoop->end(); SL != SLEnd; ++SL) {
                  if (subLoopCounter < modifiedSubLoops) {
                    worklist.push(*SL);
                    subLoopCounter++;
                  }
                }
              }
            }
          }
        }
        errs() << "> Unique loops: " << loops.size() << "\n";
      }
      errs() << "========== Finished the analysis of the module ========== \n";
      errs() << "========== Instrumented with Cache Line Pass ========== \n";
      return true;
    }

    bool shouldInstrumentFunc(Function& F) {
      return annotFuncs.find(&F) != annotFuncs.end();
    }

    bool instrumentLoop(Loop* loop, Function& F, Module& M, LoopInfo& LI, std::string demangledFuncName) {
      errs() << "Instrumenting loop in function: " << demangledFuncName << "\n";

      BasicBlock* entryBB = loop->getHeader();
      Instruction* firstInst = &*entryBB->getFirstInsertionPt();

      IRBuilder<> builder(firstInst);

      GlobalVariable* preemptNow = M.getGlobalVariable("preempt_signal");

      if (!preemptNow) {
        preemptNow = new GlobalVariable(M,
                                        Type::getInt8PtrTy(M.getContext()),
                                        false,
                                        GlobalValue::ExternalLinkage,
                                        0,
                                        "preempt_signal",
                                        0,
                                        GlobalValue::GeneralDynamicTLSModel,
                                        0);
      }

      // Use cast<PointerType> to get the element type.
      LoadInst* loadPreemptNow = builder.CreateLoad(cast<PointerType>(preemptNow->getType())->getContainedType(0), preemptNow, "loadPreemptNow");
      // Mutate type if needed.
      loadPreemptNow->mutateType(Type::getInt8PtrTy(M.getContext()));

      // Create a load for an i8 value.
      LoadInst *signalValue = builder.CreateLoad(Type::getInt8Ty(M.getContext()), loadPreemptNow, "signalValueLoad");
      ConstantInt* unsetSignalValue = ConstantInt::get(Type::getInt8Ty(M.getContext()), 0, false);

      Value* condition = builder.CreateICmpNE(signalValue, unsetSignalValue, "unset_signal_condition");

      // For SplitBlockAndInsertIfThen, cast nullptr to DomTreeUpdater*.
      Instruction* i = SplitBlockAndInsertIfThen(condition,
                                                 firstInst,
                                                 false,
                                                 nullptr,
                                                 (DomTreeUpdater*)nullptr,
                                                 &LI);
      BranchInst* br = dyn_cast<BranchInst>(entryBB->getTerminator());
      br->setMetadata("branch_weights", MDBuilder(M.getContext()).createBranchWeights(1, 10000));

      builder.SetInsertPoint(i);
      i->getParent()->setName("if_clock_fired");
      Function* concordFunc = M.getFunction("do_yield");

      if (!concordFunc) {
        FunctionType* FuncTy = FunctionType::get(IntegerType::get(M.getContext(), 32), true);
        concordFunc = Function::Create(FuncTy, GlobalValue::ExternalLinkage, "do_yield", M);
        concordFunc->setCallingConv(CallingConv::C);
      }

      builder.CreateCall(concordFunc, {});
      return false;
    }

    uint32_t estimateLoopBodySize(Loop* loop, LoopInfo& LI) {
      uint32_t estimatedInstCc = 0;
      for (Loop::block_iterator BI = loop->block_begin(), BE = loop->block_end(); BI != BE; ++BI) {
        BasicBlock* BB = *BI;
        for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE; ++II) {
          estimatedInstCc++;
        }
      }
      return estimatedInstCc;
    }

    bool instrumentFunction(Function& F, Module& M, std::string demangledFuncName) {
      errs() << "Instrumenting function: " << demangledFuncName << "\n";

      Instruction* firstInst = &*F.getEntryBlock().getFirstInsertionPt();

      IRBuilder<> builder(firstInst);

      GlobalVariable* preemptNow = M.getGlobalVariable("preempt_signal");

      if (!preemptNow) {
        preemptNow = new GlobalVariable(M,
                                        Type::getInt8Ty(M.getContext()),
                                        false,
                                        GlobalValue::ExternalLinkage,
                                        0,
                                        "preempt_signal",
                                        0,
                                        GlobalValue::GeneralDynamicTLSModel,
                                        0);
      }

      LoadInst* loadPreemptNow = builder.CreateLoad(cast<PointerType>(preemptNow->getType())->getContainedType(0), preemptNow, "loadPreemptNow");
      LoadInst *signalValue = builder.CreateLoad(Type::getInt8Ty(M.getContext()), loadPreemptNow, "signalValueLoad");
      ConstantInt* unsetSignalValue = ConstantInt::get(Type::getInt8Ty(M.getContext()), 0, false);

      BranchInst* br = dyn_cast<BranchInst>(F.getEntryBlock().getTerminator());
      br->setMetadata("branch_weights", MDBuilder(M.getContext()).createBranchWeights(1, 10000));

      builder.CreateCall(M.getFunction("do_yield"), {});
      errs() << "Function instrumented \n";
      return true;
    }
  };
} // namespace

char XMPPass::ID = 0;
static RegisterPass<XMPPass> Y("yield", "Concord Pass", true, false);

static void registerXMPPass(const PassManagerBuilder&, legacy::PassManagerBase& PM) {
  PM.add(new XMPPass());
}

static RegisterStandardPasses RegisterXMPPass(PassManagerBuilder::EP_EarlyAsPossible, registerXMPPass);
