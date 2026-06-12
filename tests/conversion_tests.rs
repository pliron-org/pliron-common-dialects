//! Tests for verifying the correctness of [CFToLLVM] rewrite patterns
//! that convert control flow operations to their LLVM IR counterparts.

use pliron::{
    builtin::ops::ModuleOp,
    combine::Parser,
    context::Context,
    init_env_logger_for_tests, input_error_noloc,
    irbuild::dialect_conversion::apply_dialect_conversion,
    irfmt::parsers::spaced,
    location,
    op::{Op, verify_op},
    operation::Operation,
    parsable::{self, state_stream_from_iterator},
    printable::Printable,
    result::ExpectOk,
};
use pliron_llvm::llvm_sys::{core::LLVMContext, lljit::LLVMLLJIT, target::initialize_native};

use expect_test::expect;
use pliron_common_dialects::cf::to_llvm::CFToLLVM;

#[test]
fn test_for_op_to_llvm_conversion() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_for: llvm.func <builtin.fp32 () variadic = false> [] {
                  ^entry():
                    c0 = index.constant <index.constant 0> : index.index;
                    c10 = index.constant <index.constant 10> : index.index;
                    c1 = index.constant <index.constant 1> : index.index;
                    init = builtin.constant <builtin.single 1.0> : builtin.fp32;
                    inc = builtin.constant <builtin.single 3.5> : builtin.fp32;
                    
                    result = cf.for c0 to c10 step c1 (init) {
                        ^entry(iv : index.index, iter_arg : builtin.fp32):
                            next = llvm.fadd <FAST> iter_arg, inc : builtin.fp32;
                            cf.yield next
                    };
                    
                    llvm.return result
                }
            }
            "#;

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));

    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let print_parsed = format!("{}", module_op.get_operation().disp(ctx));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block3v1() !0:
            llvm.func @test_for: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                v10 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64 !2;
                v11 = builtin.constant <builtin.integer <10: i64>> : builtin.integer i64 !3;
                v12 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !4;
                init_v3 = builtin.constant <builtin.single 1> : builtin.fp32  !5;
                inc_v4 = builtin.constant <builtin.single 3.5> : builtin.fp32  !6;
                llvm.br ^for_op_header_block5v1(v10, init_v3)

              ^for_op_header_block5v1(v13: builtin.integer i64, v14: builtin.fp32 ) !7:
                v15 = llvm.icmp v13 <ULT> v11 : builtin.integer i1;
                llvm.cond_br if v15 ^entry_block1v1(v13, v14) else ^entry_split_block4v1()

              ^entry_block1v1(iv_v8: builtin.integer i64, iter_arg_v9: builtin.fp32 ) !8:
                next_v7 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> iter_arg_v9, inc_v4 : builtin.fp32  !9;
                v16 = llvm.add iv_v8, v12 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block5v1(v16, next_v7)

              ^entry_split_block4v1():
                llvm.return v14 !10
            } !11
        } !12

        outlined_attributes:
        !0 = @[<in-memory>: line: 3, column: 15], []
        !1 = @[<in-memory>: line: 5, column: 19], []
        !2 = @[<in-memory>: line: 6, column: 21], []
        !3 = @[<in-memory>: line: 7, column: 21], []
        !4 = @[<in-memory>: line: 8, column: 21], []
        !5 = @[<in-memory>: line: 9, column: 21], [builtin_debug_info = builtin.debug_info [init]]
        !6 = @[<in-memory>: line: 10, column: 21], [builtin_debug_info = builtin.debug_info [inc]]
        !7 = @[<in-memory>: line: 12, column: 21], []
        !8 = @[<in-memory>: line: 13, column: 25], [builtin_debug_info = builtin.debug_info [iv, iter_arg]]
        !9 = @[<in-memory>: line: 14, column: 29], [builtin_debug_info = builtin.debug_info [next]]
        !10 = @[<in-memory>: line: 18, column: 21], []
        !11 = @[<in-memory>: line: 4, column: 17], []
        !12 = @[<in-memory>: line: 2, column: 13], []
    "#]].assert_eq(&print_parsed);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    expect![[r#"
        ; ModuleID = 'test_module'
        source_filename = "test_module"

        define float @test_for() {
        entry_block2v1:
          br label %for_op_header_block5v1

        for_op_header_block5v1:                           ; preds = %entry_block1v1, %entry_block2v1
          %v13 = phi i64 [ 0, %entry_block2v1 ], [ %v16, %entry_block1v1 ]
          %v14 = phi float [ 1.000000e+00, %entry_block2v1 ], [ %next_v7, %entry_block1v1 ]
          %v15 = icmp ult i64 %v13, 10
          br i1 %v15, label %entry_block1v1, label %entry_split_block4v1

        entry_block1v1:                                   ; preds = %for_op_header_block5v1
          %iv_v8 = phi i64 [ %v13, %for_op_header_block5v1 ]
          %iter_arg_v9 = phi float [ %v14, %for_op_header_block5v1 ]
          %next_v7 = fadd fast float %iter_arg_v9, 3.500000e+00
          %v16 = add i64 %iv_v8, 1
          br label %for_op_header_block5v1

        entry_split_block4v1:                             ; preds = %for_op_header_block5v1
          ret float %v14
        }
    "#]]
    .assert_eq(&llvm_ir.to_string());

    // Let's try and execute this function
    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_for")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn() -> f32>(symbol_addr) };
    let result = f();
    assert_eq!(result, 36.0);
}

// Test [NDForOp] to LLVM conversion with multiple loop dimensions.
#[test]
fn test_ndfor_op_to_llvm_conversion() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_ndfor: llvm.func <builtin.fp32 () variadic = false> [] {
                  ^entry():
                    c0 = index.constant <index.constant 0> : index.index;
                    c10 = index.constant <index.constant 10> : index.index;
                    c11 = index.constant <index.constant 11> : index.index;
                    c1 = index.constant <index.constant 1> : index.index;
                    c1_0 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                    accum = llvm.alloca [builtin.fp32 x c1_0] : llvm.ptr(0);
                    f0 = builtin.constant <builtin.single 0.0> : builtin.fp32;
                    f1 = builtin.constant <builtin.single 1.5> : builtin.fp32;
                    llvm.store *accum <- f0;

                    cf.nd_for [c0, c0] to [c10, c11] step [c1, c1] {
                        ^entry(i : index.index, j : index.index):
                            accum_val = llvm.load accum : builtin.fp32;
                            sum = llvm.fadd <FAST> accum_val, f1 : builtin.fp32;
                            llvm.store *accum <- sum;
                            cf.yield
                    };
                    
                    result = llvm.load accum : builtin.fp32;
                    llvm.return result
                }
            }
            "#;

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));

    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let print_parsed = format!("{}", module_op.get_operation().disp(ctx));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block3v1() !0:
            llvm.func @test_ndfor: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                v13 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64 !2;
                v14 = builtin.constant <builtin.integer <10: i64>> : builtin.integer i64 !3;
                v15 = builtin.constant <builtin.integer <11: i64>> : builtin.integer i64 !4;
                v16 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !5;
                c1_0_v4 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !6;
                accum_v5 = llvm.alloca [builtin.fp32  x c1_0_v4]  : llvm.ptr (0) !7;
                f0_v6 = builtin.constant <builtin.single 0> : builtin.fp32  !8;
                f1_v7 = builtin.constant <builtin.single 1.5> : builtin.fp32  !9;
                llvm.store *accum_v5 <- f0_v6  !10;
                llvm.br ^for_op_header_block9v1(v13)

              ^for_op_header_block9v1(v22: builtin.integer i64) !11:
                v23 = llvm.icmp v22 <ULT> v14 : builtin.integer i1;
                llvm.cond_br if v23 ^entry_block5v1(v22) else ^entry_split_block8v1()

              ^entry_block5v1(iv_v18: builtin.integer i64) !12:
                llvm.br ^for_op_header_block7v1(v13)

              ^for_op_header_block7v1(v19: builtin.integer i64):
                v20 = llvm.icmp v19 <ULT> v15 : builtin.integer i1;
                llvm.cond_br if v20 ^entry_block4v1(v19) else ^entry_split_block6v1()

              ^entry_block4v1(iv_v17: builtin.integer i64) !13:
                llvm.br ^entry_block1v1(iv_v18, iv_v17)

              ^entry_block1v1(i_v10: builtin.integer i64, j_v11: builtin.integer i64) !14:
                accum_val_v8 = llvm.load accum_v5  : builtin.fp32  !15;
                sum_v9 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> accum_val_v8, f1_v7 : builtin.fp32  !16;
                llvm.store *accum_v5 <- sum_v9  !17;
                v21 = llvm.add iv_v17, v16 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block7v1(v21)

              ^entry_split_block6v1():
                v24 = llvm.add iv_v18, v16 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block9v1(v24)

              ^entry_split_block8v1():
                result_v12 = llvm.load accum_v5  : builtin.fp32  !18;
                llvm.return result_v12 !19
            } !20
        } !21

        outlined_attributes:
        !0 = @[<in-memory>: line: 3, column: 15], []
        !1 = @[<in-memory>: line: 5, column: 19], []
        !2 = @[<in-memory>: line: 6, column: 21], []
        !3 = @[<in-memory>: line: 7, column: 21], []
        !4 = @[<in-memory>: line: 8, column: 21], []
        !5 = @[<in-memory>: line: 9, column: 21], []
        !6 = @[<in-memory>: line: 10, column: 21], [builtin_debug_info = builtin.debug_info [c1_0]]
        !7 = @[<in-memory>: line: 11, column: 21], [builtin_debug_info = builtin.debug_info [accum]]
        !8 = @[<in-memory>: line: 12, column: 21], [builtin_debug_info = builtin.debug_info [f0]]
        !9 = @[<in-memory>: line: 13, column: 21], [builtin_debug_info = builtin.debug_info [f1]]
        !10 = @[<in-memory>: line: 14, column: 21], []
        !11 = @[<in-memory>: line: 16, column: 21], []
        !12 = [builtin_debug_info = builtin.debug_info [iv]]
        !13 = [builtin_debug_info = builtin.debug_info [iv]]
        !14 = @[<in-memory>: line: 17, column: 25], [builtin_debug_info = builtin.debug_info [i, j]]
        !15 = @[<in-memory>: line: 18, column: 29], [builtin_debug_info = builtin.debug_info [accum_val]]
        !16 = @[<in-memory>: line: 19, column: 29], [builtin_debug_info = builtin.debug_info [sum]]
        !17 = @[<in-memory>: line: 20, column: 29], []
        !18 = @[<in-memory>: line: 24, column: 21], [builtin_debug_info = builtin.debug_info [result]]
        !19 = @[<in-memory>: line: 25, column: 21], []
        !20 = @[<in-memory>: line: 4, column: 17], []
        !21 = @[<in-memory>: line: 2, column: 13], []
    "#]].assert_eq(&print_parsed);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    expect![[r#"
        ; ModuleID = 'test_module'
        source_filename = "test_module"

        define float @test_ndfor() {
        entry_block2v1:
          %accum_v5 = alloca float, i64 1, align 4
          store float 0.000000e+00, ptr %accum_v5, align 4
          br label %for_op_header_block9v1

        for_op_header_block9v1:                           ; preds = %entry_split_block6v1, %entry_block2v1
          %v22 = phi i64 [ 0, %entry_block2v1 ], [ %v24, %entry_split_block6v1 ]
          %v23 = icmp ult i64 %v22, 10
          br i1 %v23, label %entry_block5v1, label %entry_split_block8v1

        entry_block5v1:                                   ; preds = %for_op_header_block9v1
          %iv_v18 = phi i64 [ %v22, %for_op_header_block9v1 ]
          br label %for_op_header_block7v1

        for_op_header_block7v1:                           ; preds = %entry_block1v1, %entry_block5v1
          %v19 = phi i64 [ 0, %entry_block5v1 ], [ %v21, %entry_block1v1 ]
          %v20 = icmp ult i64 %v19, 11
          br i1 %v20, label %entry_block4v1, label %entry_split_block6v1

        entry_block4v1:                                   ; preds = %for_op_header_block7v1
          %iv_v17 = phi i64 [ %v19, %for_op_header_block7v1 ]
          br label %entry_block1v1

        entry_block1v1:                                   ; preds = %entry_block4v1
          %i_v10 = phi i64 [ %iv_v18, %entry_block4v1 ]
          %j_v11 = phi i64 [ %iv_v17, %entry_block4v1 ]
          %accum_val_v8 = load float, ptr %accum_v5, align 4
          %sum_v9 = fadd fast float %accum_val_v8, 1.500000e+00
          store float %sum_v9, ptr %accum_v5, align 4
          %v21 = add i64 %iv_v17, 1
          br label %for_op_header_block7v1

        entry_split_block6v1:                             ; preds = %for_op_header_block7v1
          %v24 = add i64 %iv_v18, 1
          br label %for_op_header_block9v1

        entry_split_block8v1:                             ; preds = %for_op_header_block9v1
          %result_v12 = load float, ptr %accum_v5, align 4
          ret float %result_v12
        }
    "#]].assert_eq(&llvm_ir.to_string());

    // Let's try and execute this function
    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_ndfor")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn() -> f32>(symbol_addr) };
    let result = f();
    // The loop iterates 10 * 11 = 110 times, and each time it adds 1.5 to the accumulator, so the final result should be 165.0
    assert_eq!(result, 165.0);
}
