// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Tests for verifying the correctness of [CFToLLVM] rewrite patterns
//! that convert control flow operations to their LLVM IR counterparts.

use pliron::{
    builtin::{op_interfaces::SingleBlockRegionVerifyErr, ops::ModuleOp},
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
use pliron_llvm::llvm_sys::{core::LLVMContext, lljit::SimpleJIT};

use expect_test::expect;
use pliron_common_dialects::cf::{
    op_interfaces::YieldingRegionsVerifyErr, ops::IfOpVerifyErr, to_llvm::CFToLLVM,
};

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
          ^entry_block1v1() !0:
            llvm.func @test_for: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                c0_v14 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64 !2;
                c10_v15 = llvm.constant <builtin.integer <10: i64>> : builtin.integer i64 !3;
                c1_v16 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64 !4;
                init_v12 = llvm.constant <builtin.single 1> : builtin.fp32  !5;
                inc_v13 = llvm.constant <builtin.single 3.5> : builtin.fp32  !6;
                llvm.br ^for_op_header_block5v1(c0_v14, init_v12)

              ^for_op_header_block5v1(v17: builtin.integer i64, result_v18: builtin.fp32 ) !7:
                v19 = llvm.icmp v17 <ULT> c10_v15 : builtin.integer i1;
                llvm.cond_br if v19 ^entry_block3v1(v17, result_v18) else ^entry_split_block4v1()

              ^entry_block3v1(iv_v6: builtin.integer i64, iter_arg_v7: builtin.fp32 ) !8:
                next_v8 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> iter_arg_v7, inc_v13 : builtin.fp32  !9;
                v20 = llvm.add iv_v6, c1_v16 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block5v1(v20, next_v8)

              ^entry_split_block4v1():
                llvm.return result_v18 !10
            } !11
        } !12

        outlined_attributes:
        !0 = @[<in-memory>: line: 3, column: 15], []
        !1 = @[<in-memory>: line: 5, column: 19], []
        !2 = @[<in-memory>: line: 6, column: 21], [builtin_given_names = builtin.given_names [c0]]
        !3 = @[<in-memory>: line: 7, column: 21], [builtin_given_names = builtin.given_names [c10]]
        !4 = @[<in-memory>: line: 8, column: 21], [builtin_given_names = builtin.given_names [c1]]
        !5 = @[<in-memory>: line: 9, column: 21], [builtin_given_names = builtin.given_names [init]]
        !6 = @[<in-memory>: line: 10, column: 21], [builtin_given_names = builtin.given_names [inc]]
        !7 = @[<in-memory>: line: 12, column: 21], [builtin_given_names = builtin.given_names [?, result]]
        !8 = @[<in-memory>: line: 13, column: 25], [builtin_given_names = builtin.given_names [iv, iter_arg]]
        !9 = @[<in-memory>: line: 14, column: 29], [builtin_given_names = builtin.given_names [next]]
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

        for_op_header_block5v1:                           ; preds = %entry_block3v1, %entry_block2v1
          %v17 = phi i64 [ 0, %entry_block2v1 ], [ %v20, %entry_block3v1 ]
          %result_v18 = phi float [ 1.000000e+00, %entry_block2v1 ], [ %next_v8, %entry_block3v1 ]
          %v19 = icmp ult i64 %v17, 10
          br i1 %v19, label %entry_block3v1, label %entry_split_block4v1

        entry_block3v1:                                   ; preds = %for_op_header_block5v1
          %iv_v6 = phi i64 [ %v17, %for_op_header_block5v1 ]
          %iter_arg_v7 = phi float [ %result_v18, %for_op_header_block5v1 ]
          %next_v8 = fadd fast float %iter_arg_v7, 3.500000e+00
          %v20 = add i64 %iv_v6, 1
          br label %for_op_header_block5v1

        entry_split_block4v1:                             ; preds = %for_op_header_block5v1
          ret float %result_v18
        }
    "#]]
    .assert_eq(&llvm_ir.to_string());

    // Let's try and execute this function
    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    let f =
        unsafe { jit.lookup_symbol::<fn() -> f32>("test_for") }.expect("Failed to lookup symbol");
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
          ^entry_block1v1() !0:
            llvm.func @test_ndfor: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                c0_v20 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64 !2;
                c10_v21 = llvm.constant <builtin.integer <10: i64>> : builtin.integer i64 !3;
                c11_v22 = llvm.constant <builtin.integer <11: i64>> : builtin.integer i64 !4;
                c1_v23 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64 !5;
                c1_0_v17 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64 !6;
                accum_v5 = llvm.alloca [builtin.fp32  x c1_0_v17]  : llvm.ptr (0) !7;
                f0_v18 = llvm.constant <builtin.single 0> : builtin.fp32  !8;
                f1_v19 = llvm.constant <builtin.single 1.5> : builtin.fp32  !9;
                llvm.store *accum_v5 <- f0_v18  !10;
                llvm.br ^for_op_header_block9v1(c0_v20)

              ^for_op_header_block9v1(v29: builtin.integer i64) !11:
                v30 = llvm.icmp v29 <ULT> c10_v21 : builtin.integer i1;
                llvm.cond_br if v30 ^entry_block5v1(v29) else ^entry_split_block8v1()

              ^entry_block5v1(iv_v25: builtin.integer i64) !12:
                llvm.br ^for_op_header_block7v1(c0_v20)

              ^for_op_header_block7v1(v26: builtin.integer i64):
                v27 = llvm.icmp v26 <ULT> c11_v22 : builtin.integer i1;
                llvm.cond_br if v27 ^entry_block4v1(v26) else ^entry_split_block6v1()

              ^entry_block4v1(iv_v24: builtin.integer i64) !13:
                llvm.br ^entry_block3v1(iv_v25, iv_v24)

              ^entry_block3v1(i_v8: builtin.integer i64, j_v9: builtin.integer i64) !14:
                accum_val_v10 = llvm.load accum_v5  : builtin.fp32  !15;
                sum_v11 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> accum_val_v10, f1_v19 : builtin.fp32  !16;
                llvm.store *accum_v5 <- sum_v11  !17;
                v28 = llvm.add iv_v24, c1_v23 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block7v1(v28)

              ^entry_split_block6v1():
                v31 = llvm.add iv_v25, c1_v23 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block9v1(v31)

              ^entry_split_block8v1():
                result_v12 = llvm.load accum_v5  : builtin.fp32  !18;
                llvm.return result_v12 !19
            } !20
        } !21

        outlined_attributes:
        !0 = @[<in-memory>: line: 3, column: 15], []
        !1 = @[<in-memory>: line: 5, column: 19], []
        !2 = @[<in-memory>: line: 6, column: 21], [builtin_given_names = builtin.given_names [c0]]
        !3 = @[<in-memory>: line: 7, column: 21], [builtin_given_names = builtin.given_names [c10]]
        !4 = @[<in-memory>: line: 8, column: 21], [builtin_given_names = builtin.given_names [c11]]
        !5 = @[<in-memory>: line: 9, column: 21], [builtin_given_names = builtin.given_names [c1]]
        !6 = @[<in-memory>: line: 10, column: 21], [builtin_given_names = builtin.given_names [c1_0]]
        !7 = @[<in-memory>: line: 11, column: 21], [builtin_given_names = builtin.given_names [accum]]
        !8 = @[<in-memory>: line: 12, column: 21], [builtin_given_names = builtin.given_names [f0]]
        !9 = @[<in-memory>: line: 13, column: 21], [builtin_given_names = builtin.given_names [f1]]
        !10 = @[<in-memory>: line: 14, column: 21], []
        !11 = @[<in-memory>: line: 16, column: 21], []
        !12 = [builtin_given_names = builtin.given_names [iv]]
        !13 = [builtin_given_names = builtin.given_names [iv]]
        !14 = @[<in-memory>: line: 17, column: 25], [builtin_given_names = builtin.given_names [i, j]]
        !15 = @[<in-memory>: line: 18, column: 29], [builtin_given_names = builtin.given_names [accum_val]]
        !16 = @[<in-memory>: line: 19, column: 29], [builtin_given_names = builtin.given_names [sum]]
        !17 = @[<in-memory>: line: 20, column: 29], []
        !18 = @[<in-memory>: line: 24, column: 21], [builtin_given_names = builtin.given_names [result]]
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
          %v29 = phi i64 [ 0, %entry_block2v1 ], [ %v31, %entry_split_block6v1 ]
          %v30 = icmp ult i64 %v29, 10
          br i1 %v30, label %entry_block5v1, label %entry_split_block8v1

        entry_block5v1:                                   ; preds = %for_op_header_block9v1
          %iv_v25 = phi i64 [ %v29, %for_op_header_block9v1 ]
          br label %for_op_header_block7v1

        for_op_header_block7v1:                           ; preds = %entry_block3v1, %entry_block5v1
          %v26 = phi i64 [ 0, %entry_block5v1 ], [ %v28, %entry_block3v1 ]
          %v27 = icmp ult i64 %v26, 11
          br i1 %v27, label %entry_block4v1, label %entry_split_block6v1

        entry_block4v1:                                   ; preds = %for_op_header_block7v1
          %iv_v24 = phi i64 [ %v26, %for_op_header_block7v1 ]
          br label %entry_block3v1

        entry_block3v1:                                   ; preds = %entry_block4v1
          %i_v8 = phi i64 [ %iv_v25, %entry_block4v1 ]
          %j_v9 = phi i64 [ %iv_v24, %entry_block4v1 ]
          %accum_val_v10 = load float, ptr %accum_v5, align 4
          %sum_v11 = fadd fast float %accum_val_v10, 1.500000e+00
          store float %sum_v11, ptr %accum_v5, align 4
          %v28 = add i64 %iv_v24, 1
          br label %for_op_header_block7v1

        entry_split_block6v1:                             ; preds = %for_op_header_block7v1
          %v31 = add i64 %iv_v25, 1
          br label %for_op_header_block9v1

        entry_split_block8v1:                             ; preds = %for_op_header_block9v1
          %result_v12 = load float, ptr %accum_v5, align 4
          ret float %result_v12
        }
    "#]].assert_eq(&llvm_ir.to_string());

    // Let's try and execute this function
    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    let f =
        unsafe { jit.lookup_symbol::<fn() -> f32>("test_ndfor") }.expect("Failed to lookup symbol");
    let result = f();
    // The loop iterates 10 * 11 = 110 times, and each time it adds 1.5 to the accumulator, so the final result should be 165.0
    assert_eq!(result, 165.0);
}

// A nested [ForOp]'s `iter_args_init` may itself be an argument of the
// enclosing (outer) `ForOp`'s entry block, which is still being parsed at
// that point. The inner op's result type must still be resolved correctly.
#[test]
fn test_nested_for_op_iter_arg_result_type() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
      builtin.module @test_module {
        ^entry():
          llvm.func @test_nested_for: llvm.func <builtin.fp32 () variadic = false> [] {
            ^entry():
              c0 = index.constant <index.constant 0> : index.index;
              c10 = index.constant <index.constant 10> : index.index;
              c1 = index.constant <index.constant 1> : index.index;
              init = builtin.constant <builtin.single 1.0> : builtin.fp32;
              inc = builtin.constant <builtin.single 3.5> : builtin.fp32;

              outer = cf.for c0 to c10 step c1 (init) {
                ^entry(iv : index.index, iter_arg : builtin.fp32):
                    inner = cf.for c0 to c10 step c1 (iter_arg) {
                        ^entry(iv2 : index.index, iter_arg2 : builtin.fp32):
                            next2 = llvm.fadd <FAST> iter_arg2, inc : builtin.fp32;
                            cf.yield next2
                    };
                    cf.yield inner
              };

              llvm.return outer
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

    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_nested_for: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                c0_v0 = index.constant <index.constant 0> : index.index  !2;
                c10_v1 = index.constant <index.constant 10> : index.index  !3;
                c1_v2 = index.constant <index.constant 1> : index.index  !4;
                init_v3 = builtin.constant <builtin.single 1> : builtin.fp32  !5;
                inc_v4 = builtin.constant <builtin.single 3.5> : builtin.fp32  !6;
                outer_v5 = cf.for c0_v0 to c10_v1 step c1_v2 (init_v3) 
                {
                  ^entry_block3v1(iv_v6: index.index , iter_arg_v7: builtin.fp32 ) !7:
                    inner_v8 = cf.for c0_v0 to c10_v1 step c1_v2 (iter_arg_v7) 
                    {
                      ^entry_block4v1(iv2_v9: index.index , iter_arg2_v10: builtin.fp32 ) !8:
                        next2_v11 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> iter_arg2_v10, inc_v4 : builtin.fp32  !9;
                        cf.yield next2_v11 !10
                    }
         !11;
                    cf.yield inner_v8 !12
                }
         !13;
                llvm.return outer_v5 !14
            } !15
        }"#]].assert_eq(&module_op.disp(ctx).to_string());

    verify_op(&module_op, ctx).expect_ok(ctx);
}

// [ForOp] and [NDForOp] bodies must be a single block: verify must reject a
// body region with more than one block.
#[test]
fn test_for_op_multi_block_body_rejected() {
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

                    result = cf.for c0 to c10 step c1 (init) {
                        ^entry(iv : index.index, iter_arg : builtin.fp32):
                            cf.br ^second(iter_arg)
                        ^second(a : builtin.fp32):
                            cf.yield a
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

    let err = verify_op(&module_op, ctx).unwrap_err();
    err.err
        .downcast_ref::<SingleBlockRegionVerifyErr>()
        .unwrap();
}

#[test]
fn test_execute_region_op_to_llvm_conversion() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
      builtin.module @test_module {
        ^entry():
          llvm.func @test_execute_region: llvm.func <builtin.fp32 () variadic = false> [] {
            ^entry():
              c0 = index.constant <index.constant 0> : index.index;
              c10 = index.constant <index.constant 10> : index.index;
              c1 = index.constant <index.constant 1> : index.index;
              init = builtin.constant <builtin.single 1.0> : builtin.fp32;
              inc = builtin.constant <builtin.single 3.5> : builtin.fp32;
              result = cf.for c0 to c10 step c1 (init) {
                  ^entry(iv : index.index, iter_arg : builtin.fp32):
                      next = cf.execute_region -> (builtin.fp32) {
                          ^entry():
                              tmp = llvm.fadd <FAST> iter_arg, inc : builtin.fp32;
                              cf.br ^exit(tmp)
                          ^exit(v : builtin.fp32):
                              cf.yield v
                      };
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

    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_execute_region: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                c0_v16 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64 !2;
                c10_v17 = llvm.constant <builtin.integer <10: i64>> : builtin.integer i64 !3;
                c1_v18 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64 !4;
                init_v14 = llvm.constant <builtin.single 1> : builtin.fp32  !5;
                inc_v15 = llvm.constant <builtin.single 3.5> : builtin.fp32  !6;
                llvm.br ^for_op_header_block7v1(c0_v16, init_v14)

              ^for_op_header_block7v1(v19: builtin.integer i64, result_v20: builtin.fp32 ) !7:
                v21 = llvm.icmp v19 <ULT> c10_v17 : builtin.integer i1;
                llvm.cond_br if v21 ^entry_block3v1(v19, result_v20) else ^entry_split_block5v3()

              ^entry_block3v1(iv_v6: builtin.integer i64, iter_arg_v7: builtin.fp32 ) !8:
                llvm.br ^entry_block4v1()

              ^entry_block4v1() !9:
                tmp_v9 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> iter_arg_v7, inc_v15 : builtin.fp32  !10;
                llvm.br ^exit_block6v1(tmp_v9) !11

              ^exit_block6v1(v_v10: builtin.fp32 ) !12:
                llvm.br ^block8v1(v_v10)

              ^block8v1(next_v23: builtin.fp32 ) !13:
                v22 = llvm.add iv_v6, c1_v18 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block7v1(v22, next_v23)

              ^entry_split_block5v3():
                llvm.return result_v20 !14
            } !15
        }"#]].assert_eq(&module_op.disp(ctx).to_string());
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    let f = unsafe { jit.lookup_symbol::<fn() -> f32>("test_execute_region") }
        .expect("Failed to lookup symbol");
    let result = f();
    assert_eq!(result, 36.0);
}

// A non-exit block of a [YieldingRegions] op (here, a [ExecuteRegionOp]) must end
// with a terminator implementing `BranchOpInterface`.
#[test]
fn test_execute_region_op_non_branch_terminator_rejected() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
      builtin.module @test_module {
        ^entry():
          llvm.func @test_execute_region: llvm.func <builtin.fp32 () variadic = false> [] {
            ^entry():
              inc = builtin.constant <builtin.single 3.5> : builtin.fp32;
              next = cf.execute_region -> (builtin.fp32) {
                  ^entry():
                      llvm.return inc
                  ^exit(v : builtin.fp32):
                      cf.yield v
              };
              llvm.return next
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

    let err = verify_op(&module_op, ctx).unwrap_err();
    let err = err.err.downcast_ref::<YieldingRegionsVerifyErr>().unwrap();
    assert!(matches!(
        err,
        YieldingRegionsVerifyErr::NonExitBlockBadTerminator
    ));
}

// `cf.cond_br` round-trips through parsing/printing and lowers to `llvm.cond_br`,
// branching to the correct target block with the correct forwarded operand.
#[test]
fn test_cond_br_op_to_llvm_conversion() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @test_cond_br: llvm.func <builtin.fp32 () variadic = false> [] {
                ^entry():
                    zero = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                    one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                    cond = llvm.icmp zero <ULT> one : builtin.integer i1;
                    true_val = builtin.constant <builtin.single 1.0> : builtin.fp32;
                    false_val = builtin.constant <builtin.single 2.0> : builtin.fp32;
                    cf.cond_br if cond ^true_blk(true_val) else ^false_blk(false_val)
                ^true_blk(x : builtin.fp32):
                    llvm.return x
                ^false_blk(y : builtin.fp32):
                    llvm.return y
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

    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_cond_br: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                zero_v7 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64 !2;
                one_v8 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64 !3;
                cond_v2 = llvm.icmp zero_v7 <ULT> one_v8 : builtin.integer i1 !4;
                true_val_v9 = llvm.constant <builtin.single 1> : builtin.fp32  !5;
                false_val_v10 = llvm.constant <builtin.single 2> : builtin.fp32  !6;
                llvm.cond_br if cond_v2 ^true_blk_block5v1(true_val_v9) else ^false_blk_block3v3(false_val_v10) !7

              ^true_blk_block5v1(x_v5: builtin.fp32 ) !8:
                llvm.return x_v5 !9

              ^false_blk_block3v3(y_v6: builtin.fp32 ) !10:
                llvm.return y_v6 !11
            } !12
        }"#]]
    .assert_eq(&module_op.disp(ctx).to_string());
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    let f = unsafe { jit.lookup_symbol::<fn() -> f32>("test_cond_br") }
        .expect("Failed to lookup symbol");
    let result = f();
    // 0 < 1 is true, so the `true_blk` branch (returning `true_val` == 1.0) is taken.
    assert_eq!(result, 1.0);
}

// `cf.if`, with an `else` region and a result
fn if_op_with_else_input_ir(cmp_true: bool) -> String {
    let predicate = if cmp_true { "ULT" } else { "UGT" };
    format!(
        r#"
        builtin.module @test_module {{
          ^entry():
            llvm.func @test_if: llvm.func <builtin.fp32 () variadic = false> [] {{
                ^entry():
                    zero = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                    one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                    cond = llvm.icmp zero <{predicate}> one : builtin.integer i1;
                    true_val = builtin.constant <builtin.single 1.0> : builtin.fp32;
                    false_val = builtin.constant <builtin.single 2.0> : builtin.fp32;
                    result = cf.if cond -> (builtin.fp32) {{
                        ^entry():
                            cf.yield true_val
                    }} else {{
                        ^entry():
                            cf.yield false_val
                    }};
                    llvm.return result
            }}
        }}
        "#
    )
}

#[test]
fn test_if_op_with_else_to_llvm_conversion() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = if_op_with_else_input_ir(true);

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
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_if: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                zero_v0 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64 !2;
                one_v1 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !3;
                cond_v2 = llvm.icmp zero_v0 <ULT> one_v1 : builtin.integer i1 !4;
                true_val_v3 = builtin.constant <builtin.single 1> : builtin.fp32  !5;
                false_val_v4 = builtin.constant <builtin.single 2> : builtin.fp32  !6;
                result_v5 = cf.if cond_v2 -> (builtin.fp32 ) 
                {
                  ^entry_block3v1() !7:
                    cf.yield true_val_v3 !8
                } else 
                {
                  ^entry_block4v1() !9:
                    cf.yield false_val_v4 !10
                }
         !11;
                llvm.return result_v5 !12
            } !13
        }"#]]
    .assert_eq(&module_op.disp(ctx).to_string());
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);

    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_if: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                zero_v6 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64 !2;
                one_v7 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64 !3;
                cond_v2 = llvm.icmp zero_v6 <ULT> one_v7 : builtin.integer i1 !4;
                true_val_v8 = llvm.constant <builtin.single 1> : builtin.fp32  !5;
                false_val_v9 = llvm.constant <builtin.single 2> : builtin.fp32  !6;
                llvm.cond_br if cond_v2 ^entry_block3v1() else ^entry_block4v1()

              ^entry_block3v1() !7:
                llvm.br ^block5v1(true_val_v8)

              ^entry_block4v1() !8:
                llvm.br ^block5v1(false_val_v9)

              ^block5v1(result_v10: builtin.fp32 ) !9:
                llvm.return result_v10 !10
            } !11
        }"#]]
    .assert_eq(&module_op.disp(ctx).to_string());
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    let f =
        unsafe { jit.lookup_symbol::<fn() -> f32>("test_if") }.expect("Failed to lookup symbol");
    let result = f();
    // 0 < 1 is true, so the `then` region (yielding `true_val` == 1.0) is taken.
    assert_eq!(result, 1.0);
}

#[test]
fn test_if_op_with_else_false_branch_to_llvm_conversion() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = if_op_with_else_input_ir(false);

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

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    let f =
        unsafe { jit.lookup_symbol::<fn() -> f32>("test_if") }.expect("Failed to lookup symbol");
    let result = f();
    // 0 > 1 is false, so the `else` region (yielding `false_val` == 2.0) is taken.
    assert_eq!(result, 2.0);
}

// `cf.if` without an `else` region and without results
#[test]
fn test_if_op_without_else_to_llvm_conversion() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @test_if_no_else: llvm.func <builtin.fp32 () variadic = false> [] {
            ^entry():
                zero = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                cond = llvm.icmp zero <ULT> one : builtin.integer i1;
                unused = builtin.constant <builtin.single 9.0> : builtin.fp32;
                result = builtin.constant <builtin.single 42.0> : builtin.fp32;
                cf.if cond -> () {
                    ^entry():
                        cf.yield
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

    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_if_no_else: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                zero_v5 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64 !2;
                one_v6 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64 !3;
                cond_v2 = llvm.icmp zero_v5 <ULT> one_v6 : builtin.integer i1 !4;
                unused_v7 = llvm.constant <builtin.single 9> : builtin.fp32  !5;
                result_v8 = llvm.constant <builtin.single 42> : builtin.fp32  !6;
                llvm.cond_br if cond_v2 ^entry_block3v1() else ^block4v1()

              ^entry_block3v1() !7:
                llvm.br ^block4v1()

              ^block4v1():
                llvm.return result_v8 !8
            } !9
        }"#]]
    .assert_eq(&module_op.disp(ctx).to_string());
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    let f = unsafe { jit.lookup_symbol::<fn() -> f32>("test_if_no_else") }
        .expect("Failed to lookup symbol");
    let result = f();
    assert_eq!(result, 42.0);
}

// `cf.if` with results must have an `else` region
#[test]
fn test_if_op_missing_else_with_results_rejected() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @test_if: llvm.func <builtin.fp32 () variadic = false> [] {
            ^entry():
                zero = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                cond = llvm.icmp zero <ULT> one : builtin.integer i1;
                true_val = builtin.constant <builtin.single 1.0> : builtin.fp32;
                result = cf.if cond -> (builtin.fp32) {
                    ^entry():
                        cf.yield true_val
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

    let err = verify_op(&module_op, ctx).unwrap_err();
    let err = err.err.downcast_ref::<IfOpVerifyErr>().unwrap();
    assert!(matches!(err, IfOpVerifyErr::MissingElseRegion));
}

// `cf.if`'s regions must be single-block
#[test]
fn test_if_op_execute_region_in_then_branch_to_llvm_conversion() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @test_if_execute_region: llvm.func <builtin.fp32 () variadic = false> [] {
            ^entry():
                zero = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                cond = llvm.icmp zero <ULT> one : builtin.integer i1;
                inc = builtin.constant <builtin.single 3.5> : builtin.fp32;
                false_val = builtin.constant <builtin.single 2.0> : builtin.fp32;
                result = cf.if cond -> (builtin.fp32) {
                    ^entry():
                        v = cf.execute_region -> (builtin.fp32) {
                            ^entry():
                                cf.br ^next(inc)
                            ^next(w : builtin.fp32):
                                cf.yield w
                        };
                        cf.yield v
                } else {
                    ^entry():
                        cf.yield false_val
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
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_if_execute_region: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                zero_v8 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64 !2;
                one_v9 = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64 !3;
                cond_v2 = llvm.icmp zero_v8 <ULT> one_v9 : builtin.integer i1 !4;
                inc_v10 = llvm.constant <builtin.single 3.5> : builtin.fp32  !5;
                false_val_v11 = llvm.constant <builtin.single 2> : builtin.fp32  !6;
                llvm.cond_br if cond_v2 ^entry_block3v1() else ^entry_block5v3()

              ^entry_block3v1() !7:
                llvm.br ^entry_block4v1()

              ^entry_block4v1() !8:
                llvm.br ^next_block6v1(inc_v10) !9

              ^next_block6v1(w_v7: builtin.fp32 ) !10:
                llvm.br ^block8v1(w_v7)

              ^block8v1(v_v13: builtin.fp32 ) !11:
                llvm.br ^block7v1(v_v13)

              ^entry_block5v3() !12:
                llvm.br ^block7v1(false_val_v11)

              ^block7v1(result_v12: builtin.fp32 ) !13:
                llvm.return result_v12 !14
            } !15
        }"#]]
    .assert_eq(&module_op.disp(ctx).to_string());
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    let f = unsafe { jit.lookup_symbol::<fn() -> f32>("test_if_execute_region") }
        .expect("Failed to lookup symbol");
    let result = f();
    // 0 < 1 is true, so the `then` region is taken; its `cf.execute_region`
    // branches through a second block before yielding `inc` == 3.5.
    assert_eq!(result, 3.5);
}

// `cf.if`'s regions must each have a single block.
#[test]
fn test_if_op_multi_block_region_rejected() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @test_if_multi_block: llvm.func <builtin.fp32 () variadic = false> [] {
                ^entry():
                    zero = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                    one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                    cond = llvm.icmp zero <ULT> one : builtin.integer i1;
                    inc = builtin.constant <builtin.single 3.5> : builtin.fp32;
                    false_val = builtin.constant <builtin.single 2.0> : builtin.fp32;
                    result = cf.if cond -> (builtin.fp32) {
                        ^entry():
                            cf.br ^next(inc)
                        ^next(v : builtin.fp32):
                            cf.yield v
                    } else {
                        ^entry():
                            cf.yield false_val
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

    let err = verify_op(&module_op, ctx).unwrap_err();
    err.err
        .downcast_ref::<SingleBlockRegionVerifyErr>()
        .unwrap();
}
