// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Tests for the arith dialect: parsing, printing, verification, DCE,
//! SCCP, lowering to the LLVM dialect and execution.

use expect_test::expect;
use pliron::{
    builtin::{
        op_interfaces::{
            AllOperandsOfTypeVerifyErr, SameOperandsAndResultTypeVerifyErr,
            SameOperandsTypeVerifyErr,
        },
        ops::ModuleOp,
    },
    combine::Parser,
    context::Context,
    init_env_logger_for_tests, input_error_noloc,
    irbuild::{IRStatus, dialect_conversion::apply_dialect_conversion},
    irfmt::parsers::spaced,
    location,
    op::{Op, verify_op},
    operation::Operation,
    opts::{constants::sccp::sccp, dce::dce},
    parsable::{self, state_stream_from_iterator},
    printable::Printable,
    result::{Error, ExpectOk},
};
use pliron_common_dialects::arith::ops::IntBinOpVerifyErr;
use pliron_llvm::{
    builtin_to_llvm::BuiltinToLLVMConversion,
    llvm_sys::{core::LLVMContext, lljit::SimpleJIT},
};

const I0: &str = "builtin.integer i0";
const I32: &str = "builtin.integer i32";
const I64: &str = "builtin.integer i64";
const SI32: &str = "builtin.integer si32";
const F32: &str = "builtin.fp32";

/// Parse `input` as a module and return it.
fn parse_module(ctx: &mut Context, input: &str) -> ModuleOp {
    let state_stream = state_stream_from_iterator(
        input.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );

    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));

    let parsed_op = parsed.expect_ok(ctx);
    Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap()
}

/// Build a module with a function `@test(a: lhs, b: rhs) -> res` whose body
/// applies the binary op `op` to `a` and `b` and returns the result.
fn binop_ir(op: &str, lhs: &str, rhs: &str, res: &str) -> String {
    format!(
        r#"
        builtin.module @test_module {{
          ^entry():
            llvm.func @test: llvm.func <{res} ({lhs}, {rhs}) variadic = false> [] {{
              ^entry(a: {lhs}, b: {rhs}):
                s = {op} a, b : {res};
                llvm.return s
            }}
        }}"#
    )
}

/// Parse `input` and verify it, failing the test if it is invalid.
fn assert_verifies(ctx: &mut Context, input: &str) -> ModuleOp {
    let module = parse_module(ctx, input);
    verify_op(&module, ctx).expect_ok(ctx);
    module
}

/// Parse `input` and verify it, failing the test if it is valid.
/// Returns the verification error for further checks.
fn assert_rejected(ctx: &mut Context, input: &str) -> Error {
    let module = parse_module(ctx, input);
    verify_op(&module, ctx).expect_err("IR was expected to be rejected by the verifier")
}

/// Lower `module` to the LLVM dialect and verify the result.
fn lower_to_llvm(ctx: &mut Context, module: ModuleOp) {
    let status =
        apply_dialect_conversion(ctx, &mut BuiltinToLLVMConversion, module.get_operation())
            .expect_ok(ctx);
    assert!(matches!(status, IRStatus::Changed));
    verify_op(&module, ctx).expect_ok(ctx);
}

/// A valid `arith.addi` on `i32` parses, verifies and prints back.
#[test]
fn addi_i32_roundtrip() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let module = assert_verifies(ctx, &binop_ir("arith.addi", I32, I32, I32));
    let printed = format!("{}", module.get_operation().disp(ctx));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test: llvm.func <builtin.integer i32(builtin.integer i32, builtin.integer i32) variadic = false>
              [] 
            {
              ^entry_block2v1(a_v0: builtin.integer i32, b_v1: builtin.integer i32) !1:
                s_v2 = arith.addi a_v0, b_v1 : builtin.integer i32 !2;
                llvm.return s_v2 !3
            } !4
        } !5

        outlined_attributes:
        !0 = @[<in-memory>: line: 3, column: 11], []
        !1 = @[<in-memory>: line: 5, column: 15], [builtin_given_names = builtin.given_names [a, b]]
        !2 = @[<in-memory>: line: 6, column: 17], [builtin_given_names = builtin.given_names [s]]
        !3 = @[<in-memory>: line: 7, column: 17], []
        !4 = @[<in-memory>: line: 4, column: 13], []
        !5 = @[<in-memory>: line: 2, column: 9], []
    "#]].assert_eq(&printed);
}

/// Float operands are rejected.
#[test]
fn addi_rejects_float() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let err = assert_rejected(ctx, &binop_ir("arith.addi", F32, F32, F32));
    err.err
        .downcast_ref::<AllOperandsOfTypeVerifyErr>()
        .unwrap();
}

/// Operands of different widths are rejected.
#[test]
fn addi_rejects_mixed_width() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let err = assert_rejected(ctx, &binop_ir("arith.addi", I32, I64, I32));
    err.err.downcast_ref::<SameOperandsTypeVerifyErr>().unwrap();
}

/// Signed integer operands are rejected.
#[test]
fn addi_rejects_signed() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let err = assert_rejected(ctx, &binop_ir("arith.addi", SI32, SI32, SI32));
    assert!(matches!(
        err.err.downcast_ref::<IntBinOpVerifyErr>(),
        Some(IntBinOpVerifyErr::NotSignless)
    ));
}

/// Zero-width integer operands are rejected.
#[test]
fn addi_rejects_zero_width() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let err = assert_rejected(ctx, &binop_ir("arith.addi", I0, I0, I0));
    assert!(matches!(
        err.err.downcast_ref::<IntBinOpVerifyErr>(),
        Some(IntBinOpVerifyErr::ZeroWidth)
    ));
}

/// A result type different from the operand type is rejected.
#[test]
fn addi_rejects_result_mismatch() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let err = assert_rejected(ctx, &binop_ir("arith.addi", I32, I32, I64));
    err.err
        .downcast_ref::<SameOperandsAndResultTypeVerifyErr>()
        .unwrap();
}

/// An `arith.addi` whose result is unused is removed by DCE.
#[test]
fn addi_unused_is_removed_by_dce() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let module = assert_verifies(
        ctx,
        &binop_ir("arith.addi", I32, I32, I32).replace("llvm.return s", "llvm.return a"),
    );
    let status = dce(module.get_operation(), ctx).expect_ok(ctx);
    let printed = format!("{}", module.get_operation().disp(ctx));
    assert!(matches!(status, IRStatus::Changed));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test: llvm.func <builtin.integer i32(builtin.integer i32, builtin.integer i32) variadic = false>
              [] 
            {
              ^entry_block2v1(a_v0: builtin.integer i32, b_v1: builtin.integer i32) !1:
                llvm.return a_v0 !2
            } !3
        } !4

        outlined_attributes:
        !0 = @[<in-memory>: line: 3, column: 11], []
        !1 = @[<in-memory>: line: 5, column: 15], [builtin_given_names = builtin.given_names [a, b]]
        !2 = @[<in-memory>: line: 7, column: 17], []
        !3 = @[<in-memory>: line: 4, column: 13], []
        !4 = @[<in-memory>: line: 2, column: 9], []
    "#]].assert_eq(&printed);
}

/// An `arith.addi` whose result is used is kept by DCE.
#[test]
fn addi_used_is_kept_by_dce() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let module = assert_verifies(ctx, &binop_ir("arith.addi", I32, I32, I32));
    let status = dce(module.get_operation(), ctx).expect_ok(ctx);
    let printed = format!("{}", module.get_operation().disp(ctx));
    assert!(matches!(status, IRStatus::Unchanged));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test: llvm.func <builtin.integer i32(builtin.integer i32, builtin.integer i32) variadic = false>
              [] 
            {
              ^entry_block2v1(a_v0: builtin.integer i32, b_v1: builtin.integer i32) !1:
                s_v2 = arith.addi a_v0, b_v1 : builtin.integer i32 !2;
                llvm.return s_v2 !3
            } !4
        } !5

        outlined_attributes:
        !0 = @[<in-memory>: line: 3, column: 11], []
        !1 = @[<in-memory>: line: 5, column: 15], [builtin_given_names = builtin.given_names [a, b]]
        !2 = @[<in-memory>: line: 6, column: 17], [builtin_given_names = builtin.given_names [s]]
        !3 = @[<in-memory>: line: 7, column: 17], []
        !4 = @[<in-memory>: line: 4, column: 13], []
        !5 = @[<in-memory>: line: 2, column: 9], []
    "#]].assert_eq(&printed);
}

/// `arith.addi` lowers to `llvm.add` with no overflow flags.
#[test]
fn addi_lowers_to_llvm_add() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let module = assert_verifies(ctx, &binop_ir("arith.addi", I32, I32, I32));
    lower_to_llvm(ctx, module);
    let printed = format!("{}", module.get_operation().disp(ctx));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test: llvm.func <builtin.integer i32(builtin.integer i32, builtin.integer i32) variadic = false>
              [] 
            {
              ^entry_block2v1(a_v0: builtin.integer i32, b_v1: builtin.integer i32) !1:
                s_v3 = llvm.add a_v0, b_v1 <{nsw=false,nuw=false}>: builtin.integer i32 !2;
                llvm.return s_v3 !3
            } !4
        } !5

        outlined_attributes:
        !0 = @[<in-memory>: line: 3, column: 11], []
        !1 = @[<in-memory>: line: 5, column: 15], [builtin_given_names = builtin.given_names [a, b]]
        !2 = @[<in-memory>: line: 6, column: 17], [builtin_given_names = builtin.given_names [s]]
        !3 = @[<in-memory>: line: 7, column: 17], []
        !4 = @[<in-memory>: line: 4, column: 13], []
        !5 = @[<in-memory>: line: 2, column: 9], []
    "#]].assert_eq(&printed);
}

/// Lowered `arith.addi` translates to LLVM IR, runs, and wraps on overflow.
#[test]
fn addi_wraps_when_run() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();
    let module = assert_verifies(ctx, &binop_ir("arith.addi", I32, I32, I32));
    lower_to_llvm(ctx, module);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir =
        pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module, Default::default())
            .expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();
    expect![[r#"
        ; ModuleID = 'test_module'
        source_filename = "test_module"

        define i32 @test(i32 %0, i32 %1) {
        entry_block2v1:
          %s_v3 = add i32 %0, %1
          ret i32 %s_v3
        }
    "#]]
    .assert_eq(&llvm_ir.to_string());

    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    // SAFETY: the signature matches `define i32 @test(i32, i32)` above.
    let f = unsafe { jit.lookup_symbol::<fn(i32, i32) -> i32>("test") }
        .expect("Failed to lookup symbol");
    assert_eq!(f(2, 3), 5);
    // Signed overflow wraps around modulo 2^32.
    assert_eq!(f(i32::MAX, 1), i32::MIN);
    assert_eq!(f(i32::MAX, i32::MAX), i32::MAX.wrapping_add(i32::MAX));
}

/// Check that SCCP and DCE fold the `arith` ops on constant operands.
#[test]
fn addi_is_folded_by_sccp() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let module_op = parse_module(
        ctx,
        r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @arith_fold: llvm.func <builtin.integer i32 () variadic = false> [] {
              ^entry():
                c1 = builtin.constant <builtin.integer <1: i32>> : builtin.integer i32;
                c3 = builtin.constant <builtin.integer <3: i32>> : builtin.integer i32;
                c4 = builtin.constant <builtin.integer <4: i32>> : builtin.integer i32;
                max = builtin.constant <builtin.integer <2147483647: i32>> : builtin.integer i32;
                sum = arith.addi c3, c4 : builtin.integer i32;
                wrapped = arith.addi max, c1 : builtin.integer i32;
                res = arith.addi sum, wrapped : builtin.integer i32;
                llvm.return res
            };
            llvm.func @arith_no_fold: llvm.func <builtin.integer i32 (builtin.integer i32) variadic = false> [] {
              ^entry(n : builtin.integer i32):
                c3 = builtin.constant <builtin.integer <3: i32>> : builtin.integer i32;
                sum = arith.addi n, c3 : builtin.integer i32;
                llvm.return sum
            }
        }
        "#,
    );

    sccp(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);
    dce(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    // `@arith_fold` must fold to a constant.
    // `@arith_no_fold` must not fold.
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @arith_fold: llvm.func <builtin.integer i32() variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                res_v12 = builtin.constant <builtin.integer <-2147483641: i32>> : builtin.integer i32 !2;
                llvm.return res_v12 !3
            } !4;
            llvm.func @arith_no_fold: llvm.func <builtin.integer i32(builtin.integer i32) variadic = false>
              [] 
            {
              ^entry_block3v1(n_v7: builtin.integer i32) !5:
                c3_v8 = builtin.constant <builtin.integer <3: i32>> : builtin.integer i32 !6;
                sum_v9 = arith.addi n_v7, c3_v8 : builtin.integer i32 !7;
                llvm.return sum_v9 !8
            } !9
        }"#]].assert_eq(&module_op.disp(ctx).to_string());
}
