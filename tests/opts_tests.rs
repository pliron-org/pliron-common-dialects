// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Tests for the optimizations of pliron (SCCP, DCE and CFG simplification)
//! on the ops of the `cf` and `index` dialects.

use pliron::{
    builtin::ops::ModuleOp,
    combine::Parser,
    context::Context,
    init_env_logger_for_tests, input_error_noloc,
    irfmt::parsers::spaced,
    location,
    op::{Op, verify_op},
    operation::Operation,
    opts::{constants::sccp::sccp, dce::dce, simplify_cfg::simplify_cfg},
    parsable::{self, state_stream_from_iterator},
    printable::Printable,
    result::ExpectOk,
};

use expect_test::expect;
// Link the crates that register the dialects used in the test inputs.
use pliron_common_dialects as _;
use pliron_llvm as _;

/// Parse `input_ir` into a module, and verify it.
fn parse_module(ctx: &mut Context, input_ir: &str) -> ModuleOp {
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
    module_op
}

/// Check that SCCP and DCE fold the `index` ops on constant operands.
#[test]
fn test_index_ops_sccp_and_dce() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let module_op = parse_module(
        ctx,
        r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @index_fold: llvm.func <builtin.integer i64 () variadic = false> [] {
              ^entry():
                c3 = index.constant <index.constant 3> : index.index;
                c4 = index.constant <index.constant 4> : index.index;
                sum = index.add c3, c4 : index.index;
                prod = index.mul sum, c3 : index.index;
                max = index.constant <index.constant 18446744073709551615> : index.index;
                wrapped = index.add max, prod : index.index;
                int = index.to_integer wrapped to builtin.integer i64;
                idx = index.from_integer int : index.index;
                res = index.to_integer idx to builtin.integer i64;
                llvm.return res
            };
            llvm.func @index_no_fold: llvm.func <builtin.integer i64 (builtin.integer i64) variadic = false> [] {
              ^entry(n : builtin.integer i64):
                c3 = index.constant <index.constant 3> : index.index;
                n_idx = index.from_integer n : index.index;
                sum = index.add n_idx, c3 : index.index;
                res = index.to_integer sum to builtin.integer i64;
                llvm.return res
            }
        }
        "#,
    );

    sccp(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);
    dce(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    // `@index_fold` must fold to a constant. `@index_no_fold` must remain.
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @index_fold: llvm.func <builtin.integer i64() variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                res_v19 = builtin.constant <builtin.integer <20: i64>> : builtin.integer i64 !2;
                llvm.return res_v19 !3
            } !4;
            llvm.func @index_no_fold: llvm.func <builtin.integer i64(builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block3v1(n_v9: builtin.integer i64) !5:
                c3_v10 = index.constant <index.constant 3> : index.index  !6;
                n_idx_v11 = index.from_integer n_v9 : index.index  !7;
                sum_v12 = index.add n_idx_v11, c3_v10 : index.index  !8;
                res_v13 = index.to_integer sum_v12 to builtin.integer i64 !9;
                llvm.return res_v13 !10
            } !11
        }"#]].assert_eq(&module_op.disp(ctx).to_string());
}

/// Check that SCCP and DCE fold region ops and `cf.cond_br` on constant
/// and `poison` conditions.
#[test]
fn test_region_ops_sccp_and_dce() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let module_op = parse_module(
        ctx,
        r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @region_fold: llvm.func <builtin.fp32 () variadic = false> [] {
              ^entry():
                zero = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                cond = llvm.icmp zero <ULT> one : builtin.integer i1;
                true_val = builtin.constant <builtin.single 1.0> : builtin.fp32;
                false_val = builtin.constant <builtin.single 2.0> : builtin.fp32;
                if_res = cf.if cond -> (builtin.fp32) {
                    ^entry():
                        cf.yield true_val
                } else {
                    ^entry():
                        cf.yield false_val
                };
                br_res = cf.execute_region -> (builtin.fp32) {
                    ^entry():
                        cf.cond_br if cond ^on_true(if_res) else ^on_false(false_val)
                    ^on_true(x : builtin.fp32):
                        cf.br ^exit(x)
                    ^on_false(y : builtin.fp32):
                        cf.br ^exit(y)
                    ^exit(v : builtin.fp32):
                        cf.yield v
                };
                c0 = index.constant <index.constant 0> : index.index;
                c1 = index.constant <index.constant 1> : index.index;
                c5 = index.constant <index.constant 5> : index.index;
                for_res = cf.for c5 to c0 step c1 (br_res) {
                    ^entry(iv : index.index, acc : builtin.fp32):
                        next = llvm.fadd <FAST> acc, false_val : builtin.fp32;
                        cf.yield next
                };
                llvm.return for_res
            };
            llvm.func @poison_cond: llvm.func <builtin.integer i64 (builtin.integer i64) variadic = false> [] {
              ^entry(n : builtin.integer i64):
                pc = llvm.poison : builtin.integer i1;
                c3 = index.constant <index.constant 3> : index.index;
                c4 = index.constant <index.constant 4> : index.index;
                r = cf.execute_region -> (builtin.integer i64) {
                    ^entry():
                        cf.cond_br if pc ^p_true() else ^p_false()
                    ^p_true():
                        sum = index.add c3, c4 : index.index;
                        sum_int = index.to_integer sum to builtin.integer i64;
                        cf.br ^p_exit(sum_int)
                    ^p_false():
                        cf.br ^p_exit(n)
                    ^p_exit(v : builtin.integer i64):
                        cf.yield v
                };
                llvm.return r
            }
        }
        "#,
    );

    sccp(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);
    dce(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    // `@region_fold` must fold to a constant. `@poison_cond` must remain.
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @region_fold: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                true_val_v3 = builtin.constant <builtin.single 1> : builtin.fp32  !2;
                false_val_v4 = builtin.constant <builtin.single 2> : builtin.fp32  !3;
                if_res_v26 = builtin.constant <builtin.single 1> : builtin.fp32  !4;
                for_res_v30 = builtin.constant <builtin.single 1> : builtin.fp32  !5;
                llvm.return for_res_v30 !6
            } !7;
            llvm.func @poison_cond: llvm.func <builtin.integer i64(builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block10v1(n_v17: builtin.integer i64) !8:
                pc_v18 = llvm.poison : builtin.integer i1 !9;
                c3_v19 = index.constant <index.constant 3> : index.index  !10;
                c4_v20 = index.constant <index.constant 4> : index.index  !11;
                r_v21 = cf.execute_region -> (builtin.integer i64) 
                {
                  ^entry_block11v1() !12:
                    cf.cond_br if pc_v18 ^p_true_block14v1() else ^p_false_block15v1() !13

                  ^p_true_block14v1() !14:
                    sum_v22 = index.add c3_v19, c4_v20 : index.index  !15;
                    sum_int_v23 = index.to_integer sum_v22 to builtin.integer i64 !16;
                    cf.br ^p_exit_block13v3(sum_int_v23) !17

                  ^p_false_block15v1() !18:
                    cf.br ^p_exit_block13v3(n_v17) !19

                  ^p_exit_block13v3(v_v24: builtin.integer i64) !20:
                    cf.yield v_v24 !21
                }
         !22;
                llvm.return r_v21 !23
            } !24
        }"#]]
    .assert_eq(&module_op.disp(ctx).to_string());
}

/// Check that CFG simplification folds `cf.if`, `cf.for`, `cf.nd_for` and
/// `cf.cond_br` ops with constant operands.
#[test]
fn test_region_ops_simplify_cfg() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let module_op = parse_module(
        ctx,
        r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @simplify: llvm.func <builtin.fp32 (builtin.integer i64) variadic = false> [] {
              ^entry(n : builtin.integer i64):
                t = builtin.constant <builtin.integer <1: i1>> : builtin.integer i1;
                f = builtin.constant <builtin.integer <0: i1>> : builtin.integer i1;
                one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                p = llvm.alloca [builtin.fp32 x one] : llvm.ptr(0);
                f1 = builtin.constant <builtin.single 1.0> : builtin.fp32;
                f2 = builtin.constant <builtin.single 2.0> : builtin.fp32;
                f3 = builtin.constant <builtin.single 3.0> : builtin.fp32;
                c0 = index.constant <index.constant 0> : index.index;
                c1 = index.constant <index.constant 1> : index.index;
                c5 = index.constant <index.constant 5> : index.index;
                n_idx = index.from_integer n : index.index;
                a = cf.if t -> (builtin.fp32) {
                    ^entry():
                        llvm.store *p <- f1;
                        cf.yield f1
                } else {
                    ^entry():
                        cf.yield f2
                };
                b = cf.if f -> (builtin.fp32) {
                    ^entry():
                        cf.yield f1
                } else {
                    ^entry():
                        llvm.store *p <- f2;
                        cf.yield f2
                };
                cf.if f -> () {
                    ^entry():
                        llvm.store *p <- f3;
                        cf.yield
                };
                e = cf.execute_region -> (builtin.fp32) {
                    ^entry():
                        cf.cond_br if f ^on_true(f1) else ^on_false(f3)
                    ^on_true(x : builtin.fp32):
                        cf.br ^exit(x)
                    ^on_false(y : builtin.fp32):
                        cf.br ^exit(y)
                    ^exit(v : builtin.fp32):
                        cf.yield v
                };
                llvm.store *p <- e;
                r = cf.for c5 to c0 step c1 (a) {
                    ^entry(iv1 : index.index, acc1 : builtin.fp32):
                        llvm.store *p <- acc1;
                        cf.yield acc1
                };
                cf.nd_for [c0, c0] to [c5, c0] step [c1, c1] {
                    ^entry(i0 : index.index, j0 : index.index):
                        llvm.store *p <- f3;
                        cf.yield
                };
                s = cf.for c0 to n_idx step c1 (b) {
                    ^entry(iv2 : index.index, acc2 : builtin.fp32):
                        next = llvm.fadd <FAST> acc2, r : builtin.fp32;
                        cf.yield next
                };
                llvm.return s
            }
        }
        "#,
    );

    simplify_cfg(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    // Only `e`, with one block, and `s` must remain.
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @simplify: llvm.func <builtin.fp32 (builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block2v1(n_v0: builtin.integer i64) !1:
                t_v1 = builtin.constant <builtin.integer <1: i1>> : builtin.integer i1 !2;
                f_v2 = builtin.constant <builtin.integer <0: i1>> : builtin.integer i1 !3;
                one_v3 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !4;
                p_v4 = llvm.alloca [builtin.fp32  x one_v3]  : llvm.ptr (0) !5;
                f1_v5 = builtin.constant <builtin.single 1> : builtin.fp32  !6;
                f2_v6 = builtin.constant <builtin.single 2> : builtin.fp32  !7;
                f3_v7 = builtin.constant <builtin.single 3> : builtin.fp32  !8;
                c0_v8 = index.constant <index.constant 0> : index.index  !9;
                c1_v9 = index.constant <index.constant 1> : index.index  !10;
                c5_v10 = index.constant <index.constant 5> : index.index  !11;
                n_idx_v11 = index.from_integer n_v0 : index.index  !12;
                llvm.store *p_v4 <- f1_v5  !13;
                llvm.store *p_v4 <- f2_v6  !14;
                e_v14 = cf.execute_region -> (builtin.fp32 ) 
                {
                  ^entry_block8v1() !15:
                    cf.yield f3_v7 !16
                }
         !17;
                llvm.store *p_v4 <- e_v14  !18;
                s_v23 = cf.for c0_v8 to n_idx_v11 step c1_v9 (f2_v6) 
                {
                  ^entry_block14v1(iv2_v24: index.index , acc2_v25: builtin.fp32 ) !19:
                    next_v26 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> acc2_v25, f1_v5 : builtin.fp32  !20;
                    cf.yield next_v26 !21
                }
         !22;
                llvm.return s_v23 !23
            } !24
        }"#]].assert_eq(&module_op.disp(ctx).to_string());
}

/// Check that DCE erases region ops that have no uses and no side effects.
#[test]
fn test_region_ops_dce() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let module_op = parse_module(
        ctx,
        r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @dead_ops: llvm.func <builtin.fp32 (builtin.integer i64, builtin.integer i1) variadic = false> [] {
              ^entry(n : builtin.integer i64, cond : builtin.integer i1):
                one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                p = llvm.alloca [builtin.fp32 x one] : llvm.ptr(0);
                f1 = builtin.constant <builtin.single 1.0> : builtin.fp32;
                c0 = index.constant <index.constant 0> : index.index;
                c1 = index.constant <index.constant 1> : index.index;
                c5 = index.constant <index.constant 5> : index.index;
                n_idx = index.from_integer n : index.index;
                stops = cf.for c0 to c5 step c1 (f1) {
                    ^entry(iv1 : index.index, acc1 : builtin.fp32):
                        cf.yield acc1
                };
                unknown_ub = cf.for c0 to n_idx step c1 (f1) {
                    ^entry(iv2 : index.index, acc2 : builtin.fp32):
                        cf.yield acc2
                };
                cf.nd_for [c0, c0] to [c0, n_idx] step [c1, c1] {
                    ^entry(i0 : index.index, j0 : index.index):
                        cf.yield
                };
                cf.nd_for [c0] to [c5] step [c1] {
                    ^entry(i : index.index):
                        cf.yield
                };
                pure_if = cf.if cond -> (builtin.fp32) {
                    ^entry():
                        cf.yield f1
                } else {
                    ^entry():
                        cf.yield f1
                };
                cf.if cond -> () {
                    ^entry():
                        llvm.store *p <- f1;
                        cf.yield
                };
                pure_region = cf.execute_region -> (builtin.fp32) {
                    ^entry():
                        cf.br ^exit(f1)
                    ^exit(v : builtin.fp32):
                        cf.yield v
                };
                llvm.return f1
            }
        }
        "#,
    );

    dce(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    // Only the `cf.if` with the store must remain.
    expect![[r#"
    builtin.module @test_module 
    {
      ^entry_block1v1() !0:
        llvm.func @dead_ops: llvm.func <builtin.fp32 (builtin.integer i64, builtin.integer i1) variadic = false>
          [] 
        {
          ^entry_block2v1(n_v0: builtin.integer i64, cond_v1: builtin.integer i1) !1:
            one_v2 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !2;
            p_v3 = llvm.alloca [builtin.fp32  x one_v2]  : llvm.ptr (0) !3;
            f1_v4 = builtin.constant <builtin.single 1> : builtin.fp32  !4;
            cf.if cond_v1 -> () 
            {
              ^entry_block9v1() !5:
                llvm.store *p_v3 <- f1_v4  !6;
                cf.yield  !7
            }
     !8;
            llvm.return f1_v4 !9
        } !10
    }"#]].assert_eq(&module_op.disp(ctx).to_string());
}
