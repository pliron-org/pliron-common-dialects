// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Optimizations tests for `cf` and `index` dialects ops.

use pliron::{
    builtin::ops::ModuleOp,
    combine::Parser,
    context::Context,
    init_env_logger_for_tests, input_error_noloc,
    irfmt::parsers::spaced,
    location,
    op::{Op, verify_op},
    operation::Operation,
    opts::{
        canonicalize::canonicalize, constants::sccp::sccp, dce::dce, mem2reg::mem2reg,
        simplify_cfg::simplify_cfg,
    },
    parsable::{self, state_stream_from_iterator},
    pass::AnalysisManager,
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
                quot = index.divu prod, c4 : index.index;
                max = index.constant <index.constant 18446744073709551615> : index.index;
                wrapped = index.add max, quot : index.index;
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
            };
            llvm.func @index_div_by_zero: llvm.func <builtin.integer i64 () variadic = false> [] {
              ^entry():
                c3 = index.constant <index.constant 3> : index.index;
                c0 = index.constant <index.constant 0> : index.index;
                quot = index.divu c3, c0 : index.index;
                res = index.to_integer quot to builtin.integer i64;
                llvm.return res
            }
        }
        "#,
    );

    sccp(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);
    dce(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    // `@index_fold` must fold to a constant.
    // `@index_no_fold` and `@index_div_by_zero` must not fold.
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @index_fold: llvm.func <builtin.integer i64() variadic = false>
              [] 
            {
              ^entry_block2v1() !1:
                res_v25 = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64 !2;
                llvm.return res_v25 !3
            } !4;
            llvm.func @index_no_fold: llvm.func <builtin.integer i64(builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block3v1(n_v10: builtin.integer i64) !5:
                c3_v11 = index.constant <index.constant 3> : index.index  !6;
                n_idx_v12 = index.from_integer n_v10 : index.index  !7;
                sum_v13 = index.add n_idx_v12, c3_v11 : index.index  !8;
                res_v14 = index.to_integer sum_v13 to builtin.integer i64 !9;
                llvm.return res_v14 !10
            } !11;
            llvm.func @index_div_by_zero: llvm.func <builtin.integer i64() variadic = false>
              [] 
            {
              ^entry_block4v1() !12:
                c3_v15 = index.constant <index.constant 3> : index.index  !13;
                c0_v16 = index.constant <index.constant 0> : index.index  !14;
                quot_v17 = index.divu c3_v15, c0_v16 : index.index  !15;
                res_v18 = index.to_integer quot_v17 to builtin.integer i64 !16;
                llvm.return res_v18 !17
            } !18
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

/// Check that mem2reg promotes an allocation that is loaded and stored in the regions
/// of `cf.for`, `cf.if` and `cf.execute_region`.
#[test]
fn test_region_ops_mem2reg() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let module_op = parse_module(
        ctx,
        r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @promote: llvm.func <builtin.fp32 (builtin.integer i64, builtin.integer i1) variadic = false> [] {
              ^entry(n : builtin.integer i64, cond : builtin.integer i1):
                one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                p = llvm.alloca [builtin.fp32 x one] : llvm.ptr(0);
                f1 = builtin.constant <builtin.single 1.0> : builtin.fp32;
                f2 = builtin.constant <builtin.single 2.0> : builtin.fp32;
                c0 = index.constant <index.constant 0> : index.index;
                c1 = index.constant <index.constant 1> : index.index;
                n_idx = index.from_integer n : index.index;
                llvm.store *p <- f1;
                cf.for c0 to n_idx step c1 () {
                    ^entry(iv : index.index):
                        acc = llvm.load p : builtin.fp32;
                        next = llvm.fadd <FAST> acc, f1 : builtin.fp32;
                        llvm.store *p <- next;
                        cf.yield
                };
                cf.if cond -> () {
                    ^entry():
                        llvm.store *p <- f2;
                        cf.yield
                } else {
                    ^entry():
                        cf.yield
                };
                cf.execute_region -> () {
                    ^entry():
                        cf.cond_br if cond ^on_true() else ^on_false()
                    ^on_true():
                        v = llvm.load p : builtin.fp32;
                        w = llvm.fadd <FAST> v, v : builtin.fp32;
                        llvm.store *p <- w;
                        cf.br ^exit()
                    ^on_false():
                        cf.br ^exit()
                    ^exit():
                        cf.yield
                };
                res = llvm.load p : builtin.fp32;
                llvm.return res
            }
        }
        "#,
    );

    let mut analyses = AnalysisManager::default();
    mem2reg(module_op.get_operation(), ctx, &mut analyses).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @promote: llvm.func <builtin.fp32 (builtin.integer i64, builtin.integer i1) variadic = false>
              [] 
            {
              ^entry_block2v1(n_v0: builtin.integer i64, cond_v1: builtin.integer i1) !1:
                one_v2 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !2;
                f1_v4 = builtin.constant <builtin.single 1> : builtin.fp32  !3;
                f2_v5 = builtin.constant <builtin.single 2> : builtin.fp32  !4;
                c0_v6 = index.constant <index.constant 0> : index.index  !5;
                c1_v7 = index.constant <index.constant 1> : index.index  !6;
                n_idx_v8 = index.from_integer n_v0 : index.index  !7;
                p_v17 = cf.for c0_v6 to n_idx_v8 step c1_v7 (f1_v4) 
                {
                  ^entry_block3v1(iv_v9: index.index , p_v16: builtin.fp32 ) !8:
                    next_v11 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> p_v16, f1_v4 : builtin.fp32  !9;
                    cf.yield next_v11 !10
                }
         !11;
                p_v18 = cf.if cond_v1 -> (builtin.fp32 ) 
                {
                  ^entry_block4v1() !12:
                    cf.yield f2_v5 !13
                } else 
                {
                  ^entry_block5v1() !14:
                    cf.yield p_v17 !15
                }
         !16;
                p_v19 = cf.execute_region -> (builtin.fp32 ) 
                {
                  ^entry_block6v1() !17:
                    cf.cond_br if cond_v1 ^on_true_block9v1() else ^on_false_block10v1() !18

                  ^on_true_block9v1() !19:
                    w_v13 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> p_v18, p_v18 : builtin.fp32  !20;
                    cf.br ^exit_block8v3(w_v13) !21

                  ^on_false_block10v1() !22:
                    cf.br ^exit_block8v3(p_v18) !23

                  ^exit_block8v3(p_v15: builtin.fp32 ) !24:
                    cf.yield p_v15 !25
                }
         !26;
                llvm.return p_v19 !27
            } !28
        }"#]].assert_eq(&module_op.disp(ctx).to_string());
}

/// Check that canonicalization simplifies `index` ops.
#[test]
fn test_index_ops_canonicalize() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let module_op = parse_module(
        ctx,
        r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @index_canonicalize: llvm.func <builtin.integer i64 (builtin.integer i64) variadic = false> [] {
              ^entry(n : builtin.integer i64):
                c0 = index.constant <index.constant 0> : index.index;
                c1 = index.constant <index.constant 1> : index.index;
                c5 = index.constant <index.constant 5> : index.index;
                n_idx = index.from_integer n : index.index;
                a = index.add c0, n_idx : index.index;
                m = index.mul c1, a : index.index;
                z = index.mul m, c0 : index.index;
                d = index.divu m, c1 : index.index;
                s = index.add c5, d : index.index;
                p = index.add s, z : index.index;
                q = index.mul c5, p : index.index;
                q_int = index.to_integer q to builtin.integer i64;
                q_idx = index.from_integer q_int : index.index;
                res = index.to_integer q_idx to builtin.integer i64;
                llvm.return res
            };
            llvm.func @cast_round_trip: llvm.func <builtin.integer i64 (builtin.integer i64) variadic = false> [] {
              ^entry(n : builtin.integer i64):
                n_idx = index.from_integer n : index.index;
                res = index.to_integer n_idx to builtin.integer i64;
                llvm.return res
            }
        }
        "#,
    );

    canonicalize(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);
    dce(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    // `@index_canonicalize` must reduce to `(n + 5) * 5`, with constants on the rhs.
    // `@cast_round_trip` must return `n`.
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @index_canonicalize: llvm.func <builtin.integer i64(builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block2v1(n_v0: builtin.integer i64) !1:
                c5_v3 = index.constant <index.constant 5> : index.index  !2;
                n_idx_v4 = index.from_integer n_v0 : index.index  !3;
                s_v9 = index.add n_idx_v4, c5_v3 : index.index  !4;
                q_v11 = index.mul s_v9, c5_v3 : index.index  !5;
                res_v14 = index.to_integer q_v11 to builtin.integer i64 !6;
                llvm.return res_v14 !7
            } !8;
            llvm.func @cast_round_trip: llvm.func <builtin.integer i64(builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block3v1(n_v15: builtin.integer i64) !9:
                llvm.return n_v15 !10
            } !11
        }"#]].assert_eq(&module_op.disp(ctx).to_string());
}

/// Check that canonicalization simplifies cf ops.
#[test]
fn test_region_ops_canonicalize() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let module_op = parse_module(
        ctx,
        r#"
        builtin.module @test_module {
          ^entry():
            llvm.func @cf_canonicalize: llvm.func <builtin.fp32 (builtin.integer i1, builtin.integer i64) variadic = false> [] {
              ^entry(c : builtin.integer i1, n : builtin.integer i64):
                one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                p = llvm.alloca [builtin.fp32 x one] : llvm.ptr(0);
                f1 = builtin.constant <builtin.single 1.0> : builtin.fp32;
                f2 = builtin.constant <builtin.single 2.0> : builtin.fp32;
                c0 = index.constant <index.constant 0> : index.index;
                c1 = index.constant <index.constant 1> : index.index;
                c4 = index.constant <index.constant 4> : index.index;
                c5 = index.constant <index.constant 5> : index.index;
                n_idx = index.from_integer n : index.index;
                a, b, d = cf.if c -> (builtin.fp32, builtin.fp32, builtin.fp32) {
                    ^entry():
                        llvm.store *p <- f1;
                        cf.yield f1, f2, f1
                } else {
                    ^entry():
                        cf.yield f1, f1, f2
                };
                cf.if c -> () {
                    ^entry():
                        llvm.store *p <- f2;
                        cf.yield
                } else {
                    ^entry():
                        cf.yield
                };
                cf.if c -> () {
                    ^entry():
                        cf.yield
                } else {
                    ^entry():
                        cf.yield
                };
                e = cf.execute_region -> (builtin.fp32) {
                    ^entry():
                        x = llvm.fadd <FAST> a, d : builtin.fp32;
                        cf.yield x
                };
                r = cf.for c0 to c5 step c5 (e) {
                    ^entry(iv : index.index, acc : builtin.fp32):
                        iv_int = index.to_integer iv to builtin.integer i64;
                        iv_fp = llvm.uitofp <nneg=false> iv_int to builtin.fp32;
                        next = llvm.fadd <FAST> acc, iv_fp : builtin.fp32;
                        cf.yield next
                };
                s, inv, same, dead = cf.for c0 to n_idx step c1 (r, f1, f2, f1) {
                    ^entry(iv2 : index.index, acc2 : builtin.fp32, inv2 : builtin.fp32, same2 : builtin.fp32, dead2 : builtin.fp32):
                        next2 = llvm.fadd <FAST> acc2, inv2 : builtin.fp32;
                        dead_next = llvm.fadd <FAST> acc2, f1 : builtin.fp32;
                        cf.yield next2, inv2, f2, dead_next
                };
                cf.nd_for [] to [] step [] {
                    ^entry():
                        llvm.store *p <- s;
                        cf.yield
                };
                cf.nd_for [c4, c0] to [c5, c1] step [c1, c5] {
                    ^entry(i : index.index, j : index.index):
                        ij = index.add i, j : index.index;
                        ij_int = index.to_integer ij to builtin.integer i64;
                        ij_fp = llvm.uitofp <nneg=false> ij_int to builtin.fp32;
                        llvm.store *p <- ij_fp;
                        cf.yield
                };
                g = cf.execute_region -> (builtin.fp32) {
                    ^entry():
                        cf.cond_br if c ^exit(s) else ^exit(s)
                    ^exit(v : builtin.fp32):
                        cf.yield v
                };
                h = llvm.fadd <FAST> g, inv : builtin.fp32;
                k = llvm.fadd <FAST> h, same : builtin.fp32;
                llvm.return k
            };
            llvm.func @not_single_trip: llvm.func <builtin.fp32 () variadic = false> [] {
              ^entry():
                f1 = builtin.constant <builtin.single 1.0> : builtin.fp32;
                c1 = index.constant <index.constant 1> : index.index;
                c2 = index.constant <index.constant 2> : index.index;
                c5 = index.constant <index.constant 5> : index.index;
                max = index.constant <index.constant 18446744073709551615> : index.index;
                r = cf.for c1 to c5 step c2 (f1) {
                    ^entry(iv : index.index, acc : builtin.fp32):
                        next = llvm.fadd <FAST> acc, f1 : builtin.fp32;
                        cf.yield next
                };
                s = cf.for c5 to max step max (r) {
                    ^entry(iv2 : index.index, acc2 : builtin.fp32):
                        next2 = llvm.fadd <FAST> acc2, f1 : builtin.fp32;
                        cf.yield next2
                };
                llvm.return s
            }
        }
        "#,
    );

    canonicalize(module_op.get_operation(), ctx).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    // In `@cf_canonicalize`:
    // - The first `cf.if` must keep only its result `d`, and `a` must be replaced by `f1`.
    // - The second `cf.if` must lose its `else` region, and the third `cf.if` must be erased.
    // - `e`, the single-iteration `cf.for` and both `cf.nd_for` ops must be inlined.
    // - `s` must keep only `acc2`, and `inv` and `same` must be replaced by their initial values.
    // - The `cf.cond_br` in `g` must become a `cf.br`.
    // In `@not_single_trip`, a loop with two iterations, and a loop whose
    // induction variable overflows, must remain.
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @cf_canonicalize: llvm.func <builtin.fp32 (builtin.integer i1, builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block2v1(c_v0: builtin.integer i1, n_v1: builtin.integer i64) !1:
                one_v2 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64 !2;
                p_v3 = llvm.alloca [builtin.fp32  x one_v2]  : llvm.ptr (0) !3;
                f1_v4 = builtin.constant <builtin.single 1> : builtin.fp32  !4;
                f2_v5 = builtin.constant <builtin.single 2> : builtin.fp32  !5;
                c0_v6 = index.constant <index.constant 0> : index.index  !6;
                c1_v7 = index.constant <index.constant 1> : index.index  !7;
                c4_v8 = index.constant <index.constant 4> : index.index  !8;
                c5_v9 = index.constant <index.constant 5> : index.index  !9;
                n_idx_v10 = index.from_integer n_v1 : index.index  !10;
                d_v13 = cf.if c_v0 -> (builtin.fp32 ) 
                {
                  ^entry_block3v1() !11:
                    llvm.store *p_v3 <- f1_v4  !12;
                    cf.yield f1_v4 !13
                } else 
                {
                  ^entry_block4v1() !14:
                    cf.yield f2_v5 !15
                }
         !16;
                cf.if c_v0 -> () 
                {
                  ^entry_block5v1() !17:
                    llvm.store *p_v3 <- f2_v5  !18;
                    cf.yield  !19
                }
         !20;
                x_v15 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> f1_v4, d_v13 : builtin.fp32  !21;
                iv_int_v19 = index.to_integer c0_v6 to builtin.integer i64 !22;
                iv_fp_v20 = llvm.uitofp <nneg=false> iv_int_v19 to builtin.fp32  !23;
                next_v21 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> x_v15, iv_fp_v20 : builtin.fp32  !24;
                s_v22 = cf.for c0_v6 to n_idx_v10 step c1_v7 (next_v21) 
                {
                  ^entry_block11v1(iv2_v26: index.index , acc2_v27: builtin.fp32 ) !25:
                    next2_v31 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> acc2_v27, f1_v4 : builtin.fp32  !26;
                    dead_next_v32 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> acc2_v27, f1_v4 : builtin.fp32  !27;
                    cf.yield next2_v31 !28
                }
         !29;
                llvm.store *p_v3 <- s_v22  !30;
                ij_int_v36 = index.to_integer c4_v8 to builtin.integer i64 !31;
                ij_fp_v37 = llvm.uitofp <nneg=false> ij_int_v36 to builtin.fp32  !32;
                llvm.store *p_v3 <- ij_fp_v37  !33;
                g_v38 = cf.execute_region -> (builtin.fp32 ) 
                {
                  ^entry_block14v1() !34:
                    cf.br ^exit_block16v1(s_v22) !35

                  ^exit_block16v1(v_v39: builtin.fp32 ) !36:
                    cf.yield v_v39 !37
                }
         !38;
                h_v40 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> g_v38, f1_v4 : builtin.fp32  !39;
                k_v41 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> h_v40, f2_v5 : builtin.fp32  !40;
                llvm.return k_v41 !41
            } !42;
            llvm.func @not_single_trip: llvm.func <builtin.fp32 () variadic = false>
              [] 
            {
              ^entry_block15v3() !43:
                f1_v42 = builtin.constant <builtin.single 1> : builtin.fp32  !44;
                c1_v43 = index.constant <index.constant 1> : index.index  !45;
                c2_v44 = index.constant <index.constant 2> : index.index  !46;
                c5_v45 = index.constant <index.constant 5> : index.index  !47;
                max_v46 = index.constant <index.constant 18446744073709551615> : index.index  !48;
                r_v47 = cf.for c1_v43 to c5_v45 step c2_v44 (f1_v42) 
                {
                  ^entry_block17v1(iv_v48: index.index , acc_v49: builtin.fp32 ) !49:
                    next_v50 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> acc_v49, f1_v42 : builtin.fp32  !50;
                    cf.yield next_v50 !51
                }
         !52;
                s_v51 = cf.for c5_v45 to max_v46 step max_v46 (r_v47) 
                {
                  ^entry_block18v1(iv2_v52: index.index , acc2_v53: builtin.fp32 ) !53:
                    next2_v54 = llvm.fadd <NNAN | NINF | NSZ | ARCP | CONTRACT | AFN | REASSOC> acc2_v53, f1_v42 : builtin.fp32  !54;
                    cf.yield next2_v54 !55
                }
         !56;
                llvm.return s_v51 !57
            } !58
        }"#]].assert_eq(&module_op.disp(ctx).to_string());
}
