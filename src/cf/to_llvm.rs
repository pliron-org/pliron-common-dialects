//! Convert control flow dialect to LLVM dialect.

use pliron::{
    basic_block::BasicBlock,
    builtin::op_interfaces::{OneRegionInterface, OneResultInterface},
    context::{Context, Ptr},
    derive::op_interface_impl,
    input_error,
    irbuild::{
        dialect_conversion::{DialectConversion, DialectConversionRewriter, OperandsInfo},
        inserter::{BlockInsertionPoint, IRInserter, Inserter, OpInsertionPoint},
        listener::DummyListener,
        match_rewrite::MatchRewriter,
        rewriter::{Rewriter, ScopedRewriter},
    },
    linked_list::LinkedList,
    location::Located,
    op::{Op, op_cast, op_impls},
    operation::Operation,
    region::Region,
    result::Result,
    r#type::{TypeObj, Typed, type_cast},
    value::Value,
};
use pliron_llvm::{
    ToLLVMDialect, ToLLVMType,
    attributes::{ICmpPredicateAttr, IntegerOverflowFlagsAttr},
    op_interfaces::IntBinArithOpWithOverflowFlag,
    ops::{AddOp, BrOp, CondBrOp, ICmpOp},
};

use crate::cf::{
    op_interfaces::YieldingRegion,
    ops::{ForOp, NDForOp},
};

/// Implement [DialectConversion] for control-flow to LLVM conversion.
pub struct CFToLLVM;

impl DialectConversion for CFToLLVM {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn ToLLVMDialect>(&*Operation::get_op_dyn(op, ctx))
    }

    fn can_convert_type(&self, ctx: &Context, ty: Ptr<TypeObj>) -> bool {
        type_cast::<dyn ToLLVMType>(&**ty.deref(ctx)).is_some()
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: Ptr<TypeObj>) -> Result<Ptr<TypeObj>> {
        let to_llvm_ty = type_cast::<dyn ToLLVMType>(&**ty.deref(ctx))
            .expect("Type with can_convert_type=true must implement ToLLVMType")
            .converter();
        to_llvm_ty(ty, ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let to_llvm_op = op_cast::<dyn ToLLVMDialect>(&*op_dyn)
            .expect("Matched Op must implement ToLLVMDialect");
        to_llvm_op.rewrite(ctx, rewriter, operands_info)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ForOpConversionErr {
    #[error("Unsupported induction variable type for ForOp conversion")]
    UnsupportedIVType,
}

// Rewrite pattern for `ForOp`:
//     <code before ForOp>
//     llvm.br ^header(%lower_bound, %iter_args_init, ...)
//
//  ^header(%iv, %iter_args, ...):
//     %cmp = llvm.icmp %iv LT %upper_bound
//     llvm.cond_br if %cmp ^for_body_entry(%iv, %iter_args) else ^exit
//
//  ^for_body_entry(%iv, %iter_args, ...):
//     ...
//
//  ^for_body_exit(...):
//     ...
//     %yield_operands, ... = ... # remove `yield` op
//     %iv_next = llvm.add %iv, 1
//     llvm.br ^header(%iv_next, %yield_operands, ...)
//
//  ^exit:
//     <code after ForOp>
#[op_interface_impl]
impl ToLLVMDialect for ForOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let lower_bound = self.get_lower_bound(ctx);
        let upper_bound = self.get_upper_bound(ctx);
        let step = self.get_step(ctx);
        let iter_vars_init = self.get_iter_args_init(ctx);
        let iv = self.get_induction_variable(ctx);
        let for_body_entry = self.get_entry(ctx);

        let self_op = self.get_operation();
        let self_op_loc = self_op.deref(ctx).loc();

        let pre_header = self_op
            .deref(ctx)
            .get_parent_block()
            .expect("ForOp must be inside a block");

        let exit_block = rewriter.split_block(
            ctx,
            pre_header,
            OpInsertionPoint::BeforeOperation(self_op),
            None,
        );

        let iv_ty = iv.get_type(ctx);
        let to_llvm_ty = type_cast::<dyn ToLLVMType>(&**iv_ty.deref(ctx))
            .ok_or_else(|| {
                input_error!(self_op_loc.clone(), ForOpConversionErr::UnsupportedIVType)
            })?
            .converter();
        let iv_ty = to_llvm_ty(iv_ty, ctx)?;
        // We change the type of the induction variable to an LLVM integer type.
        rewriter.set_value_type(ctx, iv, iv_ty);

        // We don't convert iter_var_types here because they are just passed
        // through the header without any operations on them. They will be converted
        // when they are used in the body block.
        let iter_var_types = iter_vars_init.iter().map(|arg| arg.get_type(ctx));

        // Setup the header
        let header = rewriter.create_block(
            ctx,
            BlockInsertionPoint::AfterBlock(pre_header),
            Some("for_op_header".try_into().unwrap()),
            std::iter::once(iv_ty).chain(iter_var_types).collect(),
        );
        header.deref_mut(ctx).set_loc(self_op_loc);
        let header_iv = header.deref(ctx).get_argument(0);
        let header_args = header.deref(ctx).arguments().collect::<Vec<_>>();
        let cmp = ICmpOp::new(ctx, ICmpPredicateAttr::ULT, header_iv, upper_bound);
        let cmp_result = cmp.get_result(ctx);
        rewriter.insert_op(ctx, &cmp);
        let cond_br = CondBrOp::new(
            ctx,
            cmp_result,
            for_body_entry,
            header_args.clone(),
            exit_block,
            vec![],
        );
        rewriter.insert_op(ctx, &cond_br);

        // Pre-header must branch to header with initial induction variable and iter args
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(pre_header));
        let pre_header_br = BrOp::new(
            ctx,
            header,
            std::iter::once(lower_bound)
                .chain(iter_vars_init.iter().cloned())
                .collect(),
        );
        rewriter.insert_op(ctx, &pre_header_br);

        // Set the for body exit block to to branch to the header
        // with the next induction variable and iter args from yield.
        let for_body_entry_iv = for_body_entry.deref(ctx).get_argument(0);
        let yield_op = self.get_yield(ctx).get_operation();
        rewriter.set_insertion_point(OpInsertionPoint::AfterOperation(yield_op));
        let iv_next = AddOp::new_with_overflow_flag(
            ctx,
            for_body_entry_iv,
            step,
            IntegerOverflowFlagsAttr::default(),
        );
        rewriter.append_op(ctx, &iv_next);
        let branch_operands: Vec<_> = std::iter::once(iv_next.get_result(ctx))
            .chain(yield_op.deref(ctx).operands())
            .collect();
        let for_body_exit_br = BrOp::new(ctx, header, branch_operands);
        rewriter.append_op(ctx, &for_body_exit_br);
        rewriter.erase_operation(ctx, yield_op);

        // Inline the body of the ForOp, to after the header block.
        rewriter.inline_region(
            ctx,
            self.get_region(ctx),
            BlockInsertionPoint::AfterBlock(header),
        );

        // Replace uses of the ForOp with header block arguments
        // (except the induction variable which is not used outside).
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            header_args.into_iter().skip(1).collect(),
        );

        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum NDForOpConversionErr {
    #[error("Unsupported induction variable type for NDForOp conversion")]
    UnsupportedIVType,
}

#[op_interface_impl]
impl ToLLVMDialect for NDForOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let lower_bounds = self.get_lower_bounds(ctx);
        let upper_bounds = self.get_upper_bounds(ctx);
        let steps = self.get_steps(ctx);
        let region = self.get_region(ctx);

        // Remove the [YieldOp] in the body, it's useless and impedes LLVM conversion.
        let yield_op = self.get_yield(ctx);
        rewriter.erase_operation(ctx, yield_op.get_operation());

        // Iterate over the loop dimensions in reverse order,
        // so that we can create nested loops from the innermost to the outermost.
        let mut lb_ub_st = lower_bounds
            .iter()
            .zip(upper_bounds.iter())
            .zip(steps.iter())
            .rev();

        // The innermost loop gets the body of the original NDForOp as its loop body.
        let ((innermost_lb, innermost_ub), innermost_step) = lb_ub_st
            .next()
            .expect("NDForOp must have at least one loop dimension");

        struct State<'a> {
            innermost_for_op_entry_block: Option<Ptr<BasicBlock>>,
            last_created_for_op: Option<ForOp>,
            indices: Vec<Value>,
            rewriter: &'a mut MatchRewriter,
            ndforop_region: Ptr<Region>,
        }
        let mut state = State {
            innermost_for_op_entry_block: None,
            last_created_for_op: None,
            indices: vec![],
            rewriter,
            ndforop_region: region,
        };
        let innermost_for = ForOp::new(
            ctx,
            *innermost_lb,
            *innermost_ub,
            *innermost_step,
            &[],
            |ctx: &mut Context,
             state: &mut State,
             inserter: &mut IRInserter<DummyListener>,
             idx: Value,
             iter_args: &[Value]| {
                assert!(
                    iter_args.is_empty(),
                    "We didn't provide any init iter args, so the body shouldn't expect any iter args"
                );
                let insertion_point = inserter
                    .get_insertion_block(ctx)
                    .expect("Failed to get insertion block");
                // Move the body of the original NDForOp into the innermost ForOp.
                state.rewriter.inline_region(
                    ctx,
                    state.ndforop_region,
                    BlockInsertionPoint::AfterBlock(insertion_point),
                );
                // Note the entry block of the innermost ForOp for later convenience.
                state.innermost_for_op_entry_block = Some(insertion_point);
                state.indices.push(idx);
                vec![]
            },
            &mut state,
        );
        state.last_created_for_op = Some(innermost_for);

        // To add a branch from the innermost loop entry block to the original body entry block,
        // we don't have all the induction variables available until we create all the loops.
        // So add the branch later.

        // Now create the outer loops, if any.
        for ((lb, ub), step) in lb_ub_st {
            let for_op = ForOp::new(
                ctx,
                *lb,
                *ub,
                *step,
                &[],
                |ctx: &mut Context,
                 state: &mut State,
                 inserter: &mut IRInserter<DummyListener>,
                 idx: Value,
                 iter_args: &[Value]|
                 -> Vec<Value> {
                    assert!(
                        iter_args.is_empty(),
                        "We didn't provide any init iter args, so the body shouldn't expect any iter args"
                    );

                    // Use the outer rewriter for insertions to keep track of new operations for later loops.
                    let mut rewriter =
                        ScopedRewriter::new(state.rewriter, inserter.get_insertion_point());
                    // The entry block will have just one operation, which is the previous ForOp we created.
                    rewriter.append_op(ctx, &state.last_created_for_op.unwrap());
                    state.indices.push(idx);
                    vec![]
                },
                &mut state,
            );
            state.last_created_for_op = Some(for_op);
        }

        // Now we have all the loops created, we can add the branch from the
        // innermost loop entry block to the original body entry block.
        {
            let innermost_for_op_entry_block = state
                .innermost_for_op_entry_block
                .expect("We must have created at least one ForOp, so the innermost loop entry block must be set");
            let branch_to_block = innermost_for_op_entry_block.deref(ctx).get_next()
                .expect("The body of the original NDForOp must be in the next block after the innermost ForOp entry block");
            let mut branch_inserter = ScopedRewriter::new(
                state.rewriter,
                OpInsertionPoint::AtBlockEnd(state.innermost_for_op_entry_block.unwrap()),
            );
            let branch = BrOp::new(
                ctx,
                branch_to_block,
                state.indices.iter().rev().cloned().collect(),
            );
            branch_inserter.append_op(ctx, &branch);
        }
        // Finally replace the NDForOp with the last created ForOp, which is the outermost loop.
        state
            .rewriter
            .append_op(ctx, &state.last_created_for_op.unwrap());
        state.rewriter.replace_operation(
            ctx,
            self.get_operation(),
            state.last_created_for_op.unwrap().get_operation(),
        );

        Ok(())
    }
}
