// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! op interfaces for CF dialect.

use pliron::{
    basic_block::BasicBlock,
    builtin::op_interfaces::{BranchOpInterface, IsTerminatorInterface},
    context::{Context, Ptr},
    derive::op_interface,
    linked_list::ContainsLinkedList,
    op::{Op, op_cast},
    operation::Operation,
    result::Result,
    verify_err,
};

/// An [Operation] that can be yielded to from a [YieldingRegions].
#[op_interface]
pub trait YieldingOp: IsTerminatorInterface {
    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum YieldingRegionsVerifyErr {
    #[error("Region is empty")]
    RegionEmpty,
    #[error("Last operation in exit block is not a YieldingOp")]
    LastOpNotYield,
    #[error("Terminator of a non-exit block must implement BranchOpInterface")]
    NonExitBlockBadTerminator,
}

/// An [Operation] each of whose [Region](pliron::region::Region)s:
/// 1. Has an entry (lexicographically first) and an exit (lexicographically last) block.
/// 2. Has its exit block end with a [YieldingOp].
///
/// The entry and exit blocks of a region can be the same block.
#[op_interface]
pub trait YieldingRegions<YieldOp: YieldingOp> {
    /// Get the `yield` operation ending the `reg_idx`th region.
    fn get_yield(&self, ctx: &Context, reg_idx: usize) -> YieldOp {
        let exit_block = self.get_exit(ctx, reg_idx);
        let yield_op = exit_block
            .deref(ctx)
            .get_tail()
            .expect("Block must have at least one operation");
        Operation::get_op::<YieldOp>(yield_op, ctx)
            .expect("The last operation in a yielding region's exit block must be a YieldOp")
    }

    /// Get the entry block of the `reg_idx`th region.
    fn get_entry(&self, ctx: &Context, reg_idx: usize) -> Ptr<BasicBlock> {
        self.get_operation()
            .deref(ctx)
            .get_region(reg_idx)
            .deref(ctx)
            .get_head()
            .expect("Region must have at least one block")
    }

    /// Get the exit block of the `reg_idx`th region.
    fn get_exit(&self, ctx: &Context, reg_idx: usize) -> Ptr<BasicBlock> {
        self.get_operation()
            .deref(ctx)
            .get_region(reg_idx)
            .deref(ctx)
            .get_tail()
            .expect("Region must have at least one block")
    }

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let num_regions = op.get_operation().deref(ctx).num_regions();
        for reg_idx in 0..num_regions {
            let region = op.get_operation().deref(ctx).get_region(reg_idx).deref(ctx);

            if region.get_head().is_none() {
                return verify_err!(op.loc(ctx), YieldingRegionsVerifyErr::RegionEmpty);
            };

            let exit_block = region
                .get_tail()
                .expect("Region must have at least one block");
            let yield_op = exit_block
                .deref(ctx)
                .get_tail()
                .expect("Block must have at least one operation");
            if Operation::get_op::<YieldOp>(yield_op, ctx).is_none() {
                return verify_err!(op.loc(ctx), YieldingRegionsVerifyErr::LastOpNotYield);
            }

            // Every non-exit block's terminator must implement [BranchOpInterface],
            // so that control flow inside this region is always well understood.
            for block in region.iter(ctx).filter(|&block| block != exit_block) {
                let Some(terminator) = block.deref(ctx).get_terminator(ctx) else {
                    continue;
                };
                let terminator_op = Operation::get_op_dyn(terminator, ctx);
                if op_cast::<dyn BranchOpInterface>(terminator_op.as_ref()).is_none() {
                    return verify_err!(
                        op.loc(ctx),
                        YieldingRegionsVerifyErr::NonExitBlockBadTerminator
                    );
                }
            }
        }

        Ok(())
    }
}
