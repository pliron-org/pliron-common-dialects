// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! op interfaces for CF dialect.

use pliron::{
    basic_block::BasicBlock,
    builtin::op_interfaces::{BranchOpInterface, IsTerminatorInterface, OneRegionInterface},
    context::{Context, Ptr},
    derive::op_interface,
    linked_list::ContainsLinkedList,
    op::{Op, op_cast},
    operation::Operation,
    result::Result,
    verify_err,
};

/// An [Operation] that can be yielded to from a [YieldingRegion].
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
pub enum YieldingRegionVerifyErr {
    #[error("Region is empty")]
    RegionEmpty,
    #[error("Last operation in exit block is not a YieldingOp")]
    LastOpNotYield,
    #[error("Terminator of a non-exit block must implement BranchOpInterface")]
    NonExitBlockBadTerminator,
}

/// An [Operation] with a single [Region](pliron::region::Region) that
/// 1. Has an entry (lexicographically first) and an exit (lexicographically last) block.
/// 2. The exit block must end with a [YieldingOp].
///
/// The entry and exit blocks can be the same block.
#[op_interface]
pub trait YieldingRegion<YieldOp: YieldingOp>: OneRegionInterface {
    /// Get the `yield` operation in the loop body.
    fn get_yield(&self, ctx: &Context) -> YieldOp {
        let exit_block = self.get_exit(ctx);
        let yield_op = exit_block
            .deref(ctx)
            .get_tail()
            .expect("Block must have at least one operation");
        Operation::get_op::<YieldOp>(yield_op, ctx)
            .expect("The last operation in a ForOp exit block must be a YieldOp")
    }

    /// Get the entry block of the loop body region.
    fn get_entry(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_region(ctx)
            .deref(ctx)
            .get_head()
            .expect("ForOp region must have at least one block")
    }

    /// Get the exit block of the loop body region.
    fn get_exit(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_region(ctx)
            .deref(ctx)
            .get_tail()
            .expect("ForOp region must have at least one block")
    }

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let op = op_cast::<dyn YieldingRegion<YieldOp>>(op)
            .expect("Expected a YieldingRegion operation");
        let region = op.get_region(ctx).deref(ctx);

        if region.get_head().is_none() {
            return verify_err!(op.loc(ctx), YieldingRegionVerifyErr::RegionEmpty);
        };

        let exit_block = region
            .get_tail()
            .expect("Region must have at least one block");
        let yield_op = exit_block
            .deref(ctx)
            .get_tail()
            .expect("Block must have at least one operation");
        if Operation::get_op::<YieldOp>(yield_op, ctx).is_none() {
            return verify_err!(op.loc(ctx), YieldingRegionVerifyErr::LastOpNotYield);
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
                    YieldingRegionVerifyErr::NonExitBlockBadTerminator
                );
            }
        }

        Ok(())
    }
}
