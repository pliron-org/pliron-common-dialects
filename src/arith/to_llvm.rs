// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Convert arith dialect to LLVM dialect.

use pliron::{
    context::Context,
    derive::op_interface_impl,
    irbuild::{
        dialect_conversion::{DialectConversionRewriter, OperandsInfo},
        inserter::Inserter,
        rewriter::Rewriter,
    },
    op::Op,
    result::Result,
};
use pliron_llvm::{
    ToLLVMDialect, attributes::IntegerOverflowFlagsAttr,
    op_interfaces::IntBinArithOpWithOverflowFlag, ops::AddOp,
};

use crate::arith::ops::AddIOp;

/// [AddIOp] lowers to [AddOp] with no overflow flags,
/// since [AddIOp] wraps on overflow.
#[op_interface_impl]
impl ToLLVMDialect for AddIOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let (lhs, rhs) = {
            let op = self.get_operation().deref(ctx);
            (op.get_operand(0), op.get_operand(1))
        };
        let add = AddOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
        rewriter.insert_op(ctx, &add);
        rewriter.replace_operation(ctx, self.get_operation(), add.get_operation());
        Ok(())
    }
}
