// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Control-flow dialect for Pliron.

use pliron::derive::op_interface;

use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::op::Op;
use pliron::{context::Context, result::Result};

pub mod op_interfaces;
pub mod ops;
pub mod to_llvm;

/// Interface for rewriting to CF dialect.
#[op_interface]
pub trait ToCFDialect {
    /// Rewrite [self] to CF dialect.
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
