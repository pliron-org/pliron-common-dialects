// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Index operations

use pliron::{
    builtin::{
        op_interfaces::{
            AllOperandsOfType, AllResultsOfType, NOpdsInterface, NResultsInterface,
            OneOpdInterface, OneResultInterface,
        },
        types::IntegerType,
    },
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    verify_err,
};
use pliron_llvm::op_interfaces::CastOpInterface;

use crate::index::{attributes::ConstantIndexAttr, types::IndexType};

/// Constant operation for IndexType.
#[pliron_op(
    name = "index.constant",
    interfaces = [
        NOpdsInterface<0>,
        OneResultInterface,
        NResultsInterface<1>,
        AllResultsOfType<IndexType>
    ],
    format = "`<` $constant_index `>` ` : ` type($0)",
    attributes = (constant_index: ConstantIndexAttr),
    verifier = "succ",
)]
pub struct IndexConstantOp;

impl IndexConstantOp {
    /// Create a new IndexConstantOp with the given constant index value.
    pub fn new(ctx: &mut Context, constant_index: usize) -> Self {
        let constant_index_attr = ConstantIndexAttr { constant_index };

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![IndexType::get(ctx).into()],
            vec![],
            vec![],
            0,
        );

        let op = Self { op };
        op.set_attr_constant_index(ctx, constant_index_attr);
        op
    }
}

/// Index type to Integer type conversion operation.
#[pliron_op(
    name = "index.to_integer",
    interfaces = [
        NOpdsInterface<1>,
        OneOpdInterface,
        OneResultInterface,
        NResultsInterface<1>,
        CastOpInterface,
        AllOperandsOfType<IndexType>,
        AllResultsOfType<IntegerType>
    ],
    format = "$0 ` to ` type($0)",
)]
pub struct IndexToIntegerOp;

#[derive(Debug, thiserror::Error)]
pub enum IndexIntegerCastError {
    #[error("Integer type in casting must be \"builtin.integer i64\"")]
    InvalidIntegerType,
}
impl Verify for IndexToIntegerOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let result_ty = self.result_type(ctx).deref(ctx);

        let int_ty = result_ty
            .downcast_ref::<IntegerType>()
            .expect("Result type must be IntegerType");

        if int_ty.width() != 64 || !int_ty.is_signless() {
            return verify_err!(self.loc(ctx), IndexIntegerCastError::InvalidIntegerType);
        }

        Ok(())
    }
}

/// Integer type to Index type conversion operation.
#[pliron_op(
    name = "index.from_integer",
    interfaces = [
        NOpdsInterface<1>,
        OneOpdInterface,
        OneResultInterface,
        NResultsInterface<1>,
        CastOpInterface,
        AllOperandsOfType<IntegerType>,
        AllResultsOfType<IndexType>
    ],
    format = "$0 ` : ` type($0)",
)]
pub struct IntegerToIndexOp;

impl Verify for IntegerToIndexOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let opd_ty = self.operand_type(ctx).deref(ctx);

        let int_ty = opd_ty
            .downcast_ref::<IntegerType>()
            .expect("Operand type must be IntegerType");

        if int_ty.width() != 64 || !int_ty.is_signless() {
            return verify_err!(self.loc(ctx), IndexIntegerCastError::InvalidIntegerType);
        }

        Ok(())
    }
}
