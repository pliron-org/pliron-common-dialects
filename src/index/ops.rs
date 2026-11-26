// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Index operations

use std::num::NonZero;

use pliron::{
    attribute::AttrObj,
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::{
            AllOperandsOfType, AllResultsOfType, NOpdsInterface, NResultsInterface,
            OneOpdInterface, OneResultInterface,
        },
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    derive::{op_interface_impl, pliron_op},
    irbuild::{IRStatus, rewriter::Rewriter},
    op::Op,
    operation::Operation,
    opts::{canonicalize::CanonicalizeInterface, constants::ConstFoldInterface},
    result::Result,
    utils::apint::APInt,
    value::Value,
    verify_err,
};
use pliron_llvm::op_interfaces::CastOpInterface;

use crate::index::{
    attributes::{ConstantIndexAttr, known_constant_index},
    types::IndexType,
};

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

        let result_ty = IndexType::get(ctx).into();
        let op = Operation::new_op::<Self>(ctx, vec![result_ty], vec![], vec![], 0);
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

/// Addition of two [IndexType] values.
#[pliron_op(
    name = "index.add",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        AllOperandsOfType<IndexType>,
        AllResultsOfType<IndexType>
    ],
    format = "$0 `, ` $1 ` : ` type($0)",
    verifier = "succ",
)]
pub struct IndexAddOp;

impl IndexAddOp {
    /// Create a new IndexAddOp that computes `lhs + rhs`.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        let result_ty = IndexType::get(ctx).into();
        Operation::new_op::<Self>(ctx, vec![result_ty], vec![lhs, rhs], vec![], 0)
    }
}

/// Multiplication of two [IndexType] values.
#[pliron_op(
    name = "index.mul",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        AllOperandsOfType<IndexType>,
        AllResultsOfType<IndexType>,
    ],
    format = "$0 `, ` $1 ` : ` type($0)",
    verifier = "succ",
)]
pub struct IndexMulOp;

impl IndexMulOp {
    /// Create a new IndexMulOp that computes `lhs * rhs`.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        let result_ty = IndexType::get(ctx).into();
        Operation::new_op::<Self>(ctx, vec![result_ty], vec![lhs, rhs], vec![], 0)
    }
}

/// Unsigned division of two [IndexType] values, rounded toward zero.
///
/// Division by zero is undefined behavior.
#[pliron_op(
    name = "index.divu",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        AllOperandsOfType<IndexType>,
        AllResultsOfType<IndexType>,
    ],
    format = "$0 `, ` $1 ` : ` type($0)",
    verifier = "succ",
)]
pub struct IndexDivUOp;

impl IndexDivUOp {
    /// Create a new IndexDivUOp to compute `lhs / rhs`.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        let result_ty = IndexType::get(ctx).into();
        Operation::new_op::<Self>(ctx, vec![result_ty], vec![lhs, rhs], vec![], 0)
    }
}

/// Bit width (post lowering to LLVM integer) of [IndexType] values.
const INDEX_BITWIDTH: NonZero<usize> = NonZero::new(64).unwrap();

/// Fold a binary [IndexType] op on constant operands with `combine`.
fn check_fold_index_bin_op(
    operand_attrs: &[Option<AttrObj>],
    combine: fn(u64, u64) -> Option<u64>,
) -> Vec<Option<AttrObj>> {
    let [lhs, rhs] = operand_attrs else {
        panic!("A binary index op must have two operands");
    };
    let folded = known_constant_index(lhs)
        .zip(known_constant_index(rhs))
        .and_then(|(lhs, rhs)| combine(lhs, rhs))
        .and_then(ConstantIndexAttr::from_u64)
        .map(|attr| Box::new(attr) as AttrObj);
    vec![folded]
}

#[op_interface_impl]
impl ConstFoldInterface for IndexConstantOp {
    fn check_fold(
        &self,
        ctx: &Context,
        _operand_attrs: &[Option<AttrObj>],
    ) -> Vec<Option<AttrObj>> {
        let attr = self
            .get_attr_constant_index(ctx)
            .expect("Missing value attribute")
            .clone();
        vec![Some(Box::new(attr) as AttrObj)]
    }

    fn fold_in_place(
        &self,
        _ctx: &mut Context,
        _operand_attrs: &[Option<AttrObj>],
        _rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        IRStatus::Unchanged
    }
}

#[op_interface_impl]
impl ConstFoldInterface for IndexToIntegerOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let [operand] = operand_attrs else {
            panic!("IndexToIntegerOp must have one operand");
        };
        let folded = known_constant_index(operand).map(|value| {
            let int_ty = IntegerType::get(ctx, INDEX_BITWIDTH.get() as u32, Signedness::Signless);
            Box::new(IntegerAttr::new(
                int_ty,
                APInt::from_u64(value, INDEX_BITWIDTH),
            )) as AttrObj
        });
        vec![folded]
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for IntegerToIndexOp {
    fn check_fold(
        &self,
        _ctx: &Context,
        operand_attrs: &[Option<AttrObj>],
    ) -> Vec<Option<AttrObj>> {
        let [operand] = operand_attrs else {
            panic!("IntegerToIndexOp must have one operand");
        };
        let folded = operand
            .as_ref()
            .and_then(|attr| attr.downcast_ref::<IntegerAttr>())
            .and_then(|int_attr| ConstantIndexAttr::from_u64(int_attr.value().to_u64()))
            .map(|attr| Box::new(attr) as AttrObj);
        vec![folded]
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for IndexAddOp {
    fn check_fold(
        &self,
        _ctx: &Context,
        operand_attrs: &[Option<AttrObj>],
    ) -> Vec<Option<AttrObj>> {
        check_fold_index_bin_op(operand_attrs, |lhs, rhs| Some(lhs.wrapping_add(rhs)))
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for IndexMulOp {
    fn check_fold(
        &self,
        _ctx: &Context,
        operand_attrs: &[Option<AttrObj>],
    ) -> Vec<Option<AttrObj>> {
        check_fold_index_bin_op(operand_attrs, |lhs, rhs| Some(lhs.wrapping_mul(rhs)))
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for IndexDivUOp {
    fn check_fold(
        &self,
        _ctx: &Context,
        operand_attrs: &[Option<AttrObj>],
    ) -> Vec<Option<AttrObj>> {
        // `checked_div` doesn't division by zero (undefined behavior).
        check_fold_index_bin_op(operand_attrs, u64::checked_div)
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

/// Try and match `value` to a constant index.
pub fn try_match_constant_index(ctx: &Context, value: Value) -> Option<u64> {
    let def = value.defining_op()?;
    let const_op = Operation::get_op::<IndexConstantOp>(def, ctx)?;
    let attr = const_op.get_attr_constant_index(ctx)?;
    Some(attr.to_u64())
}

/// - For a constant `c`, `add c, x` canonicalizes to `add x, c`.
/// - `add x, 0` simplifies to `x`.
#[op_interface_impl]
impl CanonicalizeInterface for IndexAddOp {
    fn canonicalize(&self, ctx: &mut Context, rewriter: &mut dyn Rewriter) -> IRStatus {
        let op = self.get_operation();
        let lhs = op.deref(ctx).get_operand(0);
        let rhs = op.deref(ctx).get_operand(1);
        match (
            try_match_constant_index(ctx, lhs),
            try_match_constant_index(ctx, rhs),
        ) {
            (Some(_), None) => {
                Operation::replace_operand(op, ctx, 0, rhs);
                Operation::replace_operand(op, ctx, 1, lhs);
                rewriter.notify_operation_modified(ctx, op);
            }
            (_, Some(0)) => rewriter.replace_operation_with_values(ctx, op, vec![lhs]),
            _ => return IRStatus::Unchanged,
        }
        IRStatus::Changed
    }
}

/// - For a constant `c`, `mul c, x` canonicalizes to `mul x, c`.
/// - `mul x, 1` simplifies to `x`.
/// - `mul x, 0` simplifies to `0`.
#[op_interface_impl]
impl CanonicalizeInterface for IndexMulOp {
    fn canonicalize(&self, ctx: &mut Context, rewriter: &mut dyn Rewriter) -> IRStatus {
        let op = self.get_operation();
        let lhs = op.deref(ctx).get_operand(0);
        let rhs = op.deref(ctx).get_operand(1);
        match (
            try_match_constant_index(ctx, lhs),
            try_match_constant_index(ctx, rhs),
        ) {
            (Some(_), None) => {
                Operation::replace_operand(op, ctx, 0, rhs);
                Operation::replace_operand(op, ctx, 1, lhs);
                rewriter.notify_operation_modified(ctx, op);
            }
            (_, Some(1)) => rewriter.replace_operation_with_values(ctx, op, vec![lhs]),
            (_, Some(0)) => rewriter.replace_operation_with_values(ctx, op, vec![rhs]),
            _ => return IRStatus::Unchanged,
        }
        IRStatus::Changed
    }
}

/// `divu x, 1` simplifies to `x`.
#[op_interface_impl]
impl CanonicalizeInterface for IndexDivUOp {
    fn canonicalize(&self, ctx: &mut Context, rewriter: &mut dyn Rewriter) -> IRStatus {
        let op = self.get_operation();
        let rhs = op.deref(ctx).get_operand(1);
        if try_match_constant_index(ctx, rhs) != Some(1) {
            return IRStatus::Unchanged;
        }
        let lhs = op.deref(ctx).get_operand(0);
        rewriter.replace_operation_with_values(ctx, op, vec![lhs]);
        IRStatus::Changed
    }
}

/// `index.to_integer (index.from_integer x)` simplifies to `x`.
#[op_interface_impl]
impl CanonicalizeInterface for IndexToIntegerOp {
    fn canonicalize(&self, ctx: &mut Context, rewriter: &mut dyn Rewriter) -> IRStatus {
        let Some(from_integer) = self
            .get_operand(ctx)
            .defining_op()
            .and_then(|def| Operation::get_op::<IntegerToIndexOp>(def, ctx))
        else {
            return IRStatus::Unchanged;
        };
        let int = from_integer.get_operand(ctx);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![int]);
        IRStatus::Changed
    }
}

/// `index.from_integer (index.to_integer x)` simplifies to `x`.
#[op_interface_impl]
impl CanonicalizeInterface for IntegerToIndexOp {
    fn canonicalize(&self, ctx: &mut Context, rewriter: &mut dyn Rewriter) -> IRStatus {
        let Some(to_integer) = self
            .get_operand(ctx)
            .defining_op()
            .and_then(|def| Operation::get_op::<IndexToIntegerOp>(def, ctx))
        else {
            return IRStatus::Unchanged;
        };
        let index = to_integer.get_operand(ctx);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![index]);
        IRStatus::Changed
    }
}

impl_no_self_side_effects!(
    IndexConstantOp,
    IndexToIntegerOp,
    IntegerToIndexOp,
    IndexAddOp,
    IndexMulOp,
    IndexDivUOp,
);
