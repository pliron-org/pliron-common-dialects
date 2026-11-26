// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Arith operations

use pliron::{
    attribute::AttrObj,
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::{
            AllOperandsOfType, NOpdsInterface, OneResultInterface, SameOperandsAndResultType,
            SameOperandsType, SameResultsType,
        },
        types::IntegerType,
    },
    common_traits::Verify,
    context::Context,
    derive::{op_interface_impl, pliron_op},
    i,
    irbuild::{IRStatus, rewriter::Rewriter},
    op::Op,
    operation::Operation,
    opts::constants::ConstFoldInterface,
    result::Result,
    r#type::Typed,
    utils::apint::APInt,
    value::Value,
    verify_err,
};

/// Integer addition, modulo 2^n where n is the bitwidth.
///
/// Operands are treated as bit patterns, so the same result is correct for
/// both signed (two's complement) and unsigned interpretations.
///
/// ### Operands:
///
/// | operand | description |
/// |-----|-------|
/// | `lhs` | Signless, non-zero bitwidth integer |
/// | `rhs` | Signless, non-zero bitwidth integer |
///
/// ### Result(s):
///
/// | result | description |
/// |-----|-------|
/// | `res` | Same type as the operands |
#[pliron_op(
    name = "arith.addi",
    interfaces = [
        NOpdsInterface<2>,
        OneResultInterface,
        AllOperandsOfType<IntegerType>,
        SameOperandsAndResultType,
        SameOperandsType,
        SameResultsType
    ],
    format = "$0 `, ` $1 ` : ` type($0)"
)]
pub struct AddIOp;

impl AddIOp {
    /// Create a new [AddIOp] that computes `lhs + rhs`.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        Operation::new_op::<Self>(ctx, vec![lhs.get_type(ctx)], vec![lhs, rhs], vec![], 0)
    }
}

impl Verify for AddIOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let operand_ty = self.operand_type_i(ctx, i!(0)).deref(ctx);

        let int_ty = operand_ty.downcast_ref::<IntegerType>().expect(
            "Operand type must be IntegerType, guaranteed by AllOperandsOfType<IntegerType>",
        );

        if int_ty.width() == 0 {
            return verify_err!(self.loc(ctx), IntBinOpVerifyErr::ZeroWidth);
        }

        if !int_ty.is_signless() {
            return verify_err!(self.loc(ctx), IntBinOpVerifyErr::NotSignless);
        }

        Ok(())
    }
}

/// Verification errors for integer binary arith ops.
#[derive(Debug, thiserror::Error)]
pub enum IntBinOpVerifyErr {
    /// The operand integer type is signed or unsigned.
    #[error("Integer binary arithmetic Op operand must be a signless integer")]
    NotSignless,
    /// The operand integer type has zero bitwidth.
    #[error("Integer binary arithmetic Op operand must have a non-zero bitwidth")]
    ZeroWidth,
}

/// Fold a binary [IntegerType] op on constant operands with `combine`.
fn check_fold_int_bin_op(
    operand_attrs: &[Option<AttrObj>],
    combine: fn(&APInt, &APInt) -> APInt,
) -> Vec<Option<AttrObj>> {
    let [Some(lhs), Some(rhs)] = operand_attrs else {
        assert!(operand_attrs.len() == 2);
        return vec![None];
    };
    let (Some(lhs), Some(rhs)) = (
        lhs.downcast_ref::<IntegerAttr>(),
        rhs.downcast_ref::<IntegerAttr>(),
    ) else {
        return vec![None];
    };
    let folded = IntegerAttr::new(lhs.get_type(), combine(&lhs.value(), &rhs.value()));
    vec![Some(Box::new(folded) as AttrObj)]
}

#[op_interface_impl]
impl ConstFoldInterface for AddIOp {
    fn check_fold(
        &self,
        _ctx: &Context,
        operand_attrs: &[Option<AttrObj>],
    ) -> Vec<Option<AttrObj>> {
        check_fold_int_bin_op(operand_attrs, APInt::add)
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

// [AddIOp] only computes a value from its operands, so it is safe to
// erase when its result is unused.
impl_no_self_side_effects!(AddIOp);
