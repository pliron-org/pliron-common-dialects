// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Convert index dialect to LLVM dialect.

use pliron::{
    attribute::AttrObj,
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::OneOpdInterface,
        ops::ConstantOp,
        types::{IntegerType, Signedness},
    },
    context::Context,
    derive::{op_interface_impl, type_interface_impl},
    irbuild::{
        dialect_conversion::{DialectConversionRewriter, OperandsInfo},
        inserter::Inserter,
        rewriter::Rewriter,
    },
    op::Op,
    result::Result,
    r#type::TypeHandle,
    utils::apint::APInt,
};
use pliron_llvm::{ToLLVMDialect, ToLLVMType};

use crate::index::{
    attributes::ConstantIndexAttr,
    ops::{IndexConstantOp, IndexToIntegerOp, IntegerToIndexOp},
    types::IndexType,
};

#[type_interface_impl]
impl ToLLVMType for IndexType {
    fn convert(&self, ctx: &Context) -> Result<TypeHandle> {
        // Convert IndexType to i64 in LLVM dialect.
        let int_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        Ok(int_ty.into())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for IndexConstantOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let constant_index = self
            .get_attr_constant_index(ctx)
            .expect("Missing value attribute")
            .clone();
        let llvm_attr = constant_index.to_llvm_attr(ctx)?;
        let new_constant_op = ConstantOp::new(ctx, llvm_attr);
        rewriter.insert_op(ctx, &new_constant_op);
        rewriter.replace_operation(ctx, self.get_operation(), new_constant_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for IndexToIntegerOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let index_op = self.get_operand(ctx);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![index_op]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for IntegerToIndexOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let int_op = self.get_operand(ctx);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![int_op]);
        Ok(())
    }
}

impl ConstantIndexAttr {
    /// Convert the constant index attribute to an LLVM attribute.
    fn to_llvm_attr(&self, ctx: &mut Context) -> Result<AttrObj> {
        // Convert the constant index value to an LLVM attribute.
        let int_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let llvm_attr = IntegerAttr::new(
            int_ty,
            APInt::from_u64(
                self.constant_index.try_into().unwrap(),
                64.try_into().unwrap(),
            ),
        );
        Ok(llvm_attr.into())
    }
}
