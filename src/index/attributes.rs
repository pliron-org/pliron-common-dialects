// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Index dialect attributes

use pliron::{
    attribute::AttrObj,
    builtin::attr_interfaces::{MaterializableAttr, TypedAttrInterface},
    context::{Context, Ptr},
    derive::{attr_interface_impl, pliron_attr},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
};

use crate::index::{ops::IndexConstantOp, types::IndexType};

/// Attribute for index constant values.
#[pliron_attr(name = "index.constant", format = "$constant_index", verifier = "succ")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstantIndexAttr {
    /// The constant index value.
    pub constant_index: usize,
}

impl ConstantIndexAttr {
    /// The value as a 64-bit index.
    pub fn to_u64(&self) -> u64 {
        u64::try_from(self.constant_index).expect("An index must fit in 64 bits")
    }

    /// Create a [ConstantIndexAttr] from `value`, if it fits in a `usize`.
    pub fn from_u64(value: u64) -> Option<Self> {
        usize::try_from(value)
            .ok()
            .map(|constant_index| ConstantIndexAttr { constant_index })
    }
}

/// Get the value of `attr` if it is a known [ConstantIndexAttr].
pub fn known_constant_index(attr: &Option<AttrObj>) -> Option<u64> {
    attr.as_ref()?
        .downcast_ref::<ConstantIndexAttr>()
        .map(ConstantIndexAttr::to_u64)
}

#[attr_interface_impl]
impl TypedAttrInterface for ConstantIndexAttr {
    fn get_type(&self, ctx: &Context) -> TypeHandle {
        IndexType::get(ctx).into()
    }
}

#[attr_interface_impl]
impl MaterializableAttr for ConstantIndexAttr {
    fn materialize(&self, ctx: &mut Context) -> Ptr<Operation> {
        IndexConstantOp::new(ctx, self.constant_index).get_operation()
    }
}
