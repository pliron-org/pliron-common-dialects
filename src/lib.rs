// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Commonly used Pliron Dialects

/// Implement [SideEffects](pliron::opts::dce::SideEffects) with
/// [has_self_side_effects](pliron::opts::dce::SideEffects::has_self_side_effects)
/// returning `false`.
macro_rules! impl_no_self_side_effects {
    ($($op:ty),+ $(,)?) => {
        $(
            #[pliron::derive::op_interface_impl]
            impl pliron::opts::dce::SideEffects for $op {
                fn has_self_side_effects(&self, _ctx: &pliron::context::Context) -> bool {
                    false
                }
            }
        )+
    };
}

pub mod cf;
pub mod index;
