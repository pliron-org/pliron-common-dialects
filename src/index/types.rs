// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Index type

use pliron::derive::pliron_type;

/// Index type.
#[pliron_type(name = "index.index", format, generate_get = true, verifier = "succ")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexType;
