// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Control-flow ops and related functionality.

use std::ops::Range;

use pliron::{
    attribute::{AttrObj, Attribute},
    basic_block::BasicBlock,
    builtin::{
        attributes::IntegerAttr,
        given_names::set_block_arg_name,
        op_interfaces::{
            BranchOpInterface, IsTerminatorInterface, NResultsInterface, NSuccsInterface,
            OneRegionInterface, OneSuccInterface, OperandSegmentInterface, RegionBranchOpInterface,
            RegionBranchPoint, RegionEdge, RegionSuccessor, SegmentNOfType,
            SingleBlockRegionInterface,
        },
        types::{IntegerType, Signedness},
    },
    common_traits::{Named, Verify},
    context::{Context, Ptr},
    derive::{op_interface_impl, pliron_op},
    ident,
    identifier::Identifier,
    input_err,
    irbuild::{
        IRStatus,
        inserter::{IRInserter, Inserter, OpInsertionPoint},
        listener::DummyListener,
        rewriter::Rewriter,
    },
    irfmt::{
        parsers::{
            block_opd_parser, delimited_list_parser, process_parsed_ssa_defs, spaced,
            ssa_opd_parser, type_parser,
        },
        printers::{iter_with_sep, list_with_sep},
    },
    linked_list::{ContainsLinkedList, LinkedList},
    location::Location,
    op::{Op, OpObj},
    operation::Operation,
    opts::constants::{BranchOpFoldInterface, RegionBranchOpFoldInterface},
    parsable::{IntoParseResult, Parsable},
    printable::{ListSeparator, Printable},
    region::Region,
    r#type::{TypeHandle, Typed},
    value::{Use, Value},
    verify_err,
};
use pliron::{
    builtin::op_interfaces::NRegionsInterface,
    combine::{
        Parser, attempt, optional,
        parser::char::{self, spaces},
    },
};

use pliron_llvm::attributes::{PoisonAttr, UndefAttr};

use crate::{
    cf::op_interfaces::{YieldingOp, YieldingRegions},
    index::{attributes::known_constant_index, types::IndexType},
};

#[derive(thiserror::Error, Debug)]
pub enum YieldOpVerifyErr {
    #[error("YieldOp operand types do not match parent operation result types")]
    OperandTypeMismatch,
    #[error("YieldOp must have a parent operation to verify against")]
    MissingParentOp,
}

/// Loop yield and termination operation.
/// Operands must match the results of the parent operation.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `results` | variadic of any type |
#[pliron_op(
    name = "cf.yield",
    interfaces = [NResultsInterface<0>, IsTerminatorInterface, YieldingOp],
    format = "operands(CharSpace(`,`))",
)]
pub struct YieldOp;

impl YieldOp {
    /// Creates a new `YieldOp` with the specified operands.
    pub fn new(ctx: &mut Context, results: Vec<Value>) -> Self {
        Operation::new_op::<Self>(ctx, vec![], results, vec![], 0)
    }
}

impl Verify for YieldOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let Some(parent_op) = self.get_operation().deref(ctx).get_parent_op(ctx) else {
            return verify_err!(self.loc(ctx), YieldOpVerifyErr::MissingParentOp);
        };

        let expected_types: Vec<_> = parent_op
            .deref(ctx)
            .results()
            .map(|r| r.get_type(ctx))
            .collect();
        let actual_types: Vec<_> = self
            .get_operation()
            .deref(ctx)
            .operands()
            .map(|o| o.get_type(ctx))
            .collect();

        if expected_types != actual_types {
            return verify_err!(self.loc(ctx), YieldOpVerifyErr::OperandTypeMismatch);
        }

        Ok(())
    }
}

/// Unconditional branch.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `dest_opds` | (variadic) values forwarded to `dest` as block arguments |
///
/// ## Successor(s)
/// | successor | description |
/// |-----|-------|
/// | `dest` | target block |
#[pliron_op(
    name = "cf.br",
    format = "succ($0) `(` operands(CharSpace(`,`)) `)`",
    interfaces = [
        IsTerminatorInterface,
        NResultsInterface<0>,
        NSuccsInterface<1>,
        OneSuccInterface,
    ],
    verifier = "succ",
)]
pub struct BrOp;

impl BrOp {
    /// Creates a new `BrOp` branching to `dest`, forwarding `dest_opds` as its block arguments.
    pub fn new(ctx: &mut Context, dest: Ptr<BasicBlock>, dest_opds: Vec<Value>) -> Self {
        Operation::new_op::<Self>(ctx, vec![], dest_opds, vec![dest], 0)
    }
}

#[op_interface_impl]
impl BranchOpInterface for BrOp {
    fn verify_successor_operand_layout(&self, ctx: &Context) -> pliron::result::Result<()> {
        <Self as OneSuccInterface>::verify(self, ctx)
    }

    fn successor_operand_range(&self, ctx: &Context, succ_idx: usize) -> Range<usize> {
        assert!(succ_idx == 0, "BrOp has exactly one successor");
        0..self.get_operation().deref(ctx).get_num_operands()
    }

    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        assert!(succ_idx == 0, "BrOp has exactly one successor");
        Operation::push_operand(self.get_operation(), ctx, operand)
    }

    fn remove_successor_operand(
        &self,
        ctx: &mut Context,
        succ_idx: usize,
        opd_idx: usize,
    ) -> Value {
        assert!(succ_idx == 0, "BrOp has exactly one successor");
        Operation::remove_operand(self.get_operation(), ctx, opd_idx)
    }
}

/// Conditional branch.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `condition` | signless 1-bit integer |
/// | `true_dest_opds` | (variadic) values forwarded to `true_dest` as block arguments |
/// | `false_dest_opds` | (variadic) values forwarded to `false_dest` as block arguments |
///
/// ## Successor(s)
/// | successor | description |
/// |-----|-------|
/// | `true_dest` | target block when `condition` is true |
/// | `false_dest` | target block when `condition` is false |
#[pliron_op(
    name = "cf.cond_br",
    interfaces = [
        IsTerminatorInterface,
        NResultsInterface<0>,
        NSuccsInterface<2>,
    ],
    operands = (condition, true_dest_opds, false_dest_opds),
)]
pub struct CondBrOp;

impl CondBrOp {
    /// Creates a new `CondBrOp`.
    pub fn new(
        ctx: &mut Context,
        condition: Value,
        true_dest: Ptr<BasicBlock>,
        true_dest_opds: Vec<Value>,
        false_dest: Ptr<BasicBlock>,
        false_dest_opds: Vec<Value>,
    ) -> Self {
        let (operands, segments) =
            Self::compute_segment_sizes(vec![vec![condition], true_dest_opds, false_dest_opds]);

        let op = Operation::new_op::<Self>(ctx, vec![], operands, vec![true_dest, false_dest], 0);
        op.set_operand_segment_sizes(ctx, segments);
        op
    }
}

#[derive(thiserror::Error, Debug)]
pub enum CondBrOpVerifyErr {
    #[error("CondBrOp condition operand must be a signless 1-bit integer (i1)")]
    IncorrectConditionType,
}

impl Verify for CondBrOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let condition_ty = self.get_operand_condition(ctx).get_type(ctx);
        let condition_ty = condition_ty.deref(ctx);
        let Some(condition_int_ty) = condition_ty.downcast_ref::<IntegerType>() else {
            return verify_err!(self.loc(ctx), CondBrOpVerifyErr::IncorrectConditionType);
        };
        if condition_int_ty.width() != 1 || condition_int_ty.signedness() != Signedness::Signless {
            return verify_err!(self.loc(ctx), CondBrOpVerifyErr::IncorrectConditionType);
        }
        Ok(())
    }
}

#[op_interface_impl]
impl OperandSegmentInterface for CondBrOp {
    fn expected_num_segments(&self, _ctx: &Context) -> Option<usize> {
        // The condition, the true destination operands and the false destination operands.
        Some(3)
    }
}

#[op_interface_impl]
impl BranchOpInterface for CondBrOp {
    fn verify_successor_operand_layout(&self, ctx: &Context) -> pliron::result::Result<()> {
        <Self as OperandSegmentInterface>::verify(self, ctx)?;
        <Self as NSuccsInterface<2>>::verify(self, ctx)
    }

    fn successor_operand_range(&self, ctx: &Context, succ_idx: usize) -> Range<usize> {
        assert!(
            succ_idx == 0 || succ_idx == 1,
            "CondBrOp has exactly two successors"
        );
        // Skip the first segment, which is the condition.
        self.segment_range(ctx, succ_idx + 1)
    }

    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        self.push_to_segment(ctx, succ_idx + 1, operand)
    }

    fn remove_successor_operand(
        &self,
        ctx: &mut Context,
        succ_idx: usize,
        opd_idx: usize,
    ) -> Value {
        self.remove_from_segment(ctx, succ_idx + 1, opd_idx)
    }
}

impl Printable for CondBrOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &pliron::printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let op = self.get_operation().deref(ctx);
        let condition = op.get_operand(0);
        let true_dest_opds = self.successor_operands(ctx, 0);
        let false_dest_opds = self.successor_operands(ctx, 1);
        write!(
            f,
            "{} if {} ^{}({}) else ^{}({})",
            Self::get_opid_static(),
            condition.print(ctx, state),
            op.get_successor(0).deref(ctx).unique_name(ctx),
            iter_with_sep(true_dest_opds.iter(), ListSeparator::CharSpace(',')).print(ctx, state),
            op.get_successor(1).deref(ctx).unique_name(ctx),
            iter_with_sep(false_dest_opds.iter(), ListSeparator::CharSpace(',')).print(ctx, state),
        )
    }
}

impl Parsable for CondBrOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut pliron::parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> pliron::parsable::ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                results[0].1.clone(),
                pliron::builtin::op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }

        let r#if = spaced(char::string("if"));
        let true_operands = delimited_list_parser('(', ')', ',', ssa_opd_parser());
        let r_else = spaced(char::string("else"));
        let false_operands = delimited_list_parser('(', ')', ',', ssa_opd_parser());

        let (((condition, true_dest), true_dest_opds), (false_dest, false_dest_opds)) = r#if
            .with(spaced(ssa_opd_parser()))
            .and(spaced(block_opd_parser()))
            .and(true_operands)
            .and(spaced(r_else).with(spaced(block_opd_parser()).and(false_operands)))
            .parse_stream(state_stream)
            .into_result()?
            .0;

        let op = CondBrOp::new(
            state_stream.state.ctx,
            condition,
            true_dest,
            true_dest_opds,
            false_dest,
            false_dest_opds,
        );

        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

/// Represents an `if`/`then`/`else` conditional.
///
/// If `condition` is true, the `then` region is executed; otherwise, the
/// `else` region is executed, if present.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `condition` | signless 1-bit integer |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `results` | (variadic) values produced by whichever region executes |
///
/// ## Region(s)
///   - `then`: A single-block region, taking no arguments, executed when
///   `condition` is true. Must `yield` this op's results.
///   - `else` (optional): Same structure as `then`, executed when `condition`
///   is false. Required whenever this op has results (so that a value is
///   produced along every path); optional otherwise.
///
///   Use [ExecuteRegionOp] to embed multi-block control flow in either region.
#[pliron_op(
    name = "cf.if",
    operands = (condition,),
    interfaces = [YieldingRegions<YieldOp>, SingleBlockRegionInterface],
)]
pub struct IfOp;

/// Type alias for the branch builder functions used in `IfOp::new`.
pub type IfOpBranchBuilderFn<State> =
    fn(ctx: &mut Context, state: State, inserter: &mut IRInserter<DummyListener>) -> Vec<Value>;

impl IfOp {
    /// Creates a new `IfOp`.
    ///
    /// `then_builder` populates the `then` region.
    ///   - It is provided with an inserter, set to the start of the region's entry block.
    ///   - It must return the values that this op should produce as results when
    ///     `condition` is true.
    ///
    /// `else_builder`, if provided, populates the `else` region analogously, for
    /// when `condition` is false. It must be provided whenever `result_types`
    /// is non-empty.
    ///
    /// A [YieldOp] is automatically appended at the end of each region, taking
    /// that region's builder's returned values as operands.
    pub fn new<ThenState, ElseState>(
        ctx: &mut Context,
        condition: Value,
        result_types: Vec<TypeHandle>,
        then_builder: IfOpBranchBuilderFn<ThenState>,
        then_builder_state: ThenState,
        else_builder: Option<(IfOpBranchBuilderFn<ElseState>, ElseState)>,
    ) -> Self {
        assert!(
            result_types.is_empty() || else_builder.is_some(),
            "IfOp with results must have an else branch"
        );

        let op = Operation::new_op::<Self>(ctx, result_types, vec![condition], vec![], 0);

        Self::build_branch_region(ctx, op.get_operation(), then_builder, then_builder_state);
        if let Some((else_builder, else_builder_state)) = else_builder {
            Self::build_branch_region(ctx, op.get_operation(), else_builder, else_builder_state);
        }

        op
    }

    /// Appends a new region to `op`, with a single entry block, populated by `builder`.
    fn build_branch_region<State>(
        ctx: &mut Context,
        op: Ptr<Operation>,
        builder: IfOpBranchBuilderFn<State>,
        builder_state: State,
    ) {
        let region = Operation::add_region(op, ctx);
        let entry_block = BasicBlock::new(ctx, Some(ident!("entry")), vec![]);
        entry_block.insert_at_front(region, ctx);

        let op_inserter = &mut IRInserter::new_at_block_start(entry_block);
        let yield_values = builder(ctx, builder_state, op_inserter);
        let yield_op = YieldOp::new(ctx, yield_values);
        op_inserter.set_insertion_point(OpInsertionPoint::AtBlockEnd(
            region
                .deref(ctx)
                .get_tail()
                .expect("Region must have at least one block"),
        ));
        op_inserter.insert_op(ctx, &yield_op);
    }

    /// Whether the `else` region is present.
    fn has_else_region(&self, ctx: &Context) -> bool {
        self.get_operation().deref(ctx).num_regions() > 1
    }

    /// Get the `then` region.
    pub fn get_then_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }

    /// Get the `else` region, if present.
    pub fn get_else_region(&self, ctx: &Context) -> Option<Ptr<Region>> {
        self.has_else_region(ctx)
            .then(|| self.get_operation().deref(ctx).get_region(1))
    }

    /// Get the entry block of the `then` region.
    pub fn get_then_entry(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_entry(ctx, 0)
    }

    /// Get the entry block of the `else` region, if present.
    pub fn get_else_entry(&self, ctx: &Context) -> Option<Ptr<BasicBlock>> {
        self.has_else_region(ctx).then(|| self.get_entry(ctx, 1))
    }

    /// Get the `yield` operation ending the `then` region.
    pub fn get_then_yield(&self, ctx: &Context) -> YieldOp {
        self.get_yield(ctx, 0)
    }

    /// Get the `yield` operation ending the `else` region, if present.
    pub fn get_else_yield(&self, ctx: &Context) -> Option<YieldOp> {
        self.has_else_region(ctx).then(|| self.get_yield(ctx, 1))
    }
}

impl Printable for IfOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &pliron::printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let op = self.get_operation().deref(ctx);
        if op.get_num_results() > 0 {
            let results = iter_with_sep(op.results(), ListSeparator::CharSpace(','));
            write!(f, "{} = ", results.print(ctx, state))?;
        }
        let result_types = op.results().map(|r| r.get_type(ctx)).collect::<Vec<_>>();
        write!(
            f,
            "{} {} -> ({}) {}",
            Self::get_opid_static(),
            self.get_operand_condition(ctx).print(ctx, state),
            list_with_sep(&result_types, ListSeparator::CharSpace(',')).print(ctx, state),
            self.get_then_region(ctx).print(ctx, state)
        )?;
        if let Some(else_region) = self.get_else_region(ctx) {
            write!(f, " else {}", else_region.print(ctx, state))?;
        }
        writeln!(f)
    }
}

impl Parsable for IfOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut pliron::parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> pliron::parsable::ParseResult<'a, Self::Parsed> {
        let (condition, _) = ssa_opd_parser().parse_stream(state_stream).into_result()?;

        let (result_types, _) = spaced(char::string("->"))
            .with(delimited_list_parser('(', ')', ',', type_parser()))
            .parse_stream(state_stream)
            .into_result()?;

        let opop = Operation::new_op::<Self>(
            state_stream.state.ctx,
            result_types,
            vec![condition],
            vec![],
            0,
        );
        let op = opop.get_operation();

        spaces()
            .with(Region::parser(op))
            .parse_stream(state_stream)
            .into_result()?;

        optional(attempt(
            spaced(char::string("else")).with(Region::parser(op)),
        ))
        .parse_stream(state_stream)
        .into_result()?;

        process_parsed_ssa_defs(state_stream, &results, op)?;
        Ok(OpObj::new(opop)).into_parse_result()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum IfOpVerifyErr {
    #[error("IfOp condition operand must be a signless 1-bit integer (i1)")]
    IncorrectConditionType,
    #[error("IfOp with results must have an else region")]
    MissingElseRegion,
}

impl Verify for IfOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let condition_ty = self.get_operand_condition(ctx).get_type(ctx);
        let condition_ty = condition_ty.deref(ctx);
        let Some(condition_int_ty) = condition_ty.downcast_ref::<IntegerType>() else {
            return verify_err!(self.loc(ctx), IfOpVerifyErr::IncorrectConditionType);
        };
        if condition_int_ty.width() != 1 || condition_int_ty.signedness() != Signedness::Signless {
            return verify_err!(self.loc(ctx), IfOpVerifyErr::IncorrectConditionType);
        }
        Ok(())
    }
}

/// Represents a `for` loop with an initial value, an upper bound, and a step.
/// Additional loop-carried variables can be specified as operands and results.
/// The loop body is defined in a region that takes the loop induction variable
/// and loop-carried variables as arguments and yields updated loop-carried variables.
///
/// The trip count `n` is `max(0, ceil((upper_bound - lower_bound) / step))`,
/// in arbitrary precision arithmetic. For a well-defined `ForOp`:
/// - The step must be strictly positive.
/// - `lower_bound + n * step` must be representable in the index type.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lower_bound` | The starting index of the loop (inclusive). Must be of index type. |
/// | `upper_bound` | The ending index of the loop (exclusive). Must be of index type. |
/// | `step` | The step size for each iteration. Must be of index type. |
/// | `iter_args_init` | (variadic) Initial values for additional loop-carried variables. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `iter_args_res` | (variadic) Updated loop-carried variables after loop completion.
///
/// ## Region(s)
///   - A single region, with a single block, containing the loop body.
///   The block takes as arguments the loop induction variable followed by
///   the loop-carried variables, and must `yield` the updated loop-carried
///   variables at the end of each iteration.
///   Use [ExecuteRegionOp] to embed multi-block control flow in the loop body.
#[pliron_op(
    name = "cf.for",
    interfaces = [
        OneRegionInterface,
        NRegionsInterface<1>,
        SegmentNOfType<0, IndexType>,
        YieldingRegions<YieldOp>,
        SingleBlockRegionInterface
    ],
)]
pub struct ForOp;

#[op_interface_impl]
impl OperandSegmentInterface for ForOp {
    fn expected_num_segments(&self, _ctx: &Context) -> Option<usize> {
        // The lower bound, upper bound and step, and the initial loop-carried values.
        Some(2)
    }
}

/// Type alias for the body builder function used in `ForOp::new`.
pub type ForOpBodyBuilderFn<State> = fn(
    ctx: &mut Context,
    state: State,
    inserter: &mut IRInserter<DummyListener>,
    idx: Value,
    iter_args: &[Value],
) -> Vec<Value>;

impl ForOp {
    /// Creates a new `ForOp` with the specified bounds, step and initial loop carried variables.
    ///
    /// The `body_builder` function is called to populate the body of the region.
    ///   - This function is provided with the current index value and loop carried variables as arguments.
    ///   - It is also provided with an inserter, set to the start of the entry block.
    ///   - It must return the updated / result loop carried variables of an iteration;
    ///
    /// A [YieldOp] is automatically added at end of the body, taking these results as operands.
    pub fn new<State>(
        ctx: &mut Context,
        lower_bound: Value,
        upper_bound: Value,
        step: Value,
        iter_args_init: &[Value],
        body_builder: ForOpBodyBuilderFn<State>,
        body_builder_state: State,
    ) -> Self {
        let index_ty = IndexType::get(ctx);
        let result_types = iter_args_init
            .iter()
            .map(|v| v.get_type(ctx))
            .collect::<Vec<_>>();
        let region_arg_types = std::iter::once(index_ty.into())
            .chain(result_types.iter().cloned())
            .collect();
        let region_arg_names = std::iter::once(ident!("iv"))
            .chain(iter_args_init.iter().enumerate().map(|(lv_i, v)| {
                v.given_name(ctx)
                    .unwrap_or(format!("loop_var_{}", lv_i).try_into().unwrap())
            }))
            .collect::<Vec<Identifier>>();

        let (operands, segments) = Self::compute_segment_sizes(vec![
            vec![lower_bound, upper_bound, step],
            iter_args_init.to_vec(),
        ]);

        let op = Operation::new_op::<Self>(ctx, result_types, operands, vec![], 1);
        op.set_operand_segment_sizes(ctx, segments);

        // Set up the region, its entry block and arguments.
        let region = op.get_region(ctx);
        let entry_block = BasicBlock::new(ctx, Some(ident!("entry")), region_arg_types);
        entry_block.insert_at_front(region, ctx);
        for (arg_idx, name) in region_arg_names.into_iter().enumerate() {
            set_block_arg_name(ctx, entry_block, arg_idx, Some(name));
        }

        // Populate the body.
        let entry_block_args = entry_block.deref(ctx).arguments().collect::<Vec<_>>();
        let idx = entry_block_args[0];
        let iter_args = &entry_block_args[1..];
        let op_inserter = &mut IRInserter::new_at_block_start(entry_block);
        let yield_values = body_builder(ctx, body_builder_state, op_inserter, idx, iter_args);
        let yield_op = YieldOp::new(ctx, yield_values);
        op_inserter.set_insertion_point(OpInsertionPoint::AtBlockEnd(
            region
                .deref(ctx)
                .get_tail()
                .expect("Region must have at least one block"),
        ));
        op_inserter.insert_op(ctx, &yield_op);
        op
    }

    /// Get the lower bound operand.
    pub fn get_lower_bound(&self, ctx: &Context) -> Value {
        let loop_inputs = self.get_segment(ctx, 0);
        loop_inputs[0]
    }

    /// Get the upper bound operand.
    pub fn get_upper_bound(&self, ctx: &Context) -> Value {
        let loop_inputs = self.get_segment(ctx, 0);
        loop_inputs[1]
    }

    /// Get the step operand.
    pub fn get_step(&self, ctx: &Context) -> Value {
        let loop_inputs = self.get_segment(ctx, 0);
        loop_inputs[2]
    }

    /// Get initial values of the loop carried variables.
    pub fn get_iter_args_init(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }

    /// Get number of loop carried variables initializers
    pub fn get_num_iter_arg_inits(&self, ctx: &Context) -> u32 {
        self.segment_size(ctx, 1)
    }

    /// Get the induction variable of the loop.
    pub fn get_induction_variable(&self, ctx: &Context) -> Value {
        self.get_entry(ctx, 0).deref(ctx).get_argument(0)
    }

    /// Get the loop carried variables (block arguments).
    pub fn get_loop_carried_variables(&self, ctx: &Context) -> Vec<Value> {
        self.get_entry(ctx, 0)
            .deref(ctx)
            .arguments()
            .skip(1)
            .collect::<Vec<_>>()
    }
}

impl Printable for ForOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &pliron::printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let op = self.get_operation().deref(ctx);
        if op.get_num_results() > 0 {
            let results = iter_with_sep(op.results(), ListSeparator::CharSpace(','));
            write!(f, "{} = ", results.print(ctx, state))?;
        }
        let iter_args_init = self.get_iter_args_init(ctx);
        writeln!(
            f,
            "{} {} to {} step {} ({}) {}",
            Self::get_opid_static(),
            self.get_lower_bound(ctx).print(ctx, state),
            self.get_upper_bound(ctx).print(ctx, state),
            self.get_step(ctx).print(ctx, state),
            list_with_sep(&iter_args_init, ListSeparator::CharSpace(',')).print(ctx, state),
            self.get_region(ctx).print(ctx, state)
        )
    }
}

impl Parsable for ForOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut pliron::parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> pliron::parsable::ParseResult<'a, Self::Parsed> {
        let (lb, ub, step) = (
            ssa_opd_parser().skip(spaced(char::string("to"))),
            ssa_opd_parser().skip(spaced(char::string("step"))),
            ssa_opd_parser().skip(spaces()),
        );
        let iter_args_init = delimited_list_parser('(', ')', ',', ssa_opd_parser());

        let ((lb, ub, step, iter_args_init), _) = (lb, ub, step, iter_args_init)
            .parse_stream(state_stream)
            .into_result()?;

        let result_types = iter_args_init
            .iter()
            .map(|v| v.get_type(state_stream.state.ctx))
            .collect::<Vec<_>>();

        let (operands, segments) =
            Self::compute_segment_sizes(vec![vec![lb, ub, step], iter_args_init]);

        let opop =
            Operation::new_op::<Self>(state_stream.state.ctx, result_types, operands, vec![], 0);
        let op = opop.get_operation();
        opop.set_operand_segment_sizes(state_stream.state.ctx, segments);

        spaces()
            .with(Region::parser(op))
            .parse_stream(state_stream)
            .into_result()?;

        process_parsed_ssa_defs(state_stream, &results, op)?;
        Ok(OpObj::new(opop)).into_parse_result()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ForOpVerifyErr {
    #[error(
        "ForOp count mismatch: iter args initializers, number of results, loop carried variables, yielded values"
    )]
    IterArgsCountMismatch,
    #[error("ForOp induction variable type must be IndexType")]
    InductionVarTypeMismatch,
}

impl Verify for ForOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let iv_ty = self.get_induction_variable(ctx).get_type(ctx);
        if iv_ty.deref(ctx).downcast_ref::<IndexType>().is_none() {
            return verify_err!(self.loc(ctx), ForOpVerifyErr::InductionVarTypeMismatch);
        }

        Ok(())
    }
}

/// Type alias for the body builder function used in `NDForOp::new`.
pub type NDForOpBodyBuilderFn<State> = fn(
    ctx: &mut Context,
    state: State,
    inserter: &mut IRInserter<DummyListener>,
    indices: Vec<Value>,
);

/// An N-Dimensional version of [ForOp],
/// but without the loop carried variables and related operands/results.
///
/// For each dimension `i`, the trip count `n[i]` is
/// `max(0, ceil((upper_bounds[i] - lower_bounds[i]) / steps[i]))`,
/// in arbitrary precision arithmetic. For a well-defined `NDForOp`:
/// - Each step must be strictly positive.
/// - `lower_bounds[i] + n[i] * steps[i]` must be representable in the index type.
///
/// The body of this op is a region with N entry-block arguments
/// corresponding to the N induction variables. The region of this op
/// takes the induction variables as arguments, and yields zero results.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lower_bounds` | The starting indices of the loop (inclusive). Must be of index type. |
/// | `upper_bounds` | The ending indices of the loop (exclusive). Must be of index type. |
/// | `steps` | The step sizes for each iteration. Must be of index type. |
///
/// ## Region(s)
///   - A single region, with a single block, containing the loop body.
///   The block takes as arguments the loop induction variables, and must
///   end with a [YieldOp] with no operands.
///   Use [ExecuteRegionOp] to embed multi-block control flow in the loop body.
#[pliron_op(
    name = "cf.nd_for",
    interfaces = [
        OneRegionInterface,
        NRegionsInterface<1>,
        SegmentNOfType<0, IndexType>,
        SegmentNOfType<1, IndexType>,
        SegmentNOfType<2, IndexType>,
        YieldingRegions<YieldOp>,
        SingleBlockRegionInterface
    ],
)]
pub struct NDForOp;

#[op_interface_impl]
impl OperandSegmentInterface for NDForOp {
    fn expected_num_segments(&self, _ctx: &Context) -> Option<usize> {
        // The lower bounds, the upper bounds and the steps.
        Some(3)
    }
}

impl Printable for NDForOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &pliron::printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let lower_bounds = self.get_lower_bounds(ctx);
        let upper_bounds = self.get_upper_bounds(ctx);
        let steps = self.get_steps(ctx);

        writeln!(
            f,
            "{} [{}] to [{}] step [{}] {}",
            Self::get_opid_static(),
            list_with_sep(&lower_bounds, ListSeparator::CharSpace(',')).print(ctx, state),
            list_with_sep(&upper_bounds, ListSeparator::CharSpace(',')).print(ctx, state),
            list_with_sep(&steps, ListSeparator::CharSpace(',')).print(ctx, state),
            self.get_region(ctx).print(ctx, state)
        )
    }
}

#[derive(thiserror::Error, Debug)]
pub enum NDForOpParseErr {
    #[error("NDForOp must not have any results")]
    NoResultsAllowed,
}

impl Parsable for NDForOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut pliron::parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> pliron::parsable::ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(results[0].1.clone(), NDForOpParseErr::NoResultsAllowed)?;
        }

        let (lbs, ubs, steps) = (
            delimited_list_parser('[', ']', ',', ssa_opd_parser()).skip(spaced(char::string("to"))),
            delimited_list_parser('[', ']', ',', ssa_opd_parser())
                .skip(spaced(char::string("step"))),
            delimited_list_parser('[', ']', ',', ssa_opd_parser()),
        );

        let ((lbs, ubs, steps), _) = (lbs, ubs, steps).parse_stream(state_stream).into_result()?;
        let (operands, segments) = Self::compute_segment_sizes(vec![lbs, ubs, steps]);

        let opop = Operation::new_op::<Self>(state_stream.state.ctx, vec![], operands, vec![], 0);
        let op = opop.get_operation();

        opop.set_operand_segment_sizes(state_stream.state.ctx, segments);

        spaces()
            .with(Region::parser(op))
            .parse_stream(state_stream)
            .into_result()?;

        process_parsed_ssa_defs(state_stream, &results, op)?;
        Ok(OpObj::new(opop)).into_parse_result()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum NDForOpVerifyErr {
    #[error(
        "NDForOp must have exactly 3 segments corresponding to lower bounds, upper bounds, and steps"
    )]
    SegmentCountMismatch,
    #[error(
        "NDForOp count mismatch: lower bounds, upper bounds, and steps must all have the same number of operands"
    )]
    IterArgsCountMismatch,
}

impl Verify for NDForOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        if self.num_segments(ctx) != 3 {
            return verify_err!(self.loc(ctx), NDForOpVerifyErr::SegmentCountMismatch);
        }

        if self.segment_size(ctx, 0) == 0
            || self.segment_size(ctx, 0) != self.segment_size(ctx, 1)
            || self.segment_size(ctx, 0) != self.segment_size(ctx, 2)
        {
            return verify_err!(self.loc(ctx), NDForOpVerifyErr::IterArgsCountMismatch);
        }

        Ok(())
    }
}

impl NDForOp {
    /// Get the lower bound operands.
    pub fn get_lower_bounds(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 0)
    }

    /// Get the upper bound operands.
    pub fn get_upper_bounds(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }

    /// Get the step operands.
    pub fn get_steps(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 2)
    }

    /// Get the induction variables of the loop.
    pub fn get_induction_variables(&self, ctx: &Context) -> Vec<Value> {
        self.get_entry(ctx, 0)
            .deref(ctx)
            .arguments()
            .collect::<Vec<_>>()
    }

    /// Get the number of induction variables (i.e., loop dimensions).
    pub fn get_num_induction_variables(&self, ctx: &Context) -> usize {
        self.get_entry(ctx, 0).deref(ctx).get_num_arguments()
    }

    /// Create a new `NDForOp` with the specified bounds and steps.
    ///
    /// The body of the loop is populated using the provided `body_builder` function,
    /// which is called with the current induction variable values and an
    /// inserter set to the start of the entry block.
    ///
    /// A [YieldOp] is automatically added at the end of the exit block.
    pub fn new<State>(
        ctx: &mut Context,
        lower_bounds: Vec<Value>,
        upper_bounds: Vec<Value>,
        steps: Vec<Value>,
        body_builder: NDForOpBodyBuilderFn<State>,
        body_builder_state: State,
    ) -> Self {
        let num_dims = lower_bounds.len();
        assert_eq!(upper_bounds.len(), num_dims);
        assert_eq!(steps.len(), num_dims);

        let region_arg_types = lower_bounds
            .iter()
            .map(|_| IndexType::get(ctx).into())
            .collect::<Vec<_>>();

        let (operands, segments) =
            Self::compute_segment_sizes(vec![lower_bounds, upper_bounds, steps]);

        let op = Operation::new_op::<Self>(ctx, vec![], operands, vec![], 1);
        op.set_operand_segment_sizes(ctx, segments);

        // Set up the region and its entry block.
        let region = op.get_region(ctx);
        let entry_block = BasicBlock::new(ctx, Some(ident!("entry")), region_arg_types);
        entry_block.insert_at_front(region, ctx);

        // Populate the body.
        let entry_block_args = entry_block.deref(ctx).arguments().collect::<Vec<_>>();
        let op_inserter = &mut IRInserter::new_at_block_start(entry_block);
        body_builder(ctx, body_builder_state, op_inserter, entry_block_args);
        let yield_op = YieldOp::new(ctx, vec![]);
        op_inserter.set_insertion_point(OpInsertionPoint::AtBlockEnd(
            region
                .deref(ctx)
                .get_tail()
                .expect("Region must have at least one block"),
        ));
        op_inserter.insert_op(ctx, &yield_op);
        op
    }
}

/// Executes a region containing arbitrary (possibly multi-block) control flow, once, inline.
///
/// This enables embedding multi-block control flow inside [SingleBlockRegionInterface]
/// Ops (like [ForOp]).
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `results` | (variadic) values produced by the region, as `yield`ed by its exit block |
///
/// ## Region(s)
///   - A single region, taking no arguments, which may contain multiple blocks.
///   The exit (lexicographically last) block must `yield` the results of this op.
#[pliron_op(
    name = "cf.execute_region",
    interfaces = [OneRegionInterface, NRegionsInterface<1>, YieldingRegions<YieldOp>],
)]
pub struct ExecuteRegionOp;

/// Type alias for the body builder function used in `ExecuteRegionOp::new`.
pub type ExecuteRegionOpBodyBuilderFn<State> =
    fn(ctx: &mut Context, state: State, inserter: &mut IRInserter<DummyListener>) -> Vec<Value>;

impl ExecuteRegionOp {
    /// Creates a new `ExecuteRegionOp` with the specified result types.
    ///
    /// The `body_builder` function is called to populate the body of the region.
    ///   - It is provided with an inserter, set to the start of the entry block.
    ///   - It must return the values that this op should produce as results.
    ///
    /// A [YieldOp] is automatically added at the end of the body, taking these results
    /// as operands.
    pub fn new<State>(
        ctx: &mut Context,
        result_types: Vec<TypeHandle>,
        body_builder: ExecuteRegionOpBodyBuilderFn<State>,
        body_builder_state: State,
    ) -> Self {
        let op = Operation::new_op::<Self>(ctx, result_types, vec![], vec![], 1);

        // Set up the region and its entry block.
        let region = op.get_region(ctx);
        let entry_block = BasicBlock::new(ctx, Some(ident!("entry")), vec![]);
        entry_block.insert_at_front(region, ctx);

        // Populate the body.
        let op_inserter = &mut IRInserter::new_at_block_start(entry_block);
        let yield_values = body_builder(ctx, body_builder_state, op_inserter);
        let yield_op = YieldOp::new(ctx, yield_values);
        op_inserter.set_insertion_point(OpInsertionPoint::AtBlockEnd(
            region
                .deref(ctx)
                .get_tail()
                .expect("Region must have at least one block"),
        ));
        op_inserter.insert_op(ctx, &yield_op);
        op
    }
}

impl Printable for ExecuteRegionOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &pliron::printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let op = self.get_operation().deref(ctx);
        if op.get_num_results() > 0 {
            let results = iter_with_sep(op.results(), ListSeparator::CharSpace(','));
            write!(f, "{} = ", results.print(ctx, state))?;
        }
        let result_types = op.results().map(|r| r.get_type(ctx)).collect::<Vec<_>>();
        writeln!(
            f,
            "{} -> ({}) {}",
            Self::get_opid_static(),
            list_with_sep(&result_types, ListSeparator::CharSpace(',')).print(ctx, state),
            self.get_region(ctx).print(ctx, state)
        )
    }
}

impl Parsable for ExecuteRegionOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut pliron::parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> pliron::parsable::ParseResult<'a, Self::Parsed> {
        let (result_types, _) = spaced(char::string("->"))
            .with(delimited_list_parser('(', ')', ',', type_parser()))
            .parse_stream(state_stream)
            .into_result()?;

        let opop =
            Operation::new_op::<Self>(state_stream.state.ctx, result_types, vec![], vec![], 0);
        let op = opop.get_operation();

        spaces()
            .with(Region::parser(op))
            .parse_stream(state_stream)
            .into_result()?;

        process_parsed_ssa_defs(state_stream, &results, op)?;
        Ok(OpObj::new(opop)).into_parse_result()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ExecuteRegionOpVerifyErr {
    #[error("ExecuteRegionOp's region must not take any arguments")]
    RegionHasArguments,
}

impl Verify for ExecuteRegionOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        if self.get_entry(ctx, 0).deref(ctx).get_num_arguments() != 0 {
            return verify_err!(self.loc(ctx), ExecuteRegionOpVerifyErr::RegionHasArguments);
        }
        Ok(())
    }
}

/// Pair operands of `source`, starting at operand `start`, with `receivers`.
fn flows_from(
    ctx: &Context,
    source: Ptr<Operation>,
    start: usize,
    receivers: impl IntoIterator<Item = Value>,
) -> Vec<(Use<Value>, Value)> {
    receivers
        .into_iter()
        .enumerate()
        .map(|(i, receiver)| (source.deref(ctx).get_operand_as_use(start + i), receiver))
        .collect()
}

#[op_interface_impl]
impl RegionBranchOpInterface for IfOp {
    fn verify_region_edges_layout(&self, ctx: &Context) -> pliron::result::Result<()> {
        // IfOp with results must have an else region, so that a value is produced along every path.
        if self.get_else_region(ctx).is_none()
            && self.get_operation().deref(ctx).get_num_results() > 0
        {
            return verify_err!(self.loc(ctx), IfOpVerifyErr::MissingElseRegion);
        }
        <Self as YieldingRegions<YieldOp>>::verify(self, ctx)
    }

    fn region_edges(&self, ctx: &Context) -> Vec<RegionEdge> {
        let op = self.get_operation();
        let num_regions = op.deref(ctx).num_regions();
        let mut edges: Vec<RegionEdge> = op
            .deref(ctx)
            .regions()
            .map(|region| RegionEdge {
                from: RegionBranchPoint::Parent,
                to: RegionSuccessor::Region(region),
                flows: vec![],
                reads: vec![],
            })
            .collect();
        if !self.has_else_region(ctx) {
            edges.push(RegionEdge {
                from: RegionBranchPoint::Parent,
                to: RegionSuccessor::After,
                flows: vec![],
                reads: vec![],
            });
        }
        for reg_idx in 0..num_regions {
            let yield_op = self.get_yield(ctx, reg_idx).get_operation();
            edges.push(RegionEdge {
                from: RegionBranchPoint::Terminator(yield_op),
                to: RegionSuccessor::After,
                flows: flows_from(ctx, yield_op, 0, op.deref(ctx).results()),
                reads: vec![],
            });
        }
        edges
    }
}

#[op_interface_impl]
impl RegionBranchOpInterface for ForOp {
    fn verify_region_edges_layout(&self, ctx: &Context) -> pliron::result::Result<()> {
        <Self as NRegionsInterface<1>>::verify(self, ctx)?;
        <Self as YieldingRegions<YieldOp>>::verify(self, ctx)?;
        <Self as OperandSegmentInterface>::verify(self, ctx)?;
        let num_results = self.get_operation().deref(ctx).get_num_results();
        let num_yielded = self
            .get_yield(ctx, 0)
            .get_operation()
            .deref(ctx)
            .get_num_operands();
        if self.get_iter_args_init(ctx).len() != num_results
            || self.get_loop_carried_variables(ctx).len() != num_results
            || num_yielded != num_results
        {
            return verify_err!(self.loc(ctx), ForOpVerifyErr::IterArgsCountMismatch);
        }
        Ok(())
    }

    fn region_edges(&self, ctx: &Context) -> Vec<RegionEdge> {
        let op = self.get_operation();
        let region = op.deref(ctx).get_region(0);
        let yield_op = self.get_yield(ctx, 0).get_operation();
        let init_start = self.segment_range(ctx, 1).start;
        let loop_carried = self.get_loop_carried_variables(ctx);
        let results: Vec<_> = op.deref(ctx).results().collect();
        // The loop reads its upper bound and step again after each iteration.
        let start = self.segment_range(ctx, 0).start;
        let iteration_reads = vec![
            op.deref(ctx).get_operand_as_use(start + 1),
            op.deref(ctx).get_operand_as_use(start + 2),
        ];
        vec![
            RegionEdge {
                from: RegionBranchPoint::Parent,
                to: RegionSuccessor::Region(region),
                flows: flows_from(ctx, op, init_start, loop_carried.clone()),
                reads: vec![],
            },
            RegionEdge {
                from: RegionBranchPoint::Parent,
                to: RegionSuccessor::After,
                flows: flows_from(ctx, op, init_start, results.clone()),
                reads: vec![],
            },
            RegionEdge {
                from: RegionBranchPoint::Terminator(yield_op),
                to: RegionSuccessor::Region(region),
                flows: flows_from(ctx, yield_op, 0, loop_carried),
                reads: iteration_reads.clone(),
            },
            RegionEdge {
                from: RegionBranchPoint::Terminator(yield_op),
                to: RegionSuccessor::After,
                flows: flows_from(ctx, yield_op, 0, results),
                reads: iteration_reads,
            },
        ]
    }
}

#[op_interface_impl]
impl RegionBranchOpInterface for NDForOp {
    fn verify_region_edges_layout(&self, ctx: &Context) -> pliron::result::Result<()> {
        <Self as NRegionsInterface<1>>::verify(self, ctx)?;
        <Self as YieldingRegions<YieldOp>>::verify(self, ctx)
    }

    fn region_edges(&self, ctx: &Context) -> Vec<RegionEdge> {
        let op = self.get_operation();
        let region = op.deref(ctx).get_region(0);
        let yield_op = self.get_yield(ctx, 0).get_operation();
        // The loop reads its upper bounds and steps again after each iteration.
        let iteration_reads: Vec<_> = self
            .segment_range(ctx, 1)
            .chain(self.segment_range(ctx, 2))
            .map(|idx| op.deref(ctx).get_operand_as_use(idx))
            .collect();
        let edge = |from, to, reads| RegionEdge {
            from,
            to,
            flows: vec![],
            reads,
        };
        vec![
            edge(
                RegionBranchPoint::Parent,
                RegionSuccessor::Region(region),
                vec![],
            ),
            edge(RegionBranchPoint::Parent, RegionSuccessor::After, vec![]),
            edge(
                RegionBranchPoint::Terminator(yield_op),
                RegionSuccessor::Region(region),
                iteration_reads.clone(),
            ),
            edge(
                RegionBranchPoint::Terminator(yield_op),
                RegionSuccessor::After,
                iteration_reads,
            ),
        ]
    }
}

#[op_interface_impl]
impl RegionBranchOpInterface for ExecuteRegionOp {
    fn verify_region_edges_layout(&self, ctx: &Context) -> pliron::result::Result<()> {
        <Self as NRegionsInterface<1>>::verify(self, ctx)?;
        <Self as YieldingRegions<YieldOp>>::verify(self, ctx)
    }

    fn region_edges(&self, ctx: &Context) -> Vec<RegionEdge> {
        let op = self.get_operation();
        let yield_op = self.get_yield(ctx, 0).get_operation();
        vec![
            RegionEdge {
                from: RegionBranchPoint::Parent,
                to: RegionSuccessor::Region(op.deref(ctx).get_region(0)),
                flows: vec![],
                reads: vec![],
            },
            RegionEdge {
                from: RegionBranchPoint::Terminator(yield_op),
                to: RegionSuccessor::After,
                flows: flows_from(ctx, yield_op, 0, op.deref(ctx).results()),
                reads: vec![],
            },
        ]
    }
}

impl_no_self_side_effects!(
    YieldOp,
    BrOp,
    CondBrOp,
    IfOp,
    ExecuteRegionOp,
    ForOp,
    NDForOp
);

/// For a loop over `[lb, ub)`, returns
/// - `Some(true)` if the loop has zero iterations
/// - `Some(false)` if the loop has one or more iterations
/// - `None` if `lb` or `ub` is not known
fn is_zero_trip_bounds(lb: Option<u64>, ub: Option<u64>) -> Option<bool> {
    Some(lb? >= ub?)
}

/// Possible successors of a loop's entry, given whether it `is_zero_trip`.
fn loop_entry_successors(region: Ptr<Region>, is_zero_trip: Option<bool>) -> Vec<RegionSuccessor> {
    match is_zero_trip {
        Some(true) => vec![RegionSuccessor::After],
        Some(false) => vec![RegionSuccessor::Region(region)],
        None => vec![RegionSuccessor::Region(region), RegionSuccessor::After],
    }
}

/// The boolean value of the first operand, if it is a known constant.
fn known_condition(operand_attrs: &[Option<AttrObj>]) -> Option<bool> {
    let condition = operand_attrs
        .first()?
        .as_ref()?
        .downcast_ref::<IntegerAttr>()?;
    Some(!condition.value().is_zero())
}

#[op_interface_impl]
impl RegionBranchOpFoldInterface for IfOp {
    fn check_fold(
        &self,
        ctx: &Context,
        point: RegionBranchPoint,
        operand_attrs: &[Option<AttrObj>],
        _terminator_operand_attrs: &[Option<AttrObj>],
    ) -> Vec<RegionSuccessor> {
        if let RegionBranchPoint::Terminator(_) = point {
            return vec![RegionSuccessor::After];
        }
        match known_condition(operand_attrs) {
            Some(true) => vec![RegionSuccessor::Region(self.get_then_region(ctx))],
            Some(false) => vec![
                self.get_else_region(ctx)
                    .map_or(RegionSuccessor::After, RegionSuccessor::Region),
            ],
            None => self
                .edges_from(ctx, RegionBranchPoint::Parent)
                .into_iter()
                .map(|edge| edge.to)
                .collect(),
        }
    }

    // Replace this op with the operations of the region that the
    // condition selects, or erase this op if no region is selected.
    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        let Some(condition) = known_condition(operand_attrs) else {
            return IRStatus::Unchanged;
        };
        let op = self.get_operation();
        let reg_idx = if condition { 0 } else { 1 };
        if reg_idx >= op.deref(ctx).num_regions() {
            rewriter.erase_operation(ctx, op);
            return IRStatus::Changed;
        }

        let yield_op = self.get_yield(ctx, reg_idx).get_operation();
        let yielded: Vec<Value> = yield_op.deref(ctx).operands().collect();
        let mut cur = self.get_entry(ctx, reg_idx).deref(ctx).get_head();
        while let Some(inner) = cur
            && inner != yield_op
        {
            cur = inner.deref(ctx).get_next();
            rewriter.move_operation(ctx, inner, OpInsertionPoint::BeforeOperation(op));
        }
        rewriter.replace_operation_with_values(ctx, op, yielded);
        IRStatus::Changed
    }
}

impl ForOp {
    /// Given the constant values of the operands in `operand_attrs`, returns
    /// - `Some(true)` if the loop has zero iterations
    /// - `Some(false)` if the loop has one or more iterations
    /// - `None` if this is not known
    fn is_zero_trip(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Option<bool> {
        let start = self.segment_range(ctx, 0).start;
        is_zero_trip_bounds(
            known_constant_index(&operand_attrs[start]),
            known_constant_index(&operand_attrs[start + 1]),
        )
    }
}

#[op_interface_impl]
impl RegionBranchOpFoldInterface for ForOp {
    fn check_fold(
        &self,
        ctx: &Context,
        point: RegionBranchPoint,
        operand_attrs: &[Option<AttrObj>],
        _terminator_operand_attrs: &[Option<AttrObj>],
    ) -> Vec<RegionSuccessor> {
        let region = self.get_region(ctx);
        match point {
            RegionBranchPoint::Parent => {
                loop_entry_successors(region, self.is_zero_trip(ctx, operand_attrs))
            }
            RegionBranchPoint::Terminator(_) => {
                vec![RegionSuccessor::Region(region), RegionSuccessor::After]
            }
        }
    }

    // Replace a loop that runs zero iterations with its initial loop-carried values.
    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        if self.is_zero_trip(ctx, operand_attrs) != Some(true) {
            return IRStatus::Unchanged;
        }
        let iter_args_init = self.get_iter_args_init(ctx);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), iter_args_init);
        IRStatus::Changed
    }
}

impl NDForOp {
    /// Given the constant values of the operands in `operand_attrs`, returns
    /// - `Some(true)` if the loop has zero iterations
    /// - `Some(false)` if the loop has one or more iterations
    /// - `None` if this is not known
    fn is_zero_trip(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Option<bool> {
        let dims_zero_trip: Vec<_> = self
            .segment_range(ctx, 0)
            .zip(self.segment_range(ctx, 1))
            .map(|(lb, ub)| {
                is_zero_trip_bounds(
                    known_constant_index(&operand_attrs[lb]),
                    known_constant_index(&operand_attrs[ub]),
                )
            })
            .collect();
        if dims_zero_trip.contains(&Some(true)) {
            Some(true)
        } else if dims_zero_trip
            .iter()
            .all(|zero_trip| *zero_trip == Some(false))
        {
            Some(false)
        } else {
            None
        }
    }
}

#[op_interface_impl]
impl RegionBranchOpFoldInterface for NDForOp {
    fn check_fold(
        &self,
        ctx: &Context,
        point: RegionBranchPoint,
        operand_attrs: &[Option<AttrObj>],
        _terminator_operand_attrs: &[Option<AttrObj>],
    ) -> Vec<RegionSuccessor> {
        let region = self.get_region(ctx);
        match point {
            RegionBranchPoint::Parent => {
                loop_entry_successors(region, self.is_zero_trip(ctx, operand_attrs))
            }
            RegionBranchPoint::Terminator(_) => {
                vec![RegionSuccessor::Region(region), RegionSuccessor::After]
            }
        }
    }

    // Erase a loop that runs zero iterations.
    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        if self.is_zero_trip(ctx, operand_attrs) != Some(true) {
            return IRStatus::Unchanged;
        }
        rewriter.erase_operation(ctx, self.get_operation());
        IRStatus::Changed
    }
}

#[op_interface_impl]
impl RegionBranchOpFoldInterface for ExecuteRegionOp {
    fn check_fold(
        &self,
        ctx: &Context,
        point: RegionBranchPoint,
        _operand_attrs: &[Option<AttrObj>],
        _terminator_operand_attrs: &[Option<AttrObj>],
    ) -> Vec<RegionSuccessor> {
        self.edges_from(ctx, point)
            .into_iter()
            .map(|edge| edge.to)
            .collect()
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
impl BranchOpFoldInterface for BrOp {
    fn check_fold(
        &self,
        ctx: &Context,
        _operand_attrs: &[Option<AttrObj>],
    ) -> Vec<Ptr<BasicBlock>> {
        self.get_operation().deref(ctx).successors().collect()
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

/// Is `attr` an `undef` or a `poison` value?
fn is_undef_or_poison_attr(attr: &dyn Attribute) -> bool {
    attr.downcast_ref::<UndefAttr>().is_some() || attr.downcast_ref::<PoisonAttr>().is_some()
}

impl CondBrOp {
    /// Get the indices of possible successors, given `operand_attrs`.
    ///
    /// A branch on an `undef` or `poison` condition is undefined and leads to no successor.
    fn possible_successor_indices(operand_attrs: &[Option<AttrObj>]) -> Vec<usize> {
        let condition = operand_attrs.first().and_then(|cond| cond.as_ref());
        if condition.is_some_and(|cond| is_undef_or_poison_attr(&**cond)) {
            return vec![];
        }
        match known_condition(operand_attrs) {
            Some(true) => vec![0],
            Some(false) => vec![1],
            None => vec![0, 1],
        }
    }
}

#[op_interface_impl]
impl BranchOpFoldInterface for CondBrOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Ptr<BasicBlock>> {
        let op = self.get_operation().deref(ctx);
        Self::possible_successor_indices(operand_attrs)
            .into_iter()
            .map(|succ_idx| op.get_successor(succ_idx))
            .collect()
    }

    // If we can determine a single successor, replace this op with a [BrOp] to it.
    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        let [succ_idx] = Self::possible_successor_indices(operand_attrs)[..] else {
            return IRStatus::Unchanged;
        };
        let op = self.get_operation();
        let dest = op.deref(ctx).get_successor(succ_idx);
        let dest_opds = self.successor_operands(ctx, succ_idx);
        let br_op = BrOp::new(ctx, dest, dest_opds).get_operation();
        rewriter.insert_operation(ctx, br_op);
        rewriter.replace_operation(ctx, op, br_op);
        IRStatus::Changed
    }
}
