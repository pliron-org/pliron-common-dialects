// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-common-dialects contributors

//! Control-flow ops and related functionality.

use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::{
            BranchOpInterface, IsTerminatorInterface, NResultsInterface, NSuccsInterface,
            OneRegionInterface, OneSuccInterface, OperandSegmentInterface,
            SingleBlockRegionInterface,
        },
        types::{IntegerType, Signedness},
    },
    common_traits::{Named, Verify},
    context::{Context, Ptr},
    debug_info::set_block_arg_name,
    derive::{op_interface_impl, pliron_op},
    identifier::Identifier,
    input_err,
    irbuild::{
        inserter::{IRInserter, Inserter, OpInsertionPoint},
        listener::DummyListener,
    },
    irfmt::{
        parsers::{
            block_opd_parser, delimited_list_parser, process_parsed_ssa_defs, spaced,
            ssa_opd_parser, type_parser,
        },
        printers::{iter_with_sep, list_with_sep},
    },
    linked_list::ContainsLinkedList,
    location::Location,
    op::{Op, OpObj},
    operation::Operation,
    parsable::{IntoParseResult, Parsable},
    printable::{ListSeparator, Printable},
    region::Region,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err,
};
use pliron::{
    builtin::op_interfaces::NRegionsInterface,
    combine::{
        Parser,
        parser::char::{self, spaces},
    },
};

use crate::{
    cf::op_interfaces::{YieldingOp, YieldingRegion},
    index::types::IndexType,
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
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            results,
            vec![],
            0,
        );
        YieldOp { op }
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
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            dest_opds,
            vec![dest],
            0,
        );
        BrOp { op }
    }
}

#[op_interface_impl]
impl BranchOpInterface for BrOp {
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        assert!(succ_idx == 0, "BrOp has exactly one successor");
        self.get_operation().deref(ctx).operands().collect()
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

        let op = CondBrOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                operands,
                vec![true_dest, false_dest],
                0,
            ),
        };
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
impl OperandSegmentInterface for CondBrOp {}

#[op_interface_impl]
impl BranchOpInterface for CondBrOp {
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        assert!(
            succ_idx == 0 || succ_idx == 1,
            "CondBrOp has exactly two successors"
        );
        // Skip the first segment, which is the condition.
        self.get_segment(ctx, succ_idx + 1)
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
        _state: &pliron::printable::State,
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
            condition.disp(ctx),
            op.get_successor(0).deref(ctx).unique_name(ctx),
            iter_with_sep(true_dest_opds.iter(), ListSeparator::CharSpace(',')).disp(ctx),
            op.get_successor(1).deref(ctx).unique_name(ctx),
            iter_with_sep(false_dest_opds.iter(), ListSeparator::CharSpace(',')).disp(ctx),
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

/// Represents a `for` loop with an initial value, an upper bound, and a step.
/// Additional loop-carried variables can be specified as operands and results.
/// The loop body is defined in a region that takes the loop induction variable
/// and loop-carried variables as arguments and yields updated loop-carried variables.
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
    interfaces = [OneRegionInterface, NRegionsInterface<1>, OperandSegmentInterface, YieldingRegion<YieldOp>, SingleBlockRegionInterface],
)]
pub struct ForOp;

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
        let region_arg_names = std::iter::once("iv".try_into().unwrap())
            .chain(iter_args_init.iter().enumerate().map(|(lv_i, v)| {
                v.given_name(ctx)
                    .unwrap_or(format!("loop_var_{}", lv_i).try_into().unwrap())
            }))
            .collect::<Vec<Identifier>>();

        let (operands, segments) = Self::compute_segment_sizes(vec![
            vec![lower_bound, upper_bound, step],
            iter_args_init.to_vec(),
        ]);

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_types,
            operands,
            vec![],
            1,
        );

        let op = ForOp { op };
        op.set_operand_segment_sizes(ctx, segments);

        // Set up the region, its entry block and arguments.
        let region = op.get_region(ctx);
        let entry_block = BasicBlock::new(ctx, Some("entry".try_into().unwrap()), region_arg_types);
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
        self.get_entry(ctx).deref(ctx).get_argument(0)
    }

    /// Get the loop carried variables (block arguments).
    pub fn get_loop_carried_variables(&self, ctx: &Context) -> Vec<Value> {
        self.get_entry(ctx)
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
            write!(f, "{} = ", results.disp(ctx))?;
        }
        let iter_args_init = self.get_iter_args_init(ctx);
        writeln!(
            f,
            "{} {} to {} step {} ({}) {}",
            Self::get_opid_static(),
            self.get_lower_bound(ctx).disp(ctx),
            self.get_upper_bound(ctx).disp(ctx),
            self.get_step(ctx).disp(ctx),
            list_with_sep(&iter_args_init, ListSeparator::CharSpace(',')).disp(ctx),
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

        let op = Operation::new(
            state_stream.state.ctx,
            Self::get_concrete_op_info(),
            result_types,
            operands,
            vec![],
            0,
        );

        let opop = ForOp { op };
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
        "ForOp count mismatch: iter args initializers, number of results, loop carried variables"
    )]
    IterArgsCountMismatch,
    #[error(
        "ForOp induction variable, lower bound, upper bound, and step types must all be IndexType"
    )]
    InductionVarTypeMismatch,
    #[error(
        "ForOp result types, iter args initializers, and loop carried variable types must match"
    )]
    IterArgsTypeMismatch,
}

impl Verify for ForOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let results: Vec<_> = self.get_operation().deref(ctx).results().collect();
        let iter_args_init = self.get_iter_args_init(ctx);
        let loop_carried_vars = self.get_loop_carried_variables(ctx);

        if results.len() != iter_args_init.len() || results.len() != loop_carried_vars.len() {
            return verify_err!(self.loc(ctx), ForOpVerifyErr::IterArgsCountMismatch);
        }

        // Verify that the types of results, initializers, and loop-carried variables match.
        for i in 0..results.len() {
            let res_ty = results[i].get_type(ctx);
            let init_ty = iter_args_init[i].get_type(ctx);
            let var_ty = loop_carried_vars[i].get_type(ctx);
            if res_ty != init_ty || res_ty != var_ty {
                return verify_err!(self.loc(ctx), ForOpVerifyErr::IterArgsTypeMismatch);
            }
        }

        let iv_ty = self.get_induction_variable(ctx).get_type(ctx);
        let lb_ty = self.get_lower_bound(ctx).get_type(ctx);
        let ub_ty = self.get_upper_bound(ctx).get_type(ctx);
        let step_ty = self.get_step(ctx).get_type(ctx);
        if iv_ty.deref(ctx).downcast_ref::<IndexType>().is_none()
            || lb_ty.deref(ctx).downcast_ref::<IndexType>().is_none()
            || ub_ty.deref(ctx).downcast_ref::<IndexType>().is_none()
            || step_ty.deref(ctx).downcast_ref::<IndexType>().is_none()
        {
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
//// ## Region(s)
///   - A single region, with a single block, containing the loop body.
///   The block takes as arguments the loop induction variables, and must
///   end with a [YieldOp] with no operands.
///   Use [ExecuteRegionOp] to embed multi-block control flow in the loop body.
#[pliron_op(
    name = "cf.nd_for",
    interfaces = [OneRegionInterface, NRegionsInterface<1>, OperandSegmentInterface, YieldingRegion<YieldOp>, SingleBlockRegionInterface],
)]
pub struct NDForOp;

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
            list_with_sep(&lower_bounds, ListSeparator::CharSpace(',')).disp(ctx),
            list_with_sep(&upper_bounds, ListSeparator::CharSpace(',')).disp(ctx),
            list_with_sep(&steps, ListSeparator::CharSpace(',')).disp(ctx),
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

        let op = Operation::new(
            state_stream.state.ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        );

        let opop = NDForOp { op };

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
    #[error(
        "NDForOp induction variables, lower bounds, upper bounds, and steps must all be of IndexType"
    )]
    InductionVarTypeMismatch,
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

        let lower_bounds = self.get_lower_bounds(ctx);
        let upper_bounds = self.get_upper_bounds(ctx);
        let steps = self.get_steps(ctx);

        for i in 0..lower_bounds.len() {
            let lb_ty = lower_bounds[i].get_type(ctx);
            let ub_ty = upper_bounds[i].get_type(ctx);
            let step_ty = steps[i].get_type(ctx);
            if lb_ty.deref(ctx).downcast_ref::<IndexType>().is_none()
                || ub_ty.deref(ctx).downcast_ref::<IndexType>().is_none()
                || step_ty.deref(ctx).downcast_ref::<IndexType>().is_none()
            {
                return verify_err!(self.loc(ctx), NDForOpVerifyErr::InductionVarTypeMismatch);
            }
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
        self.get_entry(ctx)
            .deref(ctx)
            .arguments()
            .collect::<Vec<_>>()
    }

    /// Get the number of induction variables (i.e., loop dimensions).
    pub fn get_num_induction_variables(&self, ctx: &Context) -> usize {
        self.get_entry(ctx).deref(ctx).get_num_arguments()
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

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            1,
        );

        let op = NDForOp { op };
        op.set_operand_segment_sizes(ctx, segments);

        // Set up the region and its entry block.
        let region = op.get_region(ctx);
        let entry_block = BasicBlock::new(ctx, Some("entry".try_into().unwrap()), region_arg_types);
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
    interfaces = [OneRegionInterface, NRegionsInterface<1>, YieldingRegion<YieldOp>],
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
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_types,
            vec![],
            vec![],
            1,
        );

        let op = ExecuteRegionOp { op };

        // Set up the region and its entry block.
        let region = op.get_region(ctx);
        let entry_block = BasicBlock::new(ctx, Some("entry".try_into().unwrap()), vec![]);
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
            write!(f, "{} = ", results.disp(ctx))?;
        }
        let result_types = op.results().map(|r| r.get_type(ctx)).collect::<Vec<_>>();
        writeln!(
            f,
            "{} -> ({}) {}",
            Self::get_opid_static(),
            list_with_sep(&result_types, ListSeparator::CharSpace(',')).disp(ctx),
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

        let op = Operation::new(
            state_stream.state.ctx,
            Self::get_concrete_op_info(),
            result_types,
            vec![],
            vec![],
            0,
        );

        let opop = ExecuteRegionOp { op };

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
        if self.get_entry(ctx).deref(ctx).get_num_arguments() != 0 {
            return verify_err!(self.loc(ctx), ExecuteRegionOpVerifyErr::RegionHasArguments);
        }
        Ok(())
    }
}
