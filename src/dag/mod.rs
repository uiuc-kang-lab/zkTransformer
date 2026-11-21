use crate::util::serialization::{ark_de, ark_de_vec, ark_se, ark_se_vec};
use crate::util::transcript::Transcript;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ndarray::ArrayD;
use plonky2::{timed, util::timing::TimingTree};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::basicblock::einsum::{einsum_input_permutes, permute_evals_by_ranges};
use crate::basicblock::{reducer::Reducer, BasicBlock, BasicBlockType};
use crate::crypto::sumcheck::prover::SumcheckProver;
use crate::crypto::{polycommit::{Commitment as CommitmentTrait, MLPolyCommit}, SumcheckProof};
use crate::crypto::{LinearSumcheckProver, SparseBoolSumcheckProver, SumcheckVerifier};
use crate::util::arith::calc_pow;
use crate::util::arith::{f_to_int, get_n};
use crate::util::poly::CryptoField;
use crate::util::poly::{
  evaluate_lagrange_basis, evaluate_range_structured, evaluate_two_pow_structured,
  get_cached_inverses, range_dense, two_pow_dense, DenseMLPoly, MLPoly, SparseMLPoly,
};
use crate::TABLE_COMMIT_LOG;

#[cfg(feature = "arkworks")]
use rayon::prelude::*;

pub mod bert;
pub mod builder;
pub mod dense;
pub mod gpt2;
pub mod gptj;
pub mod llama;
pub mod nanogpt;
pub mod oneshot;

pub use builder::DagBuilder;
pub use dense::dense_add_relu;

pub type NodeId = usize;
pub type EdgeId = usize;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct AliasId(usize); // per-(edge, consumer node, input slot) alias

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
  Auxiliary,
  Constant,
  Input,
  Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
  Uint,
  Int,
  Bool,
  Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolyType {
  Dense,
  Sparse,
}

#[derive(Debug)]
pub struct Witness<F: CryptoField + 'static> {
  pub shape: Vec<usize>,
  pub data: Option<Box<dyn MLPoly<F>>>, // Could either be a dense or sparse multilinear polynomial
  pub data_int: Option<Vec<i128>>,      // integer representation of the data, only collected for dense and not auxiliary polys
  pub poly_type: PolyType,
  pub data_type: DataType,
  pub sf: usize,
  pub role: Role,
  /// If `Some(perm)`, both `data` and `data_int` have been permuted by
  /// `permute_evals_by_ranges(.., perm)`. Set by `Dag::apply_prepermute`.
  /// Consumers (currently only `Einsum::prove`) check this against the
  /// permute_vec they would otherwise apply themselves and skip the call when
  /// they match. The commitment for such a witness is to the permuted poly,
  /// and any claim point on this edge is in permuted space.
  pub is_permuted_with: Option<Vec<(usize, usize)>>,
}

impl<F: CryptoField + 'static> Witness<F> {
  pub fn clear_data(&mut self) {
    self.data = None;
    self.data_int = None;
  }

  pub fn get(&self, indices: &[usize]) -> F {
    let shape_next_pow = self.shape.iter().map(|&s| s.next_power_of_two()).collect::<Vec<usize>>();
    let index = indices.iter().enumerate().map(|(i, &index)| index * shape_next_pow[..i].iter().fold(1, |acc, &x| acc * x)).sum();
    self.data.as_ref().unwrap().index(index)
  }

  pub fn set(&mut self, indices: &[usize], value: F) {
    let shape_next_pow = self.shape.iter().map(|&s| s.next_power_of_two()).collect::<Vec<usize>>();
    let index = indices.iter().enumerate().map(|(i, &index)| index * shape_next_pow[..i].iter().fold(1, |acc, &x| acc * x)).sum();
    *self.data.as_mut().unwrap().index_mut(index) = value;
  }

  pub fn ndarray(&self) -> ArrayD<i128> {
    let shape = self.shape.clone().into_iter().rev().map(|s| s.next_power_of_two()).collect::<Vec<usize>>();
    let data = self
      .data
      .as_ref()
      .unwrap()
      .as_any()
      .downcast_ref::<DenseMLPoly<F>>()
      .unwrap()
      .evaluations
      .par_iter()
      .map(|f| f_to_int(*f))
      .collect::<Vec<i128>>();
    ArrayD::from_shape_vec(shape, data).unwrap().reversed_axes()
  }
}

impl<F: CryptoField + 'static> Witness<F> {
  pub fn new(shape: Vec<usize>, data: Vec<F>, data_type: DataType, sf: usize, role: Role) -> Self {
    let n = get_n(&shape);
    let data_int = data.par_iter().map(|f| f_to_int(*f)).collect::<Vec<i128>>();
    Self {
      shape,
      data: Some(Box::new(DenseMLPoly::new(n, data))),
      data_int: Some(data_int),
      poly_type: PolyType::Dense,
      data_type,
      sf,
      role,
      is_permuted_with: None,
    }
  }

  pub fn new_sparse(shape: Vec<usize>, data: SparseMLPoly<F>, data_type: DataType, sf: usize, role: Role) -> Self {
    Self {
      shape,
      data: Some(Box::new(data)),
      data_int: None,
      poly_type: PolyType::Sparse,
      data_type,
      sf,
      role,
      is_permuted_with: None,
    }
  }

  pub fn new_wo_data(shape: Vec<usize>, data_type: DataType, sf: usize, role: Role) -> Self {
    Self {
      shape,
      data: None,
      data_int: None,
      poly_type: PolyType::Dense,
      data_type,
      sf,
      role,
      is_permuted_with: None,
    }
  }
}

impl<F: CryptoField + 'static> Clone for Witness<F> {
  fn clone(&self) -> Self {
    Self {
      shape: self.shape.clone(),
      data: self.data.as_ref().map(|d| d.clone_box()),
      data_int: self.data_int.clone(),
      poly_type: self.poly_type,
      data_type: self.data_type,
      sf: self.sf,
      role: self.role,
      is_permuted_with: self.is_permuted_with.clone(),
    }
  }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct LookupProof<F: CryptoField + 'static> {
  pub table_proofs: Vec<SumcheckProof<F>>,
  #[serde(serialize_with = "se_nested_vec", deserialize_with = "de_nested_vec")]
  pub middle_claims: Vec<Vec<F>>,
  pub bool_proofs: Vec<SumcheckProof<F>>,
}

/// Wrapper struct for LookupProof without middle_claims (for proof size measurement)
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct LookupProofOnly<F: CryptoField + 'static> {
  pub table_proofs: Vec<SumcheckProof<F>>,
  pub bool_proofs: Vec<SumcheckProof<F>>,
}

impl<F: CryptoField + 'static> LookupProofOnly<F> {
  pub fn from_lookup_proof(proof: &LookupProof<F>) -> Self
  where
    F: Clone,
  {
    Self {
      table_proofs: proof.table_proofs.clone(),
      bool_proofs: proof.bool_proofs.clone(),
    }
  }
}

/// Extract only the proofs from LookupProof (excluding middle_claims)
pub fn extract_lookup_proof_only<F: CryptoField + 'static>(proof: &LookupProof<F>) -> LookupProofOnly<F>
where
  F: Clone,
{
  LookupProofOnly::from_lookup_proof(proof)
}

/// Serialize nested Vec<Vec<F>> for arkworks types
fn se_nested_vec<S, A: CanonicalSerialize>(a: &Vec<Vec<A>>, s: S) -> Result<S::Ok, S::Error>
where
  S: serde::Serializer,
{
  use serde::ser::SerializeSeq;
  let mut seq = s.serialize_seq(Some(a.len()))?;
  for inner in a {
    let mut inner_bytes: Vec<Vec<u8>> = Vec::with_capacity(inner.len());
    for elem in inner {
      let mut bytes = vec![];
      elem.serialize_compressed(&mut bytes).map_err(serde::ser::Error::custom)?;
      inner_bytes.push(bytes);
    }
    seq.serialize_element(&inner_bytes)?;
  }
  seq.end()
}

/// Deserialize nested Vec<Vec<F>> for arkworks types
fn de_nested_vec<'de, D, A: CanonicalDeserialize>(data: D) -> Result<Vec<Vec<A>>, D::Error>
where
  D: serde::de::Deserializer<'de>,
{
  let v: Vec<Vec<Vec<u8>>> = serde::de::Deserialize::deserialize(data)?;
  v.into_iter()
    .map(|inner| inner.into_iter().map(|bytes| A::deserialize_compressed_unchecked(bytes.as_slice()).map_err(serde::de::Error::custom)).collect())
    .collect()
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "F: CanonicalSerialize + CanonicalDeserialize, DP::Proof: Serialize, SP::Proof: Serialize",
  deserialize = "F: CanonicalSerialize + CanonicalDeserialize, DP::Proof: for<'a> Deserialize<'a>, SP::Proof: for<'a> Deserialize<'a>"
))]
pub struct EdgeProof<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>, SP: MLPolyCommit<F, SparseMLPoly<F>>> {
  pub claims: Vec<Claim<F>>,
  pub dense_opening_proof: Vec<DP::Proof>,
  pub sparse_opening_proof: Vec<SP::Proof>,
}

impl<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>, SP: MLPolyCommit<F, SparseMLPoly<F>>> EdgeProof<F, DP, SP> {
  pub fn new() -> Self {
    Self {
      claims: Vec::new(),
      dense_opening_proof: Vec::new(),
      sparse_opening_proof: Vec::new(),
    }
  }
}

/// Wrapper struct for EdgeProof opening proofs only (excluding claims)
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "DP::Proof: Serialize, SP::Proof: Serialize",
  deserialize = "DP::Proof: for<'a> Deserialize<'a>, SP::Proof: for<'a> Deserialize<'a>"
))]
pub struct EdgeOpeningProof<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>, SP: MLPolyCommit<F, SparseMLPoly<F>>> {
  pub dense_opening_proof: Vec<DP::Proof>,
  pub sparse_opening_proof: Vec<SP::Proof>,
  #[serde(skip)]
  _phantom: std::marker::PhantomData<F>,
}

impl<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>, SP: MLPolyCommit<F, SparseMLPoly<F>>> EdgeOpeningProof<F, DP, SP> {
  pub fn from_edge_proof(edge_proof: &EdgeProof<F, DP, SP>) -> Self
  where
    DP::Proof: Clone,
    SP::Proof: Clone,
  {
    Self {
      dense_opening_proof: edge_proof.dense_opening_proof.clone(),
      sparse_opening_proof: edge_proof.sparse_opening_proof.clone(),
      _phantom: std::marker::PhantomData,
    }
  }
}

/// Extract only sumcheck proofs from sumcheck_proofs (excluding claims)
pub fn extract_sumcheck_proofs_only<F: CryptoField + CanonicalSerialize + CanonicalDeserialize>(
  sumcheck_proofs: &[Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>],
) -> Vec<Option<Vec<SumcheckProof<F>>>>
where
  F: Clone,
{
  sumcheck_proofs.iter().map(|opt| opt.as_ref().map(|(proofs, _claims)| proofs.clone())).collect()
}

/// Extract only opening proofs from edge proofs (excluding claims)
pub fn extract_opening_proofs_only<F, DP, SP>(opening_proofs: &[EdgeProof<F, DP, SP>]) -> Vec<EdgeOpeningProof<F, DP, SP>>
where
  F: CryptoField + 'static,
  DP: MLPolyCommit<F, DenseMLPoly<F>>,
  SP: MLPolyCommit<F, SparseMLPoly<F>>,
  DP::Proof: Clone,
  SP::Proof: Clone,
{
  opening_proofs.iter().map(|ep| EdgeOpeningProof::from_edge_proof(ep)).collect()
}

impl<F: CryptoField + 'static, DP: MLPolyCommit<F, DenseMLPoly<F>>, SP: MLPolyCommit<F, SparseMLPoly<F>>> Clone for EdgeProof<F, DP, SP>
where
  DP::Proof: Clone,
  SP::Proof: Clone,
{
  fn clone(&self) -> Self {
    Self {
      claims: self.claims.clone(),
      dense_opening_proof: self.dense_opening_proof.clone(),
      sparse_opening_proof: self.sparse_opening_proof.clone(),
    }
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: CanonicalSerialize + CanonicalDeserialize")]
pub struct Claim<F: CryptoField> {
  pub edge_id: EdgeId,
  pub sparse_id: usize,
  #[serde(serialize_with = "ark_se_vec", deserialize_with = "ark_de_vec")]
  pub point: Vec<F>,
  #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
  pub eval: F,
}

#[derive(Debug, Clone)]
pub struct Node {
  id: NodeId,
  kind: BasicBlockType,
  inputs: Vec<EdgeId>,  // physical edges
  outputs: Vec<EdgeId>, // physical edges (1..N)
}

#[derive(Debug)]
pub struct Dag {
  nodes: Vec<Node>,
  num_edges: usize,
  topo: Vec<NodeId>,
  range: Vec<NodeId>,      // currently only support non-negative range
  two_pow: Vec<NodeId>,    // nodes that compute 2^(-k)
  zero_check: Vec<NodeId>, // nodes that assert input MLE is identically zero
  // Physical connectivity
  consumers: Vec<Vec<NodeId>>,    // edge -> consumer node list
  producers: Vec<Option<NodeId>>, // edge -> producing node (None for graph inputs)
  input_ports: Vec<EdgeId>,
  output_ports: Vec<EdgeId>,

  // ----- Alias view (for readability in backward passes) -----
  // One alias per (edge, consumer node, consumer input slot).
  edge_aliases: Vec<Vec<AliasId>>, // physical edge -> alias list
  alias_to_edge: Vec<EdgeId>,
  alias_to_consumer: Vec<NodeId>,
  alias_input_slot: Vec<usize>, // which input index on the consumer
}

impl Dag {
  pub fn num_edges(&self) -> usize {
    self.num_edges
  }

  /// Collect all unique polynomial sizes (number of variables) from witnesses in the DAG.
  /// This is used to determine which SRS sizes need to be generated/loaded during setup.
  pub fn collect_polynomial_sizes<F: CryptoField + 'static>(&self, witnesses: &[Vec<Witness<F>>]) -> BTreeSet<usize> {
    let mut sizes = BTreeSet::new();

    for witness_opt in witnesses.iter() {
      for witness in witness_opt.iter() {
        if let Some(ref poly_data) = witness.data {
          let n = poly_data.n();
          sizes.insert(n);
        } else {
          // For witnesses without data yet (e.g., inputs before witness generation),
          // compute the polynomial size from the shape
          let n = get_n(&witness.shape);
          sizes.insert(n);
        }
      }
    }

    sizes
  }

  /// Decide which edges can have their witness data pre-permuted (so that the
  /// `permute_evals_by_ranges` call inside `Einsum::prove` becomes a no-op).
  ///
  /// An edge is eligible iff:
  ///   * its current witness `role` is `Constant` (pre-permuting other roles
  ///     is unsafe because the producer node's `prove` expects the layout it
  ///     produced); AND
  ///   * every consumer of the edge is an `Einsum` node; AND
  ///   * every einsum consumer expects the *same* permute vector for the slot
  ///     where this edge is consumed.
  ///
  /// Returns a map `EdgeId -> permute_vec`. Identity permutations are
  /// excluded — applying them is a wasted memcpy.
  pub fn compute_prepermute_plan<F: CryptoField + 'static>(
    &self,
    witnesses: &[Vec<Witness<F>>],
  ) -> BTreeMap<EdgeId, Vec<(usize, usize)>> {
    let mut plan: BTreeMap<EdgeId, Vec<(usize, usize)>> = BTreeMap::new();

    'edge: for e in 0..self.num_edges {
      // Must have a single Constant witness with data.
      if witnesses[e].len() != 1 {
        continue;
      }
      let w = &witnesses[e][0];
      if w.role != Role::Constant || w.poly_type != PolyType::Dense {
        continue;
      }

      let consumers = &self.consumers[e];
      if consumers.is_empty() {
        continue;
      }

      // All consumers must be Einsum, and all must want the same permute_vec
      // for the slot(s) where they consume `e`.
      let mut agreed: Option<Vec<(usize, usize)>> = None;
      for &cnid in consumers {
        let node = &self.nodes[cnid];
        let equation = match &node.kind {
          BasicBlockType::Einsum(eq) => eq.equation.clone(),
          _ => continue 'edge, // non-einsum consumer disqualifies the edge
        };

        // Build the input shapes for this einsum from its declared inputs.
        let in_shapes: Vec<Vec<usize>> = node.inputs.iter().map(|&ie| witnesses[ie][0].shape.clone()).collect();
        let perms = einsum_input_permutes(&equation, &in_shapes);

        // For every input slot of `cnid` that is `e`, check the permute_vec.
        for (slot, &ie) in node.inputs.iter().enumerate() {
          if ie != e {
            continue;
          }
          let perm = &perms[slot];

          // Identity (single (0, n) range) means no real reordering.
          let n = get_n(&w.shape);
          let is_identity = perm.len() == 1 && perm[0] == (0usize, n);
          if is_identity {
            continue 'edge;
          }

          match &agreed {
            None => agreed = Some(perm.clone()),
            Some(prev) if prev == perm => {}
            Some(_) => continue 'edge, // conflicting permute → keep original
          }
        }
      }

      if let Some(perm) = agreed {
        plan.insert(e, perm);
      }
    }

    plan
  }

  /// Apply the prepermute plan to each tagged Constant witness in-place.
  /// After this call, `Einsum::prove` will skip its own `permute_evals_by_ranges`
  /// for each pre-permuted input. Must be called *before* `commit` so the
  /// commitment is to the permuted polynomial.
  ///
  /// Safe to call multiple times: a witness with `is_permuted_with == Some(perm)`
  /// is skipped by this function.
  pub fn apply_prepermute<F: CryptoField + 'static>(&self, witnesses: &mut [Vec<Witness<F>>]) {
    let plan = self.compute_prepermute_plan(witnesses);
    let mut applied: usize = 0;
    let mut total_elems: u128 = 0;
    for (e, perm) in plan {
      let w_vec = &mut witnesses[e];
      if w_vec.len() != 1 {
        continue;
      }
      let w = &mut w_vec[0];
      if w.is_permuted_with.is_some() {
        continue;
      }
      let n = get_n(&w.shape);
      let elems = 1u128 << n;

      // Permute data_int (used by the n>=18 fix_variables_zkgpt path).
      if let Some(di) = w.data_int.as_ref() {
        let permuted = permute_evals_by_ranges(di, n, &perm);
        w.data_int = Some(permuted);
      }

      // Permute the DenseMLPoly evaluations (used by commit + small-n path).
      // The MLPoly trait only exposes `as_any` (immutable), so we read out
      // the evaluations and rebuild a fresh DenseMLPoly box.
      if let Some(boxed) = w.data.as_ref() {
        if let Some(dense) = boxed.as_any().downcast_ref::<DenseMLPoly<F>>() {
          let permuted = permute_evals_by_ranges(&dense.evaluations, n, &perm);
          let new_dense = DenseMLPoly::new(n, permuted);
          w.data = Some(Box::new(new_dense));
        }
      }

      w.is_permuted_with = Some(perm);
      applied += 1;
      total_elems += elems;
    }
    println!(
      "apply_prepermute: applied to {} edges, total permuted elements = {}",
      applied, total_elems
    );
  }

  /// Internal evaluator that produces values for **all** edges.
  pub fn run<F: CryptoField + 'static>(&self, witnesses: &mut [Vec<Witness<F>>], feed: &[(EdgeId, Witness<F>)]) {
    assert_eq!(witnesses.len(), self.num_edges, "init vec length must match num_edges");

    // Feed can override (graph inputs, params, anything)
    for (eid, t) in feed {
      witnesses[*eid] = vec![t.clone()];
    }

    // Evaluate in topological order
    for &nid in &self.topo {
      let node = &self.nodes[nid];

      let in_refs: Vec<&Witness<F>> = node.inputs.iter().map(|&e| &witnesses[e][0]).collect();

      // 1) compute
      let outs = node.kind.run(&in_refs);

      // 2) publish
      assert_eq!(outs.len(), node.outputs.len(), "op output arity mismatch");
      for (&eid, out) in node.outputs.iter().zip(outs.into_iter()) {
        witnesses[eid] = vec![out];
      }
    }

    // split the sparse witnesses into multiple witnesses for practicality
    // (disabled by NO_SPARSE_SPLIT=1 for ablation with NonStructuredExp)
    if std::env::var("NO_SPARSE_SPLIT").is_err() {
      for witness in witnesses.iter_mut() {
        let w = &witness[0];
        if w.poly_type == PolyType::Sparse {
          let poly = w.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
          let polys = poly.split_into_blocks(*TABLE_COMMIT_LOG);
          let new_witness = polys
            .iter()
            .map(|poly| Witness::new_sparse(w.shape.clone(), poly.clone(), w.data_type, w.sf, w.role))
            .collect::<Vec<Witness<F>>>();
          *witness = new_witness;
        }
      }
    }
  }

  /// Re-run only the nodes downstream of the given edges (after a feed override).
  /// Assumes all upstream witnesses are already computed.
  pub fn rerun_downstream<F: CryptoField + 'static>(&self, witnesses: &mut [Vec<Witness<F>>], changed_edges: &[EdgeId]) {
    // Collect all nodes that transitively depend on changed_edges
    let mut dirty_nodes = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<NodeId> = std::collections::VecDeque::new();
    for &eid in changed_edges {
      for &nid in &self.consumers[eid] {
        if dirty_nodes.insert(nid) {
          queue.push_back(nid);
        }
      }
    }
    while let Some(nid) = queue.pop_front() {
      let node = &self.nodes[nid];
      for &out_eid in &node.outputs {
        for &consumer_nid in &self.consumers[out_eid] {
          if dirty_nodes.insert(consumer_nid) {
            queue.push_back(consumer_nid);
          }
        }
      }
    }

    // Re-run dirty nodes in topological order
    for &nid in &self.topo {
      if !dirty_nodes.contains(&nid) {
        continue;
      }
      let node = &self.nodes[nid];
      let in_refs: Vec<&Witness<F>> = node.inputs.iter().map(|&e| &witnesses[e][0]).collect();
      let outs = node.kind.run(&in_refs);
      assert_eq!(outs.len(), node.outputs.len(), "op output arity mismatch");
      for (&eid, out) in node.outputs.iter().zip(outs.into_iter()) {
        witnesses[eid] = vec![out];
      }
    }

    // Re-split sparse witnesses that were freshly computed (single element).
    // Already-split witnesses (multiple blocks from initial run) must NOT be re-split,
    // as witness[0] would be just the first block, not the full polynomial.
    if std::env::var("NO_SPARSE_SPLIT").is_err() {
      for witness in witnesses.iter_mut() {
        if witness.len() == 1 && witness[0].poly_type == PolyType::Sparse {
          let w = &witness[0];
          let poly = w.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
          let polys = poly.split_into_blocks(*TABLE_COMMIT_LOG);
          let new_witness = polys
            .iter()
            .map(|poly| Witness::new_sparse(w.shape.clone(), poly.clone(), w.data_type, w.sf, w.role))
            .collect::<Vec<Witness<F>>>();
          *witness = new_witness;
        }
      }
    }
  }

  fn is_advice_output_edge(&self, edge_id: EdgeId) -> bool {
    let Some(p) = self.producers[edge_id] else {
      return false;
    };
    let node = &self.nodes[p];
    match &node.kind {
      BasicBlockType::ClampLower(_)
      | BasicBlockType::RMSReciprocal(_)
      | BasicBlockType::Reciprocal(_)
      | BasicBlockType::DivConst(_)
      | BasicBlockType::SoftmaxConst(_)
      | BasicBlockType::SigmoidConst(_) => node.outputs.first() == Some(&edge_id),
      _ => false,
    }
  }

  fn is_sparse_aux_edge(&self, edge_id: EdgeId) -> bool {
    if let Some(p) = self.producers[edge_id] {
      let node = &self.nodes[p];
      match &node.kind {
        BasicBlockType::NonNegative(_) => node.outputs.first() == Some(&edge_id),
        BasicBlockType::ScaleDown(_)
        | BasicBlockType::ScaleUp(_)
        | BasicBlockType::ExpHelper(_)
        | BasicBlockType::NonStructuredExp(_) => node.outputs.get(1) == Some(&edge_id),
        _ => false,
      }
    } else {
      self.two_pow.iter().any(|&tp| self.nodes[tp].inputs.first() == Some(&edge_id))
    }
  }

  fn must_commit_edge(&self, edge_id: EdgeId) -> bool {
    self.producers[edge_id].is_none()
      || self.is_advice_output_edge(edge_id)
      || self.is_sparse_aux_edge(edge_id)
      || self.consumers[edge_id].is_empty()
  }

  fn should_commit<F: CryptoField + 'static>(&self, witness: &Witness<F>, edge_id: EdgeId) -> bool {
    if self.must_commit_edge(edge_id) {
      return true;
    }
    match witness.role {
      Role::Constant | Role::Auxiliary | Role::Input => true,
      Role::Output => self.consumers[edge_id].is_empty(),
    }
  }

  fn range_node_total_bits(&self, nid: NodeId) -> usize {
    match &self.nodes[nid].kind {
      BasicBlockType::ScaleDown(sd) => sd.input_sf - sd.output_sf,
      BasicBlockType::ScaleUp(su) => su.output_sf - su.input_sf,
      BasicBlockType::ExpHelper(_) => crate::basicblock::exp::K_BITS,
      BasicBlockType::NonNegative(nn) => nn.table_size_log,
      other => panic!("Unexpected node kind in range: {:?}", other),
    }
  }

  fn range_block_table_vars(&self, nid: NodeId, num_blocks: usize, sparse_id: usize) -> usize {
    let total_bits = self.range_node_total_bits(nid);
    if total_bits == 0 || num_blocks <= 1 {
      total_bits
    } else {
      let offset = sparse_id.saturating_mul(*TABLE_COMMIT_LOG);
      total_bits.saturating_sub(offset).min(*TABLE_COMMIT_LOG)
    }
  }

  pub fn commit<F, DP, SP>(
    &self,
    dense_key: &DP::CommitmentKey,
    sparse_key: &SP::CommitmentKey,
    witnesses: &[Vec<Witness<F>>],
    dense_commitments: &mut [Option<DP::Commitment>],
    sparse_commitments: &mut [Option<Vec<SP::Commitment>>],
    timing: &mut TimingTree,
  ) where
    F: CryptoField + 'static,
    DP: MLPolyCommit<F, DenseMLPoly<F>>,
    SP: MLPolyCommit<F, SparseMLPoly<F>>,
  {
    for (e, witness) in witnesses.iter().enumerate() {
      if !witness.is_empty() && self.should_commit(&witness[0], e) && witness[0].data.is_some() {
        match witness[0].poly_type {
          PolyType::Dense => {
            if dense_commitments[e].is_none() {
              let w = &witness[0];
              let poly = w.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().unwrap();
              if w.role == Role::Constant {
                timed!(timing, format!("commit to constant for edge {}", e).as_str(), {
                  dense_commitments[e] = Some(DP::commit(poly, dense_key));
                });
              } else {
                dense_commitments[e] = Some(DP::commit(poly, dense_key));
              }
            }
          }
          PolyType::Sparse => {
            if sparse_commitments[e].is_none() {
              let comms: Vec<SP::Commitment> = witness
                .iter()
                .map(|w| {
                  let poly = w.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
                  SP::commit(poly, sparse_key)
                })
                .collect();
              sparse_commitments[e] = Some(comms);
            }
          }
        }
      }
    }
  }

  pub fn prove<F, DP, SP>(
    &self,
    dense_key: &DP::CommitmentKey,
    sparse_key: &SP::CommitmentKey,
    witnesses: &[Vec<Witness<F>>],
    dense_commitments: &[Option<DP::Commitment>],
    sparse_commitments: &[Option<Vec<SP::Commitment>>],
    transcript: &mut Transcript<F>,
    timing: &mut TimingTree,
  ) -> (
    Vec<Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>>,
    Vec<EdgeProof<F, DP, SP>>,
    LookupProof<F>,
    LookupProof<F>,
    Vec<Option<Vec<SumcheckProof<F>>>>,
  )
  where
    F: CryptoField + 'static + Sync,
    DP: MLPolyCommit<F, DenseMLPoly<F>>,
    SP: MLPolyCommit<F, SparseMLPoly<F>>,
    DP::Proof: Clone + Send,
    SP::Proof: Clone + Send,
    DP::CommitmentKey: Sync,
    SP::CommitmentKey: Sync,
    DP::Commitment: Sync,
    SP::Commitment: Sync,
    DenseMLPoly<F>: Sync,
    SparseMLPoly<F>: Sync,
  {
    // 0. Absorb all polynomial commitments into Fiat-Shamir transcript
    for comm in dense_commitments.iter() {
      if let Some(c) = comm {
        transcript.append_message(b"dense_commitment", &c.to_transcript_bytes());
      }
    }
    for comms in sparse_commitments.iter() {
      if let Some(cs) = comms {
        for c in cs {
          transcript.append_message(b"sparse_commitment", &c.to_transcript_bytes());
        }
      }
    }

    // 1. initialize proofs with pre-allocated capacity
    let reducer = BasicBlockType::Reducer(Reducer {});
    let mut node_proofs: Vec<Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>> = Vec::with_capacity(self.nodes.len());
    node_proofs.resize_with(self.nodes.len(), || None);
    let mut reducer_proofs: Vec<Option<Vec<SumcheckProof<F>>>> = Vec::with_capacity(self.nodes.len());
    reducer_proofs.resize_with(self.nodes.len(), || None);
    let mut edge_proofs: Vec<EdgeProof<F, DP, SP>> = Vec::with_capacity(self.num_edges);
    edge_proofs.resize_with(self.num_edges, || EdgeProof::new());

    // 2. open the final output / sink polynomials at fresh Fiat-Shamir challenge points
    let mut claims: Vec<Vec<Claim<F>>> = Vec::with_capacity(self.num_edges);
    claims.resize_with(self.num_edges, || Vec::new());
    let mut nodes_to_prove = BTreeSet::new();
    for &e in &self.output_ports {
      let w = &witnesses[e][0];
      let is_active_output = w.poly_type == PolyType::Dense && (w.role == Role::Output || !self.consumers[e].is_empty());
      if !is_active_output {
        continue;
      }
      if claims[e].is_empty() {
        let point: Vec<F> = (0..get_n(&w.shape)).map(|_| transcript.challenge_scalar(b"challenge")).collect();
        let eval = w.data.as_ref().unwrap().evaluate_at_point(&point);
        transcript.append_scalar(b"output_claim_eval", &eval);
        claims[e].push(Claim {
          edge_id: e,
          sparse_id: 0,
          point,
          eval,
        });
      }
      for &cid in &self.consumers[e] {
        if matches!(
          self.nodes[cid].kind,
          BasicBlockType::NonNegative(_) | BasicBlockType::ZeroCheck(_)
        ) {
          node_proofs[cid] = Some((vec![], vec![claims[e][0].clone()]));
        }
      }
      if let Some(prod) = self.producers[e] {
        nodes_to_prove.insert(prod);
      }
    }

    // 2. prove from outputs to inputs in reverse-topological order
    let mut preserve_nodes = BTreeSet::new();
    let consumer_sets: Vec<BTreeSet<NodeId>> = self.consumers.iter().map(|consumers| consumers.iter().copied().collect()).collect();

    while !nodes_to_prove.is_empty() {
      let node_id = nodes_to_prove.pop_last().unwrap();
      let mut can_prove = true;
      for &edge in &self.nodes[node_id].outputs {
        if consumer_sets[edge].iter().any(|&c| nodes_to_prove.contains(&c)) {
          can_prove = false;
          break;
        }
      }
      if !can_prove {
        nodes_to_prove.insert(node_id);
        continue;
      }
      let node = &self.nodes[node_id];

      let total_edges = node.inputs.len() + node.outputs.len();
      let mut edge_ids = Vec::with_capacity(total_edges);
      edge_ids.extend_from_slice(&node.inputs);
      edge_ids.extend_from_slice(&node.outputs);

      let mut local_witnesses = Vec::with_capacity(edge_ids.len());
      for &e in &edge_ids {
        local_witnesses.push(&witnesses[e][0]);
      }

      let mut local_claims = Vec::new();
      for &e in &node.outputs {
        local_claims.extend(claims[e].iter());
      }

      let (reducer_proof, reducer_claims) = if local_claims.len() > 1 {
        println!("proving reducer for node {} | kind {:?}", node_id, self.nodes[node_id].kind);
        let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
        let reducer_edge_ids = vec![node.outputs[0]];
        let (proofs, new_claims) = timed!(
          timing,
          format!("prove reducer for node {} | kind {:?}", node_id, self.nodes[node_id].kind).as_str(),
          { reducer.prove(&reducer_witness, &reducer_edge_ids, &local_claims, transcript) }
        );
        (proofs, new_claims)
      } else {
        (vec![], vec![])
      };
      if !reducer_claims.is_empty() {
        local_claims = reducer_claims.iter().collect();
        let output_edge_id = node.outputs[0];
        claims[output_edge_id].push(reducer_claims[0].clone());
        reducer_proofs[node_id] = Some(reducer_proof);
      }

      // prove the node
      println!("proving node {} | kind {:?}", node_id, self.nodes[node_id].kind);
      let (proofs, new_claims) = timed!(
        timing,
        format!("prove node {} | kind {:?}", node_id, self.nodes[node_id].kind).as_str(),
        { self.nodes[node_id].kind.prove(&local_witnesses, &edge_ids, &local_claims, transcript) }
      );

      // Absorb intermediate input claim evaluations into Fiat-Shamir transcript
      for c in &new_claims {
        transcript.append_scalar(b"node_claim_eval", &c.eval);
      }

      let mut node_claims = Vec::with_capacity(new_claims.len() + local_claims.len());
      node_claims.extend_from_slice(&new_claims);
      for claim in local_claims {
        node_claims.push(claim.clone());
      }
      node_proofs[node_id] = Some((proofs, node_claims));

      for c in new_claims {
        claims[c.edge_id].push(c);
      }
      for &e in &node.inputs {
        if let Some(producer) = self.producers[e] {
          preserve_nodes.insert(producer);
        }
      }

      for &preserve_node in &preserve_nodes {
        nodes_to_prove.insert(preserve_node);
      }
      preserve_nodes.clear();
    }

    // 3. prove lookups BEFORE populating `edge_proofs` so sparse lookup & booleanity
    //    claims are included and opened via `SP::open`!
    let (two_pow_table_proofs, two_pow_middle_claims, two_pow_bool_proofs) =
      self.prove_two_pow(witnesses, &mut claims, transcript, timing);
    let (range_table_proofs, range_middle_claims, range_bool_proofs) =
      self.prove_range(witnesses, &mut claims, &node_proofs, transcript, timing);
    let two_pow_proof = LookupProof {
      table_proofs: two_pow_table_proofs,
      middle_claims: two_pow_middle_claims,
      bool_proofs: two_pow_bool_proofs,
    };
    let range_proof = LookupProof {
      table_proofs: range_table_proofs,
      middle_claims: range_middle_claims,
      bool_proofs: range_bool_proofs,
    };

    claims.iter().enumerate().for_each(|(e, c)| edge_proofs[e].claims.extend(c.iter().cloned()));

    // 4. provide opening proofs for all committed edges (including sparse lookup polys and final outputs)
    println!("proving opening proofs");
    timed!(timing, "provide opening proofs", {
      let mut dense_tasks = Vec::new();
      let mut sparse_tasks = Vec::new();

      for e in 0..self.num_edges {
        let w = &witnesses[e][0];
        if !self.should_commit(w, e) {
          continue;
        }
        for c in &edge_proofs[e].claims {
          if w.poly_type == PolyType::Dense {
            dense_tasks.push((e, c.clone()));
          } else {
            sparse_tasks.push((e, c.clone()));
          }
        }
      }

      #[cfg(feature = "arkworks")]
      let dense_results: Vec<_> = dense_tasks
        .par_iter()
        .map(|(e, claim)| {
          let w = &witnesses[*e][0];
          let proof = DP::open(
            dense_commitments[*e].as_ref().unwrap(),
            w.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().unwrap(),
            dense_key,
            &claim.point,
          );
          (*e, proof)
        })
        .collect();

      #[cfg(not(feature = "arkworks"))]
      let dense_results: Vec<_> = dense_tasks
        .iter()
        .map(|(e, claim)| {
          let w = &witnesses[*e][0];
          let proof = DP::open(
            dense_commitments[*e].as_ref().unwrap(),
            w.data.as_ref().unwrap().as_any().downcast_ref::<DenseMLPoly<F>>().unwrap(),
            dense_key,
            &claim.point,
          );
          (*e, proof)
        })
        .collect();

      #[cfg(feature = "arkworks")]
      let sparse_results: Vec<_> = sparse_tasks
        .par_iter()
        .map(|(e, claim)| {
          let w = &witnesses[*e][claim.sparse_id];
          let proof = SP::open(
            &sparse_commitments[*e].as_ref().unwrap()[claim.sparse_id],
            w.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap(),
            sparse_key,
            &claim.point,
          );
          (*e, proof)
        })
        .collect();

      #[cfg(not(feature = "arkworks"))]
      let sparse_results: Vec<_> = sparse_tasks
        .iter()
        .map(|(e, claim)| {
          let w = &witnesses[*e][claim.sparse_id];
          let proof = SP::open(
            &sparse_commitments[*e].as_ref().unwrap()[claim.sparse_id],
            w.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap(),
            sparse_key,
            &claim.point,
          );
          (*e, proof)
        })
        .collect();

      for (e, proof) in dense_results {
        edge_proofs[e].dense_opening_proof.push(proof);
      }

      for (e, proof) in sparse_results {
        edge_proofs[e].sparse_opening_proof.push(proof);
      }
    });

    (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs)
  }

  pub fn verify<F, DP, SP>(
    &self,
    node_proofs: &[Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>],
    edge_proofs: &[EdgeProof<F, DP, SP>],
    range_proof: &LookupProof<F>,
    two_pow_proof: &LookupProof<F>,
    reducer_proofs: &[Option<Vec<SumcheckProof<F>>>],
    witnesses: &[Vec<Witness<F>>],
    dense_key: &DP::VerifierKey,
    sparse_key: &SP::VerifierKey,
    dense_commitments: &[Option<DP::Commitment>],
    sparse_commitments: &[Option<Vec<SP::Commitment>>],
    transcript: &mut Transcript<F>,
  ) -> bool
  where
    F: CryptoField + 'static,
    DP: MLPolyCommit<F, DenseMLPoly<F>>,
    SP: MLPolyCommit<F, SparseMLPoly<F>>,
    DP::Proof: Clone + Sync,
    SP::Proof: Clone + Sync,
    DP::VerifierKey: Sync,
    SP::VerifierKey: Sync,
    DP::Commitment: Sync,
    SP::Commitment: Sync,
  {
    if node_proofs.len() != self.nodes.len()
      || edge_proofs.len() != self.num_edges
      || reducer_proofs.len() != self.nodes.len()
      || witnesses.len() != self.num_edges
      || dense_commitments.len() != self.num_edges
      || sparse_commitments.len() != self.num_edges
    {
      return false;
    }

    for e in 0..self.num_edges {
      if witnesses[e].is_empty() {
        return false;
      }
      let expected_sparse = self.is_sparse_aux_edge(e);
      if expected_sparse && witnesses[e][0].poly_type != PolyType::Sparse {
        return false;
      }
      if !expected_sparse && (witnesses[e][0].poly_type != PolyType::Dense || witnesses[e].len() != 1) {
        return false;
      }
      if self.is_advice_output_edge(e) && witnesses[e][0].role != Role::Auxiliary {
        return false;
      }
    }

    let committed_edge_ids: Vec<EdgeId> = (0..self.num_edges)
      .filter(|&e| !witnesses[e].is_empty() && self.should_commit(&witnesses[e][0], e))
      .collect();

    for &e in &committed_edge_ids {
      if witnesses[e][0].poly_type == PolyType::Dense {
        if dense_commitments[e].is_none() {
          return false;
        }
      } else if sparse_commitments[e].as_ref().map_or(true, |cs| cs.len() != witnesses[e].len()) {
        return false;
      }
    }

    // Absorb all polynomial commitments into Fiat-Shamir transcript (must match prover)
    for comm in dense_commitments.iter() {
      if let Some(c) = comm {
        transcript.append_message(b"dense_commitment", &c.to_transcript_bytes());
      }
    }
    for comms in sparse_commitments.iter() {
      if let Some(cs) = comms {
        for c in cs {
          transcript.append_message(b"sparse_commitment", &c.to_transcript_bytes());
        }
      }
    }

    let reducer = BasicBlockType::Reducer(Reducer {});
    let mut verified = true;

    let mut claims: Vec<Vec<Claim<F>>> = Vec::with_capacity(self.num_edges);
    claims.resize_with(self.num_edges, || Vec::new());
    let mut nodes_to_verify = BTreeSet::new();
    for &e in &self.output_ports {
      if witnesses[e].is_empty() {
        return false;
      }
      let w = &witnesses[e][0];
      let is_active_output = w.poly_type == PolyType::Dense && (w.role == Role::Output || !self.consumers[e].is_empty());
      if !is_active_output {
        continue;
      }
      if claims[e].is_empty() {
        let expected_n = get_n(&w.shape);
        if edge_proofs[e].claims.is_empty() {
          return false;
        }
        let c0 = &edge_proofs[e].claims[0];
        if c0.edge_id != e || c0.sparse_id != 0 || c0.point.len() != expected_n {
          return false;
        }
        let point: Vec<F> = (0..expected_n).map(|_| transcript.challenge_scalar(b"challenge")).collect();
        if c0.point != point {
          return false;
        }
        transcript.append_scalar(b"output_claim_eval", &c0.eval);
        claims[e].push(c0.clone());
      }
      if let Some(prod) = self.producers[e] {
        nodes_to_verify.insert(prod);
      }
    }

    let mut preserve_nodes = BTreeSet::new();
    let mut visited_nodes = BTreeSet::new();
    let consumer_sets: Vec<BTreeSet<NodeId>> = self.consumers.iter().map(|consumers| consumers.iter().copied().collect()).collect();

    let start = std::time::Instant::now();
    while !nodes_to_verify.is_empty() {
      let u = nodes_to_verify.pop_last().unwrap();
      let mut can_verify = true;
      for &edge in &self.nodes[u].outputs {
        if consumer_sets[edge].iter().any(|&c| nodes_to_verify.contains(&c)) {
          can_verify = false;
          break;
        }
      }
      if !can_verify {
        nodes_to_verify.insert(u);
        continue;
      }
      visited_nodes.insert(u);
      let node = &self.nodes[u];

      let Some((local_sumcheck_proofs_vec, local_claims_vec)) = node_proofs[u].as_ref() else {
        return false;
      };
      let expected_in_edges: Vec<EdgeId> = match &node.kind {
        BasicBlockType::TwoPow(_)
        | BasicBlockType::RMSReciprocal(_)
        | BasicBlockType::Reciprocal(_)
        | BasicBlockType::DivConst(_)
        | BasicBlockType::SoftmaxConst(_)
        | BasicBlockType::SigmoidConst(_) => vec![],
        BasicBlockType::NonStructuredExp(_) => vec![node.inputs[1], node.outputs[1]],
        BasicBlockType::NonNegative(_) | BasicBlockType::ZeroCheck(_) => return false,
        _ => node.inputs.clone(),
      };

      if local_claims_vec.len() != expected_in_edges.len() + 1 {
        return false;
      }
      for (slot, &exp_edge) in expected_in_edges.iter().enumerate() {
        if local_claims_vec[slot].edge_id != exp_edge || local_claims_vec[slot].sparse_id != 0 {
          return false;
        }
      }
      let out_claim_in_node = &local_claims_vec[expected_in_edges.len()];
      if out_claim_in_node.edge_id != node.outputs[0] || out_claim_in_node.sparse_id != 0 {
        return false;
      }

      let mut out_claims_accum: Vec<Claim<F>> = Vec::new();
      for &e in &node.outputs {
        out_claims_accum.extend(claims[e].iter().cloned());
      }
      if out_claims_accum.is_empty() {
        return false;
      }

      if out_claims_accum.len() == 1 {
        if reducer_proofs[u].is_some()
          || out_claim_in_node.edge_id != out_claims_accum[0].edge_id
          || out_claim_in_node.sparse_id != out_claims_accum[0].sparse_id
          || out_claim_in_node.point != out_claims_accum[0].point
          || out_claim_in_node.eval != out_claims_accum[0].eval
        {
          println!("=== output claim chaining mismatch for node {u} ===");
          return false;
        }
      } else {
        println!("verifying reducer for node {u} | kind {:?}", self.nodes[u].kind);
        let Some(red_proofs_vec) = reducer_proofs[u].as_ref() else {
          return false;
        };
        let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
        let mut reducer_claims = out_claims_accum;
        reducer_claims.push(out_claim_in_node.clone());
        let reducer_claims_refs: Vec<&Claim<F>> = reducer_claims.iter().collect();
        let reducer_sumcheck_proofs: Vec<&SumcheckProof<F>> = red_proofs_vec.iter().collect();
        let reducer_verified = reducer.verify(&reducer_witness, &reducer_claims_refs, &reducer_sumcheck_proofs, transcript);
        verified = verified && reducer_verified;
        if !reducer_verified {
          println!("=== verified reducer for node {u}: {reducer_verified} ===");
          return false;
        }
        // Mirror `prove` line 817 so committed auxiliary outputs record the reduced claim
        claims[node.outputs[0]].push(out_claim_in_node.clone());
      }

      let local_claims: Vec<&Claim<F>> = local_claims_vec.iter().collect();
      let local_witnesses: Vec<&Witness<F>> = local_claims.iter().map(|&c| &witnesses[c.edge_id][0]).collect();
      let local_sumcheck_proofs: Vec<&SumcheckProof<F>> = local_sumcheck_proofs_vec.iter().collect();

      println!("verifying node {u} | kind {:?}", self.nodes[u].kind);
      let node_verified = self.nodes[u].kind.verify(&local_witnesses, &local_claims, &local_sumcheck_proofs, transcript);
      if !node_verified {
        println!("=== verified node {u}: {node_verified} ===");
        return false;
      }
      verified = verified && node_verified;

      for c in &local_claims_vec[..expected_in_edges.len()] {
        transcript.append_scalar(b"node_claim_eval", &c.eval);
      }

      for &e in &node.inputs {
        if let Some(producer) = self.producers[e] {
          preserve_nodes.insert(producer);
        }
      }

      for &preserve_node in &preserve_nodes {
        nodes_to_verify.insert(preserve_node);
      }
      preserve_nodes.clear();

      for c in &local_claims_vec[..expected_in_edges.len()] {
        claims[c.edge_id].push(c.clone());
      }
    }
    let end = start.elapsed();
    println!("time taken to verify nodes: {:?}", end);

    // verify the two_pow, range, and zero_check proofs
    let start = std::time::Instant::now();
    let two_pow_verified = self.verify_two_pow(witnesses, edge_proofs, &mut claims, two_pow_proof, transcript);
    let range_verified = self.verify_range(node_proofs, witnesses, edge_proofs, &visited_nodes, &mut claims, range_proof, transcript);
    let zero_check_verified = self.verify_zero_check(node_proofs, &claims);
    let end = start.elapsed();
    println!("time taken to verify two_pow and range: {:?}", end);
    if !two_pow_verified {
      println!("=== verified two_pow: {two_pow_verified} ===");
    }
    if !range_verified {
      println!("=== verified range: {range_verified} ===");
    }
    if !zero_check_verified {
      println!("=== verified zero_check: {zero_check_verified} ===");
    }
    verified = verified && range_verified && two_pow_verified && zero_check_verified;
    if !verified {
      return false;
    }

    // verify all committed edges (exact claim binding + opening proofs) - fully parallelized
    let start = std::time::Instant::now();
    let opening_verified = committed_edge_ids.par_iter().all(|&e| {
      if edge_proofs[e].claims.len() != claims[e].len() {
        println!(
          "claim count mismatch for edge {e}: edge_proofs has {}, expected {}",
          edge_proofs[e].claims.len(),
          claims[e].len()
        );
        return false;
      }
      for i in 0..claims[e].len() {
        let got = &edge_proofs[e].claims[i];
        let exp = &claims[e][i];
        if got.edge_id != exp.edge_id || got.sparse_id != exp.sparse_id || got.point != exp.point || got.eval != exp.eval {
          println!("claim mismatch for edge {e} claim {i}");
          return false;
        }
      }
      if witnesses[e][0].poly_type == PolyType::Dense {
        if edge_proofs[e].dense_opening_proof.len() != claims[e].len() || !edge_proofs[e].sparse_opening_proof.is_empty() {
          return false;
        }
        (0..claims[e].len()).into_par_iter().all(|i| {
          let (ok, opened_eval) = DP::verify_and_extract(
            dense_commitments[e].as_ref().unwrap(),
            &edge_proofs[e].dense_opening_proof[i],
            dense_key,
            &claims[e][i].point,
          );
          if ok && opened_eval != claims[e][i].eval {
            println!("opening eval mismatch for dense edge {e} claim {i}");
            return false;
          }
          ok
        })
      } else {
        if edge_proofs[e].sparse_opening_proof.len() != claims[e].len() || !edge_proofs[e].dense_opening_proof.is_empty() {
          return false;
        }
        let comms = sparse_commitments[e].as_ref().unwrap();
        (0..claims[e].len()).into_par_iter().all(|i| {
          if claims[e][i].sparse_id >= comms.len() {
            return false;
          }
          let (ok, opened_eval) = SP::verify_and_extract(
            &comms[claims[e][i].sparse_id],
            &edge_proofs[e].sparse_opening_proof[i],
            sparse_key,
            &claims[e][i].point,
          );
          if ok && opened_eval != claims[e][i].eval {
            println!("opening eval mismatch for sparse edge {e} claim {i}");
            return false;
          }
          ok
        })
      }
    });
    let end = start.elapsed();
    println!("time taken to verify irreducible edges: {:?}", end);
    verified && opening_verified
  }

  /// Verifies that every ZeroCheck node's input-edge claim matches the accumulated
  /// output-port claim on `node.inputs[0]` and evaluates to zero.
  pub fn verify_zero_check<F: CryptoField + 'static>(
    &self,
    node_proofs: &[Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>],
    accum_claims: &[Vec<Claim<F>>],
  ) -> bool {
    for &nid in &self.zero_check {
      let in_edge = self.nodes[nid].inputs[0];
      let Some((sc_proofs, sink_claims)) = node_proofs[nid].as_ref() else {
        println!("zero_check: missing node_proof for node {nid}");
        return false;
      };
      if !sc_proofs.is_empty() || sink_claims.len() != 1 || accum_claims[in_edge].is_empty() {
        return false;
      }
      let exp = &accum_claims[in_edge][0];
      if sink_claims[0].edge_id != in_edge
        || sink_claims[0].point != exp.point
        || sink_claims[0].eval != exp.eval
        || sink_claims[0].eval != <F as CryptoField>::zero()
      {
        println!("zero_check: invalid or non-zero eval for node {nid}");
        return false;
      }
    }
    true
  }

  pub fn verify_range<F, DP, SP>(
    &self,
    node_proofs: &[Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>],
    witnesses: &[Vec<Witness<F>>],
    edge_proofs: &[EdgeProof<F, DP, SP>],
    visited_nodes: &BTreeSet<NodeId>,
    claims: &mut Vec<Vec<Claim<F>>>,
    range_proof: &LookupProof<F>,
    transcript: &mut Transcript<F>,
  ) -> bool
  where
    F: CryptoField + 'static,
    DP: MLPolyCommit<F, DenseMLPoly<F>>,
    SP: MLPolyCommit<F, SparseMLPoly<F>>,
  {
    if self.range.is_empty() {
      return range_proof.table_proofs.is_empty()
        && range_proof.middle_claims.is_empty()
        && range_proof.bool_proofs.is_empty();
    }
    println!("verifying range");

    let range_table_proofs = &range_proof.table_proofs;
    let range_middle_claims = &range_proof.middle_claims;
    let range_bool_proofs = &range_proof.bool_proofs;

    if range_middle_claims.len() != self.range.len() {
      return false;
    }

    struct RangeBlockMeta<F> {
      range_idx: usize,
      aux_edge: EdgeId,
      sparse_id: usize,
      table_vars: usize,
      inp_point: Vec<F>,
    }
    let mut blocks_meta: Vec<RangeBlockMeta<F>> = Vec::new();
    let mut max_blocks = 1usize;

    for (i, &n) in self.range.iter().enumerate() {
      let node = &self.nodes[n];
      let aux_id = if matches!(node.kind, BasicBlockType::NonNegative(_)) { 0 } else { 1 };
      let aux_edge = node.outputs[aux_id];
      let num_blocks = witnesses[aux_edge].len();
      let total_bits = self.range_node_total_bits(n);
      let expected_split_blocks = if total_bits == 0 {
        1
      } else {
        (total_bits + *TABLE_COMMIT_LOG - 1) / *TABLE_COMMIT_LOG
      };
      if num_blocks == 0 || (num_blocks != 1 && num_blocks != expected_split_blocks) {
        return false;
      }
      if range_middle_claims[i].len() != num_blocks {
        return false;
      }
      max_blocks = max_blocks.max(num_blocks);

      let Some((sc_proofs, node_claim)) = node_proofs[n].as_ref() else {
        return false;
      };
      let expected_inp_n = get_n(&witnesses[node.inputs[0]][0].shape);
      if matches!(node.kind, BasicBlockType::NonNegative(_)) {
        let in_edge = node.inputs[0];
        if !sc_proofs.is_empty() || node_claim.len() != 1 || claims[in_edge].is_empty() {
          return false;
        }
        let exp = &claims[in_edge][0];
        if node_claim[0].edge_id != in_edge
          || node_claim[0].point != exp.point
          || node_claim[0].eval != exp.eval
          || node_claim[0].point.len() != expected_inp_n
        {
          return false;
        }
      } else if !visited_nodes.contains(&n) || node_claim.len() != 2 || node_claim[0].point.len() != expected_inp_n {
        return false;
      }

      for sparse_id in 0..num_blocks {
        let table_vars = self.range_block_table_vars(n, num_blocks, sparse_id);
        blocks_meta.push(RangeBlockMeta {
          range_idx: i,
          aux_edge,
          sparse_id,
          table_vars,
          inp_point: node_claim[0].point.clone(),
        });
      }
    }

    // Absorb `range_middle_claims` BEFORE squeezing challenges (fixes adaptive Fiat-Shamir forgery)
    for mc in range_middle_claims {
      for &m in mc {
        transcript.append_scalar(b"range_middle_claim", &m);
      }
    }

    let alpha: F = transcript.challenge_scalar(b"table_alpha");
    let beta: F = transcript.challenge_scalar(b"table_beta");
    let betas = calc_pow(beta, self.range.len());
    let gamma: F = transcript.challenge_scalar(b"table_gamma");
    let gammas = calc_pow(gamma, max_blocks);

    for (i, &n) in self.range.iter().enumerate() {
      println!("verifying range for node {n} | kind {:?}", self.nodes[n].kind);
      let node = &self.nodes[n];
      let aux_id = if matches!(node.kind, BasicBlockType::NonNegative(_)) { 0 } else { 1 };
      let auxs = &witnesses[node.outputs[aux_id]];
      let node_claim = &node_proofs[n].as_ref().unwrap().1;
      let eval_to_check = if matches!(node.kind, BasicBlockType::ScaleDown(_)) {
        let input_sf = witnesses[node.inputs[0]][0].sf;
        let output_sf = witnesses[node.outputs[0]][0].sf;
        let shift = input_sf - output_sf;
        if shift >= 127 {
          return false;
        }
        let rescale_factor: i128 = 1i128 << shift;
        let rescale_factor_divided_by_2: i128 = rescale_factor / 2;
        let rescale_factor_f = F::from(rescale_factor);
        let rescale_factor_divided_by_2_f = F::from(rescale_factor_divided_by_2);
        node_claim[0].eval - node_claim[1].eval * rescale_factor_f + rescale_factor_divided_by_2_f
      } else if matches!(node.kind, BasicBlockType::ScaleUp(_)) {
        let input_sf = witnesses[node.inputs[0]][0].sf;
        let output_sf = witnesses[node.outputs[0]][0].sf;
        let shift = output_sf - input_sf;
        if shift >= 127 {
          return false;
        }
        let rescale_factor: i128 = 1i128 << shift;
        let rescale_factor_divided_by_2: i128 = rescale_factor / 2;
        let rescale_factor_f = F::from(rescale_factor);
        let rescale_factor_divided_by_2_f = F::from(rescale_factor_divided_by_2);
        node_claim[0].eval * rescale_factor_f - node_claim[1].eval + rescale_factor_divided_by_2_f
      } else if matches!(node.kind, BasicBlockType::ExpHelper(_)) {
        let cache = get_cached_inverses::<F>();
        (node_claim[0].eval - node_claim[1].eval) * cache.neg_ln2_inv
      } else {
        node_claim[0].eval
      };

      let mut eval_acc = <F as CryptoField>::zero();
      for sparse_id in 0..auxs.len() {
        let shift = sparse_id.saturating_mul(*TABLE_COMMIT_LOG);
        if shift >= 127 {
          return false;
        }
        let middle_sum = range_middle_claims[i][sparse_id];
        eval_acc += middle_sum * F::from(1i128 << shift);
      }
      if eval_to_check != eval_acc {
        println!("range eval_to_check mismatch for node {n} | kind {:?} | #auxs {}", node.kind, auxs.len());
        return false;
      }
    }

    // Group blocks deterministically by `table_vars` and `aux_num_var`
    let mut table_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut bool_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (b_idx, meta) in blocks_meta.iter().enumerate() {
      table_groups.entry(meta.table_vars).or_default().push(b_idx);
      bool_groups.entry(meta.inp_point.len() + meta.table_vars).or_default().push(b_idx);
    }

    if range_table_proofs.len() != table_groups.len() || range_bool_proofs.len() != bool_groups.len() {
      return false;
    }

    // Verify each table sumcheck proof AND its terminal evaluation against opened sparse claims
    for (group_idx, (&table_vars, block_idxs)) in table_groups.iter().enumerate() {
      let mut group_expected_sum = <F as CryptoField>::zero();
      for &b_idx in block_idxs {
        let meta = &blocks_meta[b_idx];
        let mid = range_middle_claims[meta.range_idx][meta.sparse_id];
        group_expected_sum += (mid + alpha) * betas[meta.range_idx] * gammas[meta.sparse_id];
      }

      let mut sumcheck_verifier = SumcheckVerifier::new(table_vars, 2, transcript);
      let (verification_result, r_tbl) =
        sumcheck_verifier.verify(transcript, range_table_proofs[group_idx].round_messages.clone(), group_expected_sum);
      let Some(running_sum) = verification_result else {
        println!("range table proof verification failed");
        return false;
      };

      let mut group_aux_eval = <F as CryptoField>::zero();
      for &b_idx in block_idxs {
        let meta = &blocks_meta[b_idx];
        let claim_idx = claims[meta.aux_edge].len();
        if claim_idx >= edge_proofs[meta.aux_edge].claims.len() {
          return false;
        }
        let c = &edge_proofs[meta.aux_edge].claims[claim_idx];
        let expected_point: Vec<F> = meta.inp_point.iter().chain(r_tbl.iter()).cloned().collect();
        if c.edge_id != meta.aux_edge || c.sparse_id != meta.sparse_id || c.point != expected_point {
          return false;
        }
        transcript.append_scalar(b"range_table_eval", &c.eval);
        group_aux_eval += betas[meta.range_idx] * gammas[meta.sparse_id] * c.eval;
        claims[meta.aux_edge].push(c.clone());
      }

      let range_eval = alpha + evaluate_range_structured(&r_tbl);
      if running_sum != group_aux_eval * range_eval {
        println!("range table proof terminal evaluation mismatch");
        return false;
      }
    }

    // Verify each booleanity sumcheck proof AND its terminal evaluation against opened sparse claims
    let one = <F as CryptoField>::one();
    for (group_idx, (&aux_num_var, block_idxs)) in bool_groups.iter().enumerate() {
      let mut bool_verifier = SumcheckVerifier::new(aux_num_var, 3, transcript);
      let eq_challenge: Vec<F> = (0..aux_num_var).map(|_| transcript.challenge_scalar(b"challenge")).collect();
      let expected_sum = <F as CryptoField>::zero();
      let (verification_result, r_bool) =
        bool_verifier.verify(transcript, range_bool_proofs[group_idx].round_messages.clone(), expected_sum);
      let Some(running_sum) = verification_result else {
        println!("bool proof verification failed for aux_num_var {}", aux_num_var);
        return false;
      };

      let mut inner_bool = <F as CryptoField>::zero();
      for &b_idx in block_idxs {
        let meta = &blocks_meta[b_idx];
        let claim_idx = claims[meta.aux_edge].len();
        if claim_idx >= edge_proofs[meta.aux_edge].claims.len() {
          return false;
        }
        let c = &edge_proofs[meta.aux_edge].claims[claim_idx];
        if c.edge_id != meta.aux_edge || c.sparse_id != meta.sparse_id || c.point != r_bool {
          return false;
        }
        transcript.append_scalar(b"range_bool_eval", &c.eval);
        let bg = betas[meta.range_idx] * gammas[meta.sparse_id];
        let w = bg * bg;
        inner_bool += w * c.eval * (c.eval - one);
        claims[meta.aux_edge].push(c.clone());
      }

      let eq_eval = eq_challenge
        .iter()
        .zip(r_bool.iter())
        .fold(one, |acc, (&ec, &rb)| acc * (rb * ec + (one - rb) * (one - ec)));
      if running_sum != eq_eval * inner_bool {
        println!("bool proof terminal evaluation mismatch for aux_num_var {}", aux_num_var);
        return false;
      }
    }

    true
  }

  pub fn prove_range<F: CryptoField + 'static>(
    &self,
    witnesses: &[Vec<Witness<F>>],
    claims: &mut Vec<Vec<Claim<F>>>,
    node_proofs: &[Option<(Vec<SumcheckProof<F>>, Vec<Claim<F>>)>],
    transcript: &mut Transcript<F>,
    timing: &mut TimingTree,
  ) -> (Vec<SumcheckProof<F>>, Vec<Vec<F>>, Vec<SumcheckProof<F>>) {
    let mut table_proofs = Vec::new();
    let mut bool_proofs = Vec::new();
    let mut middle_claims: Vec<Vec<F>> = vec![vec![]; self.range.len()];
    if !self.range.is_empty() {
      println!("proving range");
      timed!(timing, "prove range", {
        struct RangeBlockProverData<F> {
          range_idx: usize,
          aux_edge: EdgeId,
          sparse_id: usize,
          table_vars: usize,
          inp_point: Vec<F>,
          part_evals: Vec<F>,
        }
        let mut blocks_data: Vec<RangeBlockProverData<F>> = Vec::new();
        let mut max_blocks = 1usize;

        for (i, &n) in self.range.iter().enumerate() {
          let node = &self.nodes[n];
          let aux_id = if matches!(node.kind, BasicBlockType::NonNegative(_)) { 0 } else { 1 };
          let aux_edge = node.outputs[aux_id];
          let inp_claim = &node_proofs[n].as_ref().unwrap().1[0];
          let d = inp_claim.point.len();
          let mask = if d == 0 { 0 } else { (1usize << d) - 1 };
          let lagrange_basis = evaluate_lagrange_basis(&inp_claim.point);
          let auxs = &witnesses[aux_edge];
          let num_blocks = auxs.len();
          max_blocks = max_blocks.max(num_blocks);

          for (sparse_id, aux) in auxs.iter().enumerate() {
            let table_vars = self.range_block_table_vars(n, num_blocks, sparse_id);
            let aux_poly = aux.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
            debug_assert_eq!(aux_poly.n(), d + table_vars);

            let mut part_evals = vec![<F as CryptoField>::zero(); 1usize << table_vars];
            for (&full_idx, &val) in &aux_poly.evaluations {
              let input_idx = full_idx & mask;
              let table_idx = full_idx >> d;
              part_evals[table_idx] += val * lagrange_basis[input_idx];
            }

            let mid = part_evals
              .iter()
              .enumerate()
              .fold(<F as CryptoField>::zero(), |acc, (idx, &val)| acc + val * F::from(idx as u32));
            middle_claims[i].push(mid);

            blocks_data.push(RangeBlockProverData {
              range_idx: i,
              aux_edge,
              sparse_id,
              table_vars,
              inp_point: inp_claim.point.clone(),
              part_evals,
            });
          }
        }

        // Absorb `middle_claims` BEFORE squeezing challenges
        for mc in &middle_claims {
          for &m in mc {
            transcript.append_scalar(b"range_middle_claim", &m);
          }
        }

        let alpha: F = transcript.challenge_scalar(b"table_alpha");
        let beta: F = transcript.challenge_scalar(b"table_beta");
        let betas = calc_pow(beta, self.range.len());
        let gamma: F = transcript.challenge_scalar(b"table_gamma");
        let gammas = calc_pow(gamma, max_blocks);

        let mut table_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut bool_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (b_idx, bd) in blocks_data.iter().enumerate() {
          table_groups.entry(bd.table_vars).or_default().push(b_idx);
          bool_groups.entry(bd.inp_point.len() + bd.table_vars).or_default().push(b_idx);
        }

        // 1. Batched Shout table sumchecks (grouped by `table_vars`) + table evaluation claims
        for (&table_vars, block_idxs) in &table_groups {
          let mut batched_evals = vec![<F as CryptoField>::zero(); 1usize << table_vars];
          for &b_idx in block_idxs {
            let bd = &blocks_data[b_idx];
            let bg = betas[bd.range_idx] * gammas[bd.sparse_id];
            for (k, &v) in bd.part_evals.iter().enumerate() {
              if v != <F as CryptoField>::zero() {
                batched_evals[k] += bg * v;
              }
            }
          }
          let batched_aux_poly = DenseMLPoly::new(table_vars, batched_evals);
          let range_poly = range_dense::<F>(table_vars).add_by_scalar(alpha);
          let mut sumcheck_prover = LinearSumcheckProver::<F>::new(table_vars, 2, transcript);
          let sumcheck_proof = sumcheck_prover.prove(&vec![batched_aux_poly, range_poly], transcript);
          let r_tbl = &sumcheck_prover.challenges;
          let tbl_basis = evaluate_lagrange_basis(r_tbl);
          table_proofs.push(sumcheck_proof);

          for &b_idx in block_idxs {
            let bd = &blocks_data[b_idx];
            let eval = bd
              .part_evals
              .iter()
              .zip(tbl_basis.iter())
              .fold(<F as CryptoField>::zero(), |acc, (&pv, &tb)| acc + pv * tb);
            let point: Vec<F> = bd.inp_point.iter().chain(r_tbl.iter()).cloned().collect();
            transcript.append_scalar(b"range_table_eval", &eval);
            claims[bd.aux_edge].push(Claim {
              edge_id: bd.aux_edge,
              sparse_id: bd.sparse_id,
              point,
              eval,
            });
          }
        }

        // 2. Batched booleanity sumchecks (grouped by `aux_num_var`) + booleanity evaluation claims
        for (&aux_num_var, block_idxs) in &bool_groups {
          let mut bool_weights = Vec::with_capacity(block_idxs.len());
          let mut bool_entries = Vec::with_capacity(block_idxs.len());
          for &b_idx in block_idxs {
            let bd = &blocks_data[b_idx];
            let bg = betas[bd.range_idx] * gammas[bd.sparse_id];
            bool_weights.push(bg * bg);
            let aux_poly = witnesses[bd.aux_edge][bd.sparse_id]
              .data
              .as_ref()
              .unwrap()
              .as_any()
              .downcast_ref::<SparseMLPoly<F>>()
              .unwrap();
            let entries: Vec<(usize, F)> = aux_poly.evaluations.iter().map(|(&idx, &val)| (idx, val)).collect();
            bool_entries.push(entries);
          }

          let mut bool_prover = SparseBoolSumcheckProver::<F>::new(aux_num_var, transcript);
          let eq_challenge: Vec<F> = (0..aux_num_var).map(|_| transcript.challenge_scalar(b"challenge")).collect();
          let bool_proof = bool_prover.prove_with_entries(&bool_weights, &bool_entries, &eq_challenge, transcript);
          bool_proofs.push(bool_proof);

          for (term_idx, &b_idx) in block_idxs.iter().enumerate() {
            let bd = &blocks_data[b_idx];
            let eval = bool_prover.term_final_eval(term_idx);
            transcript.append_scalar(b"range_bool_eval", &eval);
            claims[bd.aux_edge].push(Claim {
              edge_id: bd.aux_edge,
              sparse_id: bd.sparse_id,
              point: bool_prover.challenges.clone(),
              eval,
            });
          }
        }
      });
    }
    (table_proofs, middle_claims, bool_proofs)
  }

  pub fn verify_two_pow<F, DP, SP>(
    &self,
    witnesses: &[Vec<Witness<F>>],
    edge_proofs: &[EdgeProof<F, DP, SP>],
    claims: &mut Vec<Vec<Claim<F>>>,
    two_pow_proof: &LookupProof<F>,
    transcript: &mut Transcript<F>,
  ) -> bool
  where
    F: CryptoField + 'static,
    DP: MLPolyCommit<F, DenseMLPoly<F>>,
    SP: MLPolyCommit<F, SparseMLPoly<F>>,
  {
    if self.two_pow.is_empty() {
      return two_pow_proof.table_proofs.is_empty()
        && two_pow_proof.middle_claims.is_empty()
        && two_pow_proof.bool_proofs.is_empty();
    }
    println!("verifying two_pow");

    let two_pow_table_proofs = &two_pow_proof.table_proofs;
    let two_pow_middle_claims = &two_pow_proof.middle_claims;
    let two_pow_bool_proofs = &two_pow_proof.bool_proofs;

    if two_pow_table_proofs.len() != 1 || two_pow_middle_claims.len() != self.two_pow.len() {
      return false;
    }
    for mc in two_pow_middle_claims {
      if mc.len() != 1 {
        return false;
      }
    }

    // Absorb `two_pow_middle_claims` BEFORE squeezing challenges
    for mc in two_pow_middle_claims {
      transcript.append_scalar(b"two_pow_middle_claim", &mc[0]);
    }

    // `two_pow_alpha` enforces Shout's row-sum == 1 (Hamming weight 1) invariant
    let alpha: F = transcript.challenge_scalar(b"two_pow_alpha");
    let beta: F = transcript.challenge_scalar(b"table_beta");
    let betas = calc_pow(beta, self.two_pow.len());

    let mut table_expected_sum = <F as CryptoField>::zero();
    for (i, &n) in self.two_pow.iter().enumerate() {
      let out_edge = self.nodes[n].outputs[0];
      if claims[out_edge].is_empty() {
        return false;
      }
      let eval_to_check = claims[out_edge].last().unwrap().eval;
      let mid = two_pow_middle_claims[i][0];
      if eval_to_check != mid {
        return false;
      }
      table_expected_sum += (mid + alpha) * betas[i];
    }

    let mut sumcheck_verifier = SumcheckVerifier::new(4, 2, transcript);
    let (verification_result, r_tbl) =
      sumcheck_verifier.verify(transcript, two_pow_table_proofs[0].round_messages.clone(), table_expected_sum);
    let Some(running_sum) = verification_result else {
      return false;
    };

    let mut batched_aux_eval = <F as CryptoField>::zero();
    for (i, &n) in self.two_pow.iter().enumerate() {
      let node = &self.nodes[n];
      let aux_edge = node.inputs[0];
      let out_edge = node.outputs[0];
      let inp_point = &claims[out_edge].last().unwrap().point;
      let claim_idx = claims[aux_edge].len();
      if claim_idx >= edge_proofs[aux_edge].claims.len() {
        return false;
      }
      let c = &edge_proofs[aux_edge].claims[claim_idx];
      let expected_point: Vec<F> = inp_point.iter().chain(r_tbl.iter()).cloned().collect();
      if c.edge_id != aux_edge || c.sparse_id != 0 || c.point != expected_point {
        return false;
      }
      transcript.append_scalar(b"two_pow_table_eval", &c.eval);
      batched_aux_eval += betas[i] * c.eval;
      claims[aux_edge].push(c.clone());
    }

    let table_eval = alpha + evaluate_two_pow_structured(&r_tbl);
    if running_sum != batched_aux_eval * table_eval {
      println!("two_pow table proof terminal evaluation mismatch");
      return false;
    }

    // Verify booleanity for any standalone TwoPow selector not already covered by `self.range`
    let mut uncovered_groups: BTreeMap<usize, Vec<(usize, NodeId)>> = BTreeMap::new();
    for (i, &n) in self.two_pow.iter().enumerate() {
      let aux_edge = self.nodes[n].inputs[0];
      if witnesses[aux_edge].len() != 1 {
        return false;
      }
      let covered_by_range = self.producers[aux_edge].map_or(false, |p| {
        if !self.range.contains(&p) {
          return false;
        }
        let rnode = &self.nodes[p];
        let r_aux_id = if matches!(rnode.kind, BasicBlockType::NonNegative(_)) { 0 } else { 1 };
        rnode.outputs.get(r_aux_id) == Some(&aux_edge) && self.range_block_table_vars(p, 1, 0) == 4
      });
      if !covered_by_range {
        let aux_num_var = get_n(&witnesses[self.nodes[n].outputs[0]][0].shape) + 4;
        uncovered_groups.entry(aux_num_var).or_default().push((i, n));
      }
    }

    if two_pow_bool_proofs.len() != uncovered_groups.len() {
      return false;
    }
    let one = <F as CryptoField>::one();
    for (group_idx, (&aux_num_var, items)) in uncovered_groups.iter().enumerate() {
      let mut bool_verifier = SumcheckVerifier::new(aux_num_var, 3, transcript);
      let eq_challenge: Vec<F> = (0..aux_num_var).map(|_| transcript.challenge_scalar(b"challenge")).collect();
      let (verification_result, r_bool) =
        bool_verifier.verify(transcript, two_pow_bool_proofs[group_idx].round_messages.clone(), <F as CryptoField>::zero());
      let Some(bool_running_sum) = verification_result else {
        return false;
      };

      let mut inner_bool = <F as CryptoField>::zero();
      for &(i, n) in items {
        let aux_edge = self.nodes[n].inputs[0];
        let claim_idx = claims[aux_edge].len();
        if claim_idx >= edge_proofs[aux_edge].claims.len() {
          return false;
        }
        let c = &edge_proofs[aux_edge].claims[claim_idx];
        if c.edge_id != aux_edge || c.sparse_id != 0 || c.point != r_bool {
          return false;
        }
        transcript.append_scalar(b"two_pow_bool_eval", &c.eval);
        let w = betas[i] * betas[i];
        inner_bool += w * c.eval * (c.eval - one);
        claims[aux_edge].push(c.clone());
      }

      let eq_eval = eq_challenge
        .iter()
        .zip(r_bool.iter())
        .fold(one, |acc, (&ec, &rb)| acc * (rb * ec + (one - rb) * (one - ec)));
      if bool_running_sum != eq_eval * inner_bool {
        return false;
      }
    }

    true
  }

  pub fn prove_two_pow<F: CryptoField + 'static>(
    &self,
    witnesses: &[Vec<Witness<F>>],
    claims: &mut Vec<Vec<Claim<F>>>,
    transcript: &mut Transcript<F>,
    timing: &mut TimingTree,
  ) -> (Vec<SumcheckProof<F>>, Vec<Vec<F>>, Vec<SumcheckProof<F>>) {
    let mut table_proofs = Vec::new();
    let mut bool_proofs = Vec::new();
    let mut middle_claims: Vec<Vec<F>> = vec![vec![]; self.two_pow.len()];
    if !self.two_pow.is_empty() {
      println!("proving two_pow");
      timed!(timing, "prove two_pow", {
        let mut part_evals_list: Vec<(EdgeId, Vec<F>, Vec<F>)> = Vec::with_capacity(self.two_pow.len());
        for (i, &n) in self.two_pow.iter().enumerate() {
          let node = &self.nodes[n];
          let inp_claim = claims[node.outputs[0]].last().unwrap();
          let auxs = &witnesses[node.inputs[0]];
          assert!(auxs.len() == 1, "two_pow expects 1 input");
          let aux_poly = auxs[0].data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
          let d = inp_claim.point.len();
          let mask = if d == 0 { 0 } else { (1usize << d) - 1 };
          let lagrange_basis = evaluate_lagrange_basis(&inp_claim.point);
          let mut part_evals = vec![<F as CryptoField>::zero(); 16];
          for (&full_idx, &val) in &aux_poly.evaluations {
            part_evals[full_idx >> d] += val * lagrange_basis[full_idx & mask];
          }
          let mid = (0..16).fold(<F as CryptoField>::zero(), |acc, idx| {
            acc + part_evals[idx] * F::from(1u32 << (15 - idx))
          });
          middle_claims[i].push(mid);
          part_evals_list.push((node.inputs[0], inp_claim.point.clone(), part_evals));
        }

        for mc in &middle_claims {
          transcript.append_scalar(b"two_pow_middle_claim", &mc[0]);
        }

        let alpha: F = transcript.challenge_scalar(b"two_pow_alpha");
        let beta: F = transcript.challenge_scalar(b"table_beta");
        let betas = calc_pow(beta, self.two_pow.len());

        let mut batched_evals = vec![<F as CryptoField>::zero(); 16];
        for (i, (_, _, part_evals)) in part_evals_list.iter().enumerate() {
          for k in 0..16 {
            if part_evals[k] != <F as CryptoField>::zero() {
              batched_evals[k] += betas[i] * part_evals[k];
            }
          }
        }
        let aux_poly = DenseMLPoly::new(4, batched_evals);
        let two_pow_poly = two_pow_dense::<F>(4).add_by_scalar(alpha);
        let mut sumcheck_prover = LinearSumcheckProver::<F>::new(4, 2, transcript);
        let sumcheck_proof = sumcheck_prover.prove(&vec![aux_poly, two_pow_poly], transcript);
        let r_tbl = &sumcheck_prover.challenges;
        let tbl_basis = evaluate_lagrange_basis(r_tbl);
        table_proofs.push(sumcheck_proof);

        for (aux_edge, inp_point, part_evals) in &part_evals_list {
          let eval = (0..16).fold(<F as CryptoField>::zero(), |acc, k| acc + part_evals[k] * tbl_basis[k]);
          let point: Vec<F> = inp_point.iter().chain(r_tbl.iter()).cloned().collect();
          transcript.append_scalar(b"two_pow_table_eval", &eval);
          claims[*aux_edge].push(Claim {
            edge_id: *aux_edge,
            sparse_id: 0,
            point,
            eval,
          });
        }

        // Emit booleanity proofs for standalone TwoPow selectors not already covered by `self.range`
        let mut uncovered_groups: BTreeMap<usize, Vec<(usize, NodeId)>> = BTreeMap::new();
        for (i, &n) in self.two_pow.iter().enumerate() {
          let aux_edge = self.nodes[n].inputs[0];
          let covered_by_range = self.producers[aux_edge].map_or(false, |p| {
            if !self.range.contains(&p) || witnesses[aux_edge].len() != 1 {
              return false;
            }
            let rnode = &self.nodes[p];
            let r_aux_id = if matches!(rnode.kind, BasicBlockType::NonNegative(_)) { 0 } else { 1 };
            rnode.outputs.get(r_aux_id) == Some(&aux_edge) && self.range_block_table_vars(p, 1, 0) == 4
          });
          if !covered_by_range {
            let aux_num_var = get_n(&witnesses[self.nodes[n].outputs[0]][0].shape) + 4;
            uncovered_groups.entry(aux_num_var).or_default().push((i, n));
          }
        }
        for (&aux_num_var, items) in &uncovered_groups {
          let mut bool_weights = Vec::with_capacity(items.len());
          let mut bool_entries = Vec::with_capacity(items.len());
          for &(i, n) in items {
            bool_weights.push(betas[i] * betas[i]);
            let aux_edge = self.nodes[n].inputs[0];
            let aux_poly = witnesses[aux_edge][0].data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly<F>>().unwrap();
            let entries: Vec<(usize, F)> = aux_poly.evaluations.iter().map(|(&idx, &val)| (idx, val)).collect();
            bool_entries.push(entries);
          }
          let mut bool_prover = SparseBoolSumcheckProver::<F>::new(aux_num_var, transcript);
          let eq_challenge: Vec<F> = (0..aux_num_var).map(|_| transcript.challenge_scalar(b"challenge")).collect();
          let bool_proof = bool_prover.prove_with_entries(&bool_weights, &bool_entries, &eq_challenge, transcript);
          bool_proofs.push(bool_proof);

          for (term_idx, &(_, n)) in items.iter().enumerate() {
            let aux_edge = self.nodes[n].inputs[0];
            let eval = bool_prover.term_final_eval(term_idx);
            transcript.append_scalar(b"two_pow_bool_eval", &eval);
            claims[aux_edge].push(Claim {
              edge_id: aux_edge,
              sparse_id: 0,
              point: bool_prover.challenges.clone(),
              eval,
            });
          }
        }
      });
    }
    (table_proofs, middle_claims, bool_proofs)
  }

  /* ---------- Alias helper ---------- */
  pub fn alias_info(&self, a: AliasId) -> (EdgeId, NodeId, usize) {
    (self.alias_to_edge[a.0], self.alias_to_consumer[a.0], self.alias_input_slot[a.0])
  }

  /// Pretty printer that shows NodeId, node kind, and both EdgeId and AliasId context with a post-order traversal.
  pub fn print_dag_structure(&self, targets: &[EdgeId]) {
    let (_need_edge, need_node) = self.backward_mark_needed(targets);
    let needed_out_aliases = self.collect_needed_out_aliases(&need_node);

    // Build initial successor alias counts
    let mut succ_alias_count: Vec<usize> = needed_out_aliases.iter().map(|v| v.len()).collect();

    // Initial queue: nodes with all needed outputs already reached (sinks).
    let mut q: Vec<NodeId> = (0..self.nodes.len()).filter(|&n| need_node[n] && succ_alias_count[n] == 0).collect();

    println!("=== reverse post-order (from targeted edges: #{:?}) ===", targets);

    while let Some(u) = q.pop() {
      // ---- Pretty print this node ----
      println!("node #{u}: {:?}", self.nodes[u].kind);

      // Outputs that contributed to the slice (original dependency set)
      if needed_out_aliases[u].is_empty() {
        println!("  outputs -> (targets or no downstream consumers)");
      } else {
        println!("  outputs:");
        for &(e, a) in &needed_out_aliases[u] {
          let (_pe, consumer, slot) = self.alias_info(a);
          println!("    edge #{e} via alias {:?} -> consumer node #{consumer} (input slot {slot})", a);
        }
      }

      // Inputs actually consumed to unlock predecessors
      if self.nodes[u].inputs.is_empty() {
        println!("  inputs: (none)");
      } else {
        println!("  inputs:");
        for (slot, &ie) in self.nodes[u].inputs.iter().enumerate() {
          if let Some(p) = self.producers[ie] {
            // find the exact alias from producer edge to this (u, slot)
            if let Some(&aid) = self.edge_aliases[ie].iter().find(|&&aid| {
              let (_e, c, s) = self.alias_info(aid);
              c == u && s == slot
            }) {
              println!("    edge #{ie}: alias {:?} from producer node #{p} (slot {slot})", aid);
            } else {
              println!("    edge #{ie}: from producer node #{p} (slot {slot})");
            }
          } else {
            if self.input_ports.contains(&ie) {
              println!("    edge #{ie}: GRAPH INPUT (slot {slot})");
            } else {
              println!("    edge #{ie}: CONSTANT (slot {slot})");
            }
          }
        }
      }

      // ---- Propagate to predecessors (decrement by matching aliases) ----
      for (slot, &ie) in self.nodes[u].inputs.iter().enumerate() {
        if let Some(p) = self.producers[ie] {
          if need_node[p] {
            let dec = self.edge_aliases[ie]
              .iter()
              .filter(|&&aid| {
                let (_e, c, s) = self.alias_info(aid);
                c == u && s == slot
              })
              .count();
            if dec > 0 {
              succ_alias_count[p] -= dec;
              if succ_alias_count[p] == 0 {
                q.push(p);
              }
            }
          }
        }
      }
    }

    println!("=== end reverse post-order ===");
  }

  // ------ small helpers reused by backward methods ------
  fn backward_mark_needed(&self, targets: &[EdgeId]) -> (Vec<bool>, Vec<bool>) {
    let mut need_edge = vec![false; self.num_edges];
    let mut need_node = vec![false; self.nodes.len()];
    let mut stack: Vec<EdgeId> = Vec::new();

    for &e in targets {
      if !need_edge[e] {
        need_edge[e] = true;
        stack.push(e);
      }
    }
    while let Some(e) = stack.pop() {
      if let Some(n) = self.producers[e] {
        if !need_node[n] {
          need_node[n] = true;
          for &ie in &self.nodes[n].inputs {
            if !need_edge[ie] {
              need_edge[ie] = true;
              stack.push(ie);
            }
          }
        }
      }
    }
    (need_edge, need_node)
  }

  fn collect_needed_out_aliases(&self, need_node: &[bool]) -> Vec<Vec<(EdgeId, AliasId)>> {
    let mut needed_out_aliases: Vec<Vec<(EdgeId, AliasId)>> = vec![Vec::new(); self.nodes.len()];
    for n in 0..self.nodes.len() {
      if !need_node[n] {
        continue;
      }
      for &e in &self.nodes[n].outputs {
        for &a in &self.edge_aliases[e] {
          let (_pe, consumer, _slot) = self.alias_info(a);
          if need_node[consumer] {
            needed_out_aliases[n].push((e, a));
          }
        }
      }
    }
    needed_out_aliases
  }
}
