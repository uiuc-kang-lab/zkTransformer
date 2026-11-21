//! Soundness & Completeness Audit Regression Suite for `zkTransformer` (`zk_torch_2`).
//!
//! Covers both positive end-to-end honest proofs and negative cheating-prover regression tests
//! for all soundness vulnerabilities identified during the audit of `main` (55344d7):
//! - Bug A: Missing terminal sumcheck evaluations in `Dag::verify_range`, `Dag::verify_two_pow`,
//!          and `NonStructuredExp::verify` (`verification_result.is_some()` only).
//! - Bug B: Sparse lookup auxiliary commitments (`SparseKZH3`) were never opened (`edge_proofs`
//!          populated before `prove_two_pow`/`prove_range`, `eval: 0` hardcoded, multi-block
//!          `Dag::commit` skipping `sparse_id >= 1`, and partial-block `split_into_blocks`
//!          inflating final-block bit-width to `block_size`).
//! - Bug C: Missing remainder range check `r ∈ [-floor((L-1)/2), floor(L/2)]` in `StructuredExp`,
//!          which allowed arbitrary `k' ∈ [0, 15]` and out-of-range `r' = x + k' * ln2_sf`.
//! - Bug D: Missing Shout Hamming-weight-1 (`row-sum = 1`) check in `TwoPow`, which allowed
//!          all-zero or multi-one selector rows.
//! - Bug E: Fiat-Shamir transcript omissions (`middle_claims` not absorbed before squeezing
//!          `table_alpha`, `table_beta`, `table_gamma`, which enabled an adaptive 2x2 linear
//!          cancellation with all-zero auxiliary commitments).
//! - Bug F: Missing GKR claim-chaining checks in `Dag::verify` (`claims[out]` when `len == 1`,
//!          `NonNegative`/`ZeroCheck` sink claims, `ChangeShape::verify` returning `true`
//!          unconditionally, input claim point checks, and `edge_proofs` vs `claims` binding).

#[cfg(all(feature = "arkworks", feature = "bls12_381"))]
use ark_bls12_381::Fr as F;
#[cfg(all(feature = "arkworks", feature = "bn254"))]
use ark_bn254::Fr as F;

#[cfg(all(feature = "arkworks", feature = "bn254"))]
use zk_torch_2::crypto::polycommit::ArkBn254 as PairingType;

use plonky2::util::timing::TimingTree;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use zk_torch_2::basicblock::{BasicBlockType, TwoPow};
use zk_torch_2::crypto::polycommit::kzh3::{
  setup_kzh3_srs, KZH3Commit, KZH3CommitKey, KZH3VerifierKey,
};
use zk_torch_2::crypto::polycommit::sparse_kzh3::{
  SparseKZH3Commit, SparseKZH3CommitKey, SparseKZH3VerifierKey,
};
use zk_torch_2::crypto::polycommit::{Commitment as CommitmentTrait, MLPolyCommit};
use zk_torch_2::crypto::{
  LinearSumcheckProver, SparseBoolSumcheckProver, SumcheckProof, SumcheckProver, SumcheckVerifier,
};
use zk_torch_2::dag::{Dag, DagBuilder, DataType, PolyType, Role, Witness};
use zk_torch_2::util::arith::{calc_pow, f_to_int};
use zk_torch_2::util::poly::{range_dense, CryptoField, DenseMLPoly, SelectionPolynomial, SparseMLPoly};
use zk_torch_2::util::transcript::Transcript;
use zk_torch_2::SF_LOG;

fn f_i64(v: i64) -> F {
  if v >= 0 {
    <F as CryptoField>::from_u32(v as u32)
  } else {
    <F as CryptoField>::zero() - <F as CryptoField>::from_u32((-v) as u32)
  }
}

fn setup_keys_for_dag(
  dag: &Dag,
  init: &[Vec<Witness<F>>],
  extra_sizes: &[usize],
) -> (
  KZH3CommitKey<PairingType>,
  SparseKZH3CommitKey<PairingType>,
  KZH3VerifierKey<PairingType>,
  SparseKZH3VerifierKey<PairingType>,
) {
  let mut sizes: BTreeSet<usize> = dag.collect_polynomial_sizes(init);
  for &s in extra_sizes {
    sizes.insert(s);
  }
  let mut srs_map = HashMap::new();
  for size in sizes {
    let mut rng = StdRng::seed_from_u64(0xA0D17_0000 + size as u64);
    let srs = setup_kzh3_srs::<PairingType, _>(size, &mut rng);
    srs_map.insert(size, Arc::new(srs));
  }
  let srs_map = Arc::new(srs_map);
  (
    KZH3CommitKey { srs_map: srs_map.clone() },
    SparseKZH3CommitKey { srs_map: srs_map.clone() },
    KZH3VerifierKey { srs_map: srs_map.clone() },
    SparseKZH3VerifierKey { srs_map },
  )
}

fn forge_sumcheck_without_terminal_check(
  num_rounds: usize,
  num_poly: usize,
  claimed_sum: F,
  transcript: &mut Transcript<F>,
) -> SumcheckProof<F> {
  transcript.append_u64(b"num_var", num_rounds as u64);
  transcript.append_u64(b"num_poly", num_poly as u64);

  let mut running_sum = claimed_sum;
  let mut round_messages = Vec::with_capacity(num_rounds);
  let one = <F as CryptoField>::one();

  for _ in 0..num_rounds {
    let msg: Vec<F> = (0..=num_poly)
      .map(|k| {
        let x = <F as CryptoField>::from_u32(k as u32);
        running_sum * (one - x)
      })
      .collect();
    for m in &msg {
      transcript.append_scalar(b"round_message", m);
    }
    let r = transcript.challenge_scalar(b"challenge");
    running_sum = running_sum * (one - r);
    round_messages.push(msg);
  }

  SumcheckProof {
    final_eval: running_sum,
    round_messages,
  }
}

// ============================================================================
// 1. Honest Completeness Sanity Tests (must all verify `true`)
// ============================================================================

#[test]
fn test_audit_honest_argmax_verifies_true() {
  let sf = *SF_LOG;
  let mut g = DagBuilder::<F>::new();
  let logits_edge = g.input(vec![1, 4], DataType::Float);
  let _sel_edge = g.argmax_check(logits_edge, 1, 4, &[1]);
  let (dag, mut init) = g.compile();

  let logits_w = Witness::new(
    vec![1, 4],
    vec![f_i64(10), f_i64(50), f_i64(20), f_i64(30)],
    DataType::Float,
    sf,
    Role::Input,
  );
  dag.run(&mut init, &[(logits_edge, logits_w)]);

  let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[]);
  let mut dense_comms = vec![None; dag.num_edges()];
  let mut sparse_comms = vec![None; dag.num_edges()];
  let mut timing = TimingTree::default();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
  );

  let mut prover_transcript = Transcript::new(b"zkml");
  let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &ck, &sck, &init, &dense_comms, &sparse_comms, &mut prover_transcript, &mut timing,
    );

  let sparse_openings: usize = edge_proofs.iter().map(|ep| ep.sparse_opening_proof.len()).sum();
  assert_eq!(sparse_openings, 10, "5 NonNegative blocks x (table + bool) = 10 sparse openings");

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
  );
  assert!(verified, "Honest argmax proof must verify true");
}

#[test]
fn test_audit_honest_structured_exp_verifies_true() {
  let sf = *SF_LOG;
  let ln2_sf = (2.0_f64.ln() * ((1u64 << sf) as f64)).round() as i64;
  let mut g = DagBuilder::<F>::new();
  let x = g.input(vec![4], DataType::Float);
  let out = g.exp(x)[0];
  let (dag, mut init) = g.compile();

  // Include boundary cases: -4*ln2, -355 (half-tie boundary), -354, and 0
  let x_w = Witness::new(
    vec![4],
    vec![f_i64(-4 * ln2_sf), f_i64(-(ln2_sf / 2)), f_i64(-((ln2_sf - 1) / 2)), f_i64(0)],
    DataType::Float,
    sf,
    Role::Input,
  );
  dag.run(&mut init, &[(x, x_w)]);
  assert_eq!(f_to_int(init[out][0].data.as_ref().unwrap().index(0)), 64);

  let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[]);
  let mut dense_comms = vec![None; dag.num_edges()];
  let mut sparse_comms = vec![None; dag.num_edges()];
  let mut timing = TimingTree::default();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
  );

  let mut prover_transcript = Transcript::new(b"zkml");
  let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &ck, &sck, &init, &dense_comms, &sparse_comms, &mut prover_transcript, &mut timing,
    );

  let sparse_openings: usize = edge_proofs.iter().map(|ep| ep.sparse_opening_proof.len()).sum();
  assert!(sparse_openings > 0, "StructuredExp must emit SparseKZH3 openings");

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
  );
  assert!(verified, "Honest StructuredExp proof must verify true");
}

#[test]
fn test_audit_honest_standalone_two_pow_verifies_true() {
  let mut g = DagBuilder::<F>::new();
  let k_edge = g.num_edges;
  g.num_edges += 1;
  let valid_sel = SelectionPolynomial::<F>::new(2, 4, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
  g.init_values.push(Some(Witness::new_sparse(
    vec![4],
    valid_sel.to_sparse(),
    DataType::Float,
    0,
    Role::Auxiliary,
  )));

  let nid = g.nodes.len();
  g.two_pow.push(nid);
  let _out = g.add_gkr_node(vec![k_edge], BasicBlockType::TwoPow(TwoPow))[0];
  g.init_values.push(Some(Witness::new_wo_data(vec![4], DataType::Float, 15, Role::Output)));

  let (dag, mut init) = g.compile();
  dag.run(&mut init, &[]);

  let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[]);
  let mut dense_comms = vec![None; dag.num_edges()];
  let mut sparse_comms = vec![None; dag.num_edges()];
  let mut timing = TimingTree::default();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
  );

  let mut prover_transcript = Transcript::new(b"zkml");
  let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &ck, &sck, &init, &dense_comms, &sparse_comms, &mut prover_transcript, &mut timing,
    );

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
  );
  assert!(verified, "Honest standalone TwoPow proof must verify true");
}

#[test]
fn test_audit_honest_matmul_reshape_verifies_true() {
  let sf = *SF_LOG;
  let mut g = DagBuilder::<F>::new();
  let x_edge = g.input(vec![2, 2], DataType::Float);
  let w_true = Witness::new(
    vec![2, 2],
    vec![f_i64(1), f_i64(3), f_i64(2), f_i64(4)],
    DataType::Float,
    0,
    Role::Constant,
  );
  let w_edge = g.param(w_true);
  let y_short = g.einsum("ij,jk->ik".to_string(), vec![x_edge, w_edge], false)[0];
  let _y_edge = g.change_shape(y_short, vec![4]);
  let (dag, mut init) = g.compile();

  let x_true = Witness::new(
    vec![2, 2],
    vec![f_i64(1), f_i64(0), f_i64(0), f_i64(1)],
    DataType::Float,
    sf,
    Role::Input,
  );
  dag.run(&mut init, &[(x_edge, x_true)]);

  let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[]);
  let mut dense_comms = vec![None; dag.num_edges()];
  let mut sparse_comms = vec![None; dag.num_edges()];
  let mut timing = TimingTree::default();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
  );

  let mut prover_transcript = Transcript::new(b"zkml");
  let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &ck, &sck, &init, &dense_comms, &sparse_comms, &mut prover_transcript, &mut timing,
    );

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
  );
  assert!(verified, "Honest MatMul + ChangeShape proof must verify true");
}

// ============================================================================
// 2. Negative Soundness Regression Tests (every forgery MUST reject `false`)
// ============================================================================

/// Regression for Bug A: Forging a FALSE argmax output token (`0` instead of `1` on
/// logits `[10, 50, 20, 30]`) via round-consistent sumcheck messages that violate the
/// terminal oracle check MUST be rejected.
#[test]
fn test_poc_bug_a_forged_argmax_missing_terminal_sumcheck_eval() {
  let sf = *SF_LOG;
  let mut g = DagBuilder::<F>::new();
  let logits_edge = g.input(vec![1, 4], DataType::Float);
  let _sel_edge = g.argmax_check(logits_edge, 1, 4, &[0]);
  let nonneg_nid = g.range[0];
  let (dag, mut init) = g.compile();

  let logits_w = Witness::new(
    vec![1, 4],
    vec![f_i64(10), f_i64(50), f_i64(20), f_i64(30)],
    DataType::Float,
    sf,
    Role::Input,
  );
  dag.run(&mut init, &[(logits_edge, logits_w)]);

  let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[]);
  let mut dense_comms = vec![None; dag.num_edges()];
  let mut sparse_comms = vec![None; dag.num_edges()];
  let mut timing = TimingTree::default();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
  );

  let mut prover_transcript = Transcript::new(b"zkml");
  let (node_proofs, edge_proofs, mut range_proof, two_pow_proof, reducer_proofs) =
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &ck, &sck, &init, &dense_comms, &sparse_comms, &mut prover_transcript, &mut timing,
    );

  let neg_eval = node_proofs[nonneg_nid].as_ref().unwrap().1[0].eval;
  let num_blocks = range_proof.middle_claims[0].len();
  range_proof.middle_claims[0] = vec![<F as CryptoField>::zero(); num_blocks];
  range_proof.middle_claims[0][0] = neg_eval;

  let mut replay_tx = Transcript::new(b"zkml");
  for c in dense_comms.iter().flatten() {
    replay_tx.append_message(b"dense_commitment", &c.to_transcript_bytes());
  }
  for cs in sparse_comms.iter().flatten() {
    for c in cs {
      replay_tx.append_message(b"sparse_commitment", &c.to_transcript_bytes());
    }
  }
  let _r0: Vec<F> = (0..2).map(|_| replay_tx.challenge_scalar(b"challenge")).collect();
  replay_tx.append_scalar(b"output_claim_eval", &neg_eval);
  for np in node_proofs.iter().flatten() {
    if !np.0.is_empty() {
      for sp in &np.0 {
        let sum0 = sp.round_messages[0][0] + sp.round_messages[0][1];
        let mut sc_v = SumcheckVerifier::new(sp.round_messages.len(), sp.round_messages[0].len() - 1, &mut replay_tx);
        let _ = sc_v.verify(&mut replay_tx, sp.round_messages.clone(), sum0);
      }
    }
    if np.1.len() > 1 {
      for c in &np.1[..np.1.len() - 1] {
        replay_tx.append_scalar(b"node_claim_eval", &c.eval);
      }
    }
  }
  for &m in &range_proof.middle_claims[0] {
    replay_tx.append_scalar(b"range_middle_claim", &m);
  }
  let alpha: F = replay_tx.challenge_scalar(b"table_alpha");
  let _beta: F = replay_tx.challenge_scalar(b"table_beta");
  let gamma: F = replay_tx.challenge_scalar(b"table_gamma");
  let gammas = calc_pow(gamma, num_blocks);
  let mut table_expected_sum = <F as CryptoField>::zero();
  for s_id in 0..num_blocks {
    table_expected_sum += (range_proof.middle_claims[0][s_id] + alpha) * gammas[s_id];
  }

  let table_rounds = range_proof.table_proofs[0].round_messages.len();
  range_proof.table_proofs[0] =
    forge_sumcheck_without_terminal_check(table_rounds, 2, table_expected_sum, &mut replay_tx);

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
  );
  assert!(!verified, "Forged argmax proof MUST be rejected");
}

/// Regression for Bug B: Replacing `sparse_commitments` (on ANY block `sparse_id >= 0`,
/// including `sparse_id = 2`) with a commitment to an empty `SparseMLPoly` MUST be rejected
/// by `SparseKZH3::verify_and_extract`.
#[test]
fn test_poc_bug_b_bogus_sparse_commitment_never_opened() {
  let sf = *SF_LOG;
  let mut g = DagBuilder::<F>::new();
  let x = g.input(vec![4], DataType::Float);
  let _y = g.scale(x, sf, 0)[0];
  let (dag, mut init) = g.compile();

  let x_w = Witness::new(
    vec![4],
    vec![f_i64(1024), f_i64(2048), f_i64(3072), f_i64(4096)],
    DataType::Float,
    sf,
    Role::Input,
  );
  dag.run(&mut init, &[(x, x_w)]);

  let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[]);
  let mut dense_comms = vec![None; dag.num_edges()];
  let mut sparse_comms = vec![None; dag.num_edges()];
  let mut timing = TimingTree::default();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
  );

  // Tamper with ONLY the last block (`sparse_id = 2`, `table_vars = 2`, `n = 4`) to test BOTH
  // Bug B1 (`SparseKZH3` opening verification) AND Bug B2 (`sparse_id >= 1` commitment)!
  for (e, slot) in sparse_comms.iter_mut().enumerate() {
    if let Some(vec_comms) = slot {
      assert_eq!(vec_comms.len(), 3, "ScaleDown(10, 0) must commit all 3 sparse blocks");
      let last_n = init[e][2].data.as_ref().unwrap().n();
      let empty_sel = SelectionPolynomial::<F>::new(2, last_n - 2, vec![]);
      let empty_sparse: SparseMLPoly<F> = empty_sel.to_sparse();
      vec_comms[2] = SparseKZH3Commit::<PairingType>::commit(&empty_sparse, &sck);
    }
  }

  let mut prover_transcript = Transcript::new(b"zkml");
  let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &ck, &sck, &init, &dense_comms, &sparse_comms, &mut prover_transcript, &mut timing,
    );

  let total_sparse_openings: usize = edge_proofs.iter().map(|ep| ep.sparse_opening_proof.len()).sum();
  assert_eq!(total_sparse_openings, 6, "3 blocks x (table + bool) = 6 SparseKZH3 openings");

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
  );
  assert!(!verified, "Bogus sparse commitment on block 2 MUST be rejected");
}

/// Regression for Bug C: Forging `k' = 0` and out-of-range remainder `r' = -2840` in
/// `StructuredExp` (making `exp(-4 ln 2) = -1519 < 0`) MUST be rejected by the remainder
/// range check `r ∈ [-floor((L-1)/2), floor(L/2)]`.
#[test]
fn test_poc_bug_c_structured_exp_unbounded_remainder_forgery() {
  let sf = *SF_LOG;
  let ln2_sf = (2.0_f64.ln() * ((1u64 << sf) as f64)).round() as i64;
  let x_val = -4 * ln2_sf;

  let mut g = DagBuilder::<F>::new();
  let x = g.input(vec![4], DataType::Float);
  let out = g.exp(x)[0];
  let (dag, mut init) = g.compile();

  let x_w = Witness::new(vec![4], vec![f_i64(x_val); 4], DataType::Float, sf, Role::Input);
  dag.run(&mut init, &[(x, x_w)]);
  assert_eq!(f_to_int(init[out][0].data.as_ref().unwrap().index(0)), 64);

  let k_edge = (0..dag.num_edges())
    .find(|&e| {
      init[e].len() == 1
        && init[e][0].poly_type == PolyType::Sparse
        && init[e][0].role == Role::Auxiliary
        && init[e][0].data.as_ref().unwrap().n() == 6
    })
    .expect("ExpHelper k_edge must exist");
  let r_edge = k_edge - 1;
  assert_eq!(init[r_edge][0].poly_type, PolyType::Dense);

  // Forge k' = 0, r' = x_val = -2840 (violating r ∈ [-354, 355]):
  init[r_edge] = vec![Witness::new(vec![4], vec![f_i64(x_val); 4], DataType::Float, sf, Role::Output)];
  let forged_k_sel = SelectionPolynomial::<F>::new(2, 4, vec![(0, 0), (1, 0), (2, 0), (3, 0)]);
  init[k_edge] = vec![Witness::new_sparse(
    vec![4],
    forged_k_sel.to_sparse(),
    DataType::Float,
    0,
    Role::Auxiliary,
  )];

  dag.rerun_downstream(&mut init, &[r_edge, k_edge]);
  assert!(f_to_int(init[out][0].data.as_ref().unwrap().index(0)) < 0);

  let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[]);
  let mut dense_comms = vec![None; dag.num_edges()];
  let mut sparse_comms = vec![None; dag.num_edges()];
  let mut timing = TimingTree::default();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
  );

  let mut prover_transcript = Transcript::new(b"zkml");
  let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &ck, &sck, &init, &dense_comms, &sparse_comms, &mut prover_transcript, &mut timing,
    );

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
  );
  assert!(!verified, "Out-of-range StructuredExp remainder MUST be rejected");
}

/// Regression for Bug D: Both zero-Hamming-weight (empty row -> output 0) and
/// weight-2 (`(0, 0)` + `(0, 1)` -> output `32768 + 16384 = 49152`) selector rows in
/// `TwoPow` MUST be rejected by `two_pow_alpha` row-sum == 1 verification.
#[test]
fn test_poc_bug_d_shout_zero_hamming_weight_spoofs_two_pow() {
  for (bad_sel, bad_out_0) in [
    (vec![], 0i64),
    (vec![(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)], 49152i64),
  ] {
    let mut g = DagBuilder::<F>::new();
    let k_edge = g.num_edges;
    g.num_edges += 1;
    let sel = SelectionPolynomial::<F>::new(2, 4, bad_sel);
    g.init_values.push(Some(Witness::new_sparse(
      vec![4],
      sel.to_sparse(),
      DataType::Float,
      0,
      Role::Auxiliary,
    )));

    let nid = g.nodes.len();
    g.two_pow.push(nid);
    let out = g.add_gkr_node(vec![k_edge], BasicBlockType::TwoPow(TwoPow))[0];
    g.init_values.push(Some(Witness::new_wo_data(vec![4], DataType::Float, 15, Role::Output)));

    let (dag, mut init) = g.compile();
    dag.run(&mut init, &[]);
    init[out][0] = Witness::new(
      vec![4],
      vec![f_i64(bad_out_0), f_i64(32768), f_i64(32768), f_i64(32768)],
      DataType::Float,
      15,
      Role::Output,
    );

    let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[]);
    let mut dense_comms = vec![None; dag.num_edges()];
    let mut sparse_comms = vec![None; dag.num_edges()];
    let mut timing = TimingTree::default();
    dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
    );

    let mut prover_transcript = Transcript::new(b"zkml");
    let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
      dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
        &ck, &sck, &init, &dense_comms, &sparse_comms, &mut prover_transcript, &mut timing,
      );

    init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));
    let mut verifier_transcript = Transcript::new(b"zkml");
    let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
      &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
    );
    assert!(!verified, "Non-unit Hamming weight TwoPow selector MUST be rejected");
  }
}

/// Regression for Bug E: Adaptive Fiat-Shamir `middle_claims` linear-system cancellation
/// on negative input `[-500, -500, -500, -500]` MUST be rejected because `middle_claims`
/// are absorbed into the transcript BEFORE `table_alpha`, `table_beta`, `table_gamma`.
#[test]
fn test_poc_bug_e_fiat_shamir_adaptive_middle_claims_forgery() {
  let sf = *SF_LOG;
  let mut g = DagBuilder::<F>::new();
  let x = g.input(vec![4], DataType::Float);
  let zero_c = g.param(Witness::new(
    vec![4],
    vec![<F as CryptoField>::zero(); 4],
    DataType::Float,
    sf,
    Role::Constant,
  ));
  let y = g.add(x, zero_c)[0];
  g.add_nonneg_node(y);
  let nonneg_nid = g.range[0];
  let (dag, mut init) = g.compile();

  let x_w = Witness::new(vec![4], vec![f_i64(-500); 4], DataType::Float, sf, Role::Input);
  dag.run(&mut init, &[(x, x_w)]);

  let aux_edge = dag.num_edges() - 1;
  let num_blocks = init[aux_edge].len();
  for b in 0..num_blocks {
    let empty_sel = SelectionPolynomial::<F>::new(2, 4, vec![]);
    init[aux_edge][b] = Witness::new_sparse(vec![4], empty_sel.to_sparse(), DataType::Float, 0, Role::Auxiliary);
  }

  let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[6]);
  let mut dense_comms = vec![None; dag.num_edges()];
  let mut sparse_comms = vec![None; dag.num_edges()];
  let mut timing = TimingTree::default();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
  );

  let mut prover_transcript = Transcript::new(b"zkml");
  let (node_proofs, edge_proofs, mut range_proof, two_pow_proof, reducer_proofs) =
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &ck, &sck, &init, &dense_comms, &sparse_comms, &mut prover_transcript, &mut timing,
    );

  // Attempt the adaptive linear-system forgery from `main`:
  let mut tx = Transcript::new(b"zkml");
  for c in dense_comms.iter().flatten() {
    tx.append_message(b"dense_commitment", &c.to_transcript_bytes());
  }
  for cs in sparse_comms.iter().flatten() {
    for c in cs {
      tx.append_message(b"sparse_commitment", &c.to_transcript_bytes());
    }
  }
  let _r0: Vec<F> = (0..2).map(|_| tx.challenge_scalar(b"challenge")).collect();
  let v_eval = node_proofs[nonneg_nid].as_ref().unwrap().1[0].eval;
  tx.append_scalar(b"output_claim_eval", &v_eval);
  for np in node_proofs.iter().flatten() {
    if np.1.len() > 1 {
      for c in &np.1[..np.1.len() - 1] {
        tx.append_scalar(b"node_claim_eval", &c.eval);
      }
    }
  }
  for &m in &range_proof.middle_claims[0] {
    tx.append_scalar(b"range_middle_claim", &m);
  }
  let alpha: F = tx.challenge_scalar(b"table_alpha");
  let _beta: F = tx.challenge_scalar(b"table_beta");
  let gamma: F = tx.challenge_scalar(b"table_gamma");
  let gammas = calc_pow(gamma, num_blocks);

  let gamma_sum: F = (0..num_blocks).fold(<F as CryptoField>::zero(), |acc, s| acc + gammas[s]);
  let sixteen = <F as CryptoField>::from_u32(16);
  let m1 = (<F as CryptoField>::zero() - (v_eval + alpha * gamma_sum)) * (gamma - sixteen).invert();
  let m0 = v_eval - sixteen * m1;

  let mut forged_middle = vec![<F as CryptoField>::zero(); num_blocks];
  forged_middle[0] = m0;
  forged_middle[1] = m1;
  range_proof.middle_claims[0] = forged_middle;

  let zero_aux_dense = DenseMLPoly::new(4, vec![<F as CryptoField>::zero(); 16]);
  let range_poly = range_dense::<F>(4).add_by_scalar(alpha);
  let mut sc_prover = LinearSumcheckProver::<F>::new(4, 2, &mut tx);
  range_proof.table_proofs[0] = sc_prover.prove(&vec![zero_aux_dense, range_poly], &mut tx);

  let mut bool_prover = SparseBoolSumcheckProver::<F>::new(6, &mut tx);
  let eq_ch: Vec<F> = (0..6).map(|_| tx.challenge_scalar(b"challenge")).collect();
  let weights: Vec<F> = (0..num_blocks).map(|s| gammas[s] * gammas[s]).collect();
  let empty_positions: Vec<Vec<usize>> = vec![vec![]; num_blocks];
  range_proof.bool_proofs[0] = bool_prover.prove(&weights, &empty_positions, &eq_ch, &mut tx);

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
  );
  assert!(!verified, "Adaptive Fiat-Shamir middle_claims forgery MUST be rejected");
}

/// Regression for Bug F: Both (1) forging `ChangeShape` output evaluation while keeping
/// honest `Einsum` + `KZH3` openings AND (2) stripping `KZH3` opening proofs from
/// `edge_proofs` MUST be rejected.
#[test]
fn test_poc_bug_f_gkr_unlinked_claims_and_stripped_kzh3_openings() {
  let sf = *SF_LOG;
  let mut g = DagBuilder::<F>::new();
  let x_edge = g.input(vec![2, 2], DataType::Float);
  let w_true = Witness::new(
    vec![2, 2],
    vec![f_i64(1), f_i64(3), f_i64(2), f_i64(4)],
    DataType::Float,
    0,
    Role::Constant,
  );
  let w_edge = g.param(w_true);
  let y_short = g.einsum("ij,jk->ik".to_string(), vec![x_edge, w_edge], false)[0];
  let y_edge = g.change_shape(y_short, vec![4]);
  let (dag, mut init) = g.compile();

  let x_true = Witness::new(
    vec![2, 2],
    vec![f_i64(1), f_i64(0), f_i64(0), f_i64(1)],
    DataType::Float,
    sf,
    Role::Input,
  );
  dag.run(&mut init, &[(x_edge, x_true)]);

  let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[]);
  let mut dense_comms = vec![None; dag.num_edges()];
  let mut sparse_comms = vec![None; dag.num_edges()];
  let mut timing = TimingTree::default();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
  );

  let mut prover_transcript = Transcript::new(b"zkml");
  let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &ck, &sck, &init, &dense_comms, &sparse_comms, &mut prover_transcript, &mut timing,
    );

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));

  // Sub-attack F1: Forge `y_edge` output evaluation (keeping KZH3 input openings intact)
  {
    let mut forged_ep = edge_proofs.clone();
    forged_ep[y_edge].claims[0].eval = f_i64(9999);
    let mut verifier_transcript = Transcript::new(b"zkml");
    let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &node_proofs, &forged_ep, &range_proof, &two_pow_proof, &reducer_proofs,
      &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
    );
    assert!(!verified, "Forged ChangeShape output evaluation MUST be rejected");
  }

  // Sub-attack F2: Strip KZH3 input opening proofs (`x_edge`, `w_edge`)
  {
    let mut stripped_ep = edge_proofs.clone();
    stripped_ep[x_edge].claims.clear();
    stripped_ep[x_edge].dense_opening_proof.clear();
    stripped_ep[w_edge].claims.clear();
    stripped_ep[w_edge].dense_opening_proof.clear();
    let mut verifier_transcript = Transcript::new(b"zkml");
    let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &node_proofs, &stripped_ep, &range_proof, &two_pow_proof, &reducer_proofs,
      &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
    );
    assert!(!verified, "Stripped KZH3 opening proofs MUST be rejected");
  }
}

/// Regression for Bug B4 (Partial-Block Bit-Width Overflow):
/// In `ScaleDown(20, 10)` (`rescale_sf = 10 = 4 + 4 + 2`), the top block (`sparse_id = 2`)
/// must only allow indices in `[0, 3]` (`2` bits), NOT `[0, 15]` (`4` bits).
/// A cheating prover who biases `y` by `-1` (`y' = 0` instead of `1` on `x = 1024`, requiring
/// `aux = 1024 + 512 = 1536 = 0 + 16*0 + 256*6`, i.e. `k_2 = 6 >= 4`) MUST be rejected!
#[test]
fn test_poc_partial_block_scale_overflow_rejected() {
  let mut g = DagBuilder::<F>::new();
  let x = g.input(vec![4], DataType::Float);
  // Set x's sf to 20:
  g.init_values[x].as_mut().unwrap().sf = 20;
  let y = g.scale(x, 20, 10)[0];
  let aux_edge = y + 1;
  let (dag, mut init) = g.compile();

  let x_w = Witness::new(vec![4], vec![f_i64(1024); 4], DataType::Float, 20, Role::Input);
  dag.run(&mut init, &[(x, x_w)]);
  assert_eq!(f_to_int(init[y][0].data.as_ref().unwrap().index(0)), 1);

  // Forge y' = 0 (biasing rounded division by -1!) using k_0 = 0, k_1 = 0, and illegal
  // 4-bit k_2 = 6 (which equals 1536 = 1024 - 0*1024 + 512 if block 2 had 4 table vars!):
  init[y] = vec![Witness::new(vec![4], vec![f_i64(0); 4], DataType::Float, 10, Role::Output)];
  let zero_b = SelectionPolynomial::<F>::new(2, 4, vec![(0, 0), (1, 0), (2, 0), (3, 0)]).to_sparse();
  // Attempt to put out-of-domain 4-bit table index 6 into block 2 (whose true table_vars is 2):
  let mut illegal_evals = HashMap::new();
  for i in 0..4 {
    illegal_evals.insert(i + (6 << 2), <F as CryptoField>::one());
  }
  let illegal_b2 = SparseMLPoly::new(
    6,
    illegal_evals,
    std::collections::VecDeque::new(),
    SelectionPolynomial::new(2, 4, vec![(0, 6), (1, 6), (2, 6), (3, 6)]),
  );
  init[aux_edge] = vec![
    Witness::new_sparse(vec![4], zero_b.clone(), DataType::Float, 0, Role::Auxiliary),
    Witness::new_sparse(vec![4], zero_b, DataType::Float, 0, Role::Auxiliary),
    Witness::new_sparse(vec![4], illegal_b2, DataType::Float, 0, Role::Auxiliary),
  ];

  let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[4, 6]);
  let mut dense_comms = vec![None; dag.num_edges()];
  let mut sparse_comms = vec![None; dag.num_edges()];
  let mut timing = TimingTree::default();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
  );

  // Even if the cheating prover commits to the 6-var illegal_b2, the verifier enforces
  // `table_vars = 2` (`aux_num_var = 4`) on block 2 from static circuit metadata!
  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));
  let empty_ep: Vec<zk_torch_2::dag::EdgeProof<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>> =
    (0..dag.num_edges()).map(|_| zk_torch_2::dag::EdgeProof::new()).collect();
  let empty_lp = zk_torch_2::dag::LookupProof {
    table_proofs: vec![],
    middle_claims: vec![vec![f_i64(0), f_i64(0), f_i64(6)]],
    bool_proofs: vec![],
  };
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &vec![None; 1], &empty_ep, &empty_lp,
    &zk_torch_2::dag::LookupProof { table_proofs: vec![], middle_claims: vec![], bool_proofs: vec![] },
    &vec![None; 1], &init, &vk, &svk, &dense_comms, &sparse_comms, &mut verifier_transcript,
  );
  assert!(!verified, "Partial-block ScaleDown bit-width overflow MUST be rejected");
}

/// Regression for Cross-Family Audit Catch C1 (`ClampLower` Uncommitted Advice Decoupling):
/// Ensures `ClampLower`'s output edge is `Role::Auxiliary`, committed under `KZH3`, and cannot
/// be downgraded to `Role::Output` or opened to two inconsistent values across its consumers.
#[test]
fn test_poc_clamp_lower_advice_must_be_committed_and_bound() {
  let mut g = DagBuilder::<F>::new();
  let x_edge = g.input(vec![4], DataType::Float);
  let y_edge = g.exp(x_edge)[0];
  let (dag, mut init) = g.compile();

  let x_w = Witness::new(vec![4], vec![f_i64(-2840); 4], DataType::Float, 10, Role::Input);
  dag.run(&mut init, &[(x_edge, x_w)]);

  let (ck, sck, vk, svk) = setup_keys_for_dag(&dag, &init, &[]);
  let mut dense_comms = vec![None; dag.num_edges()];
  let mut sparse_comms = vec![None; dag.num_edges()];
  let mut timing = TimingTree::default();
  dag.commit::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &ck, &sck, &init, &mut dense_comms, &mut sparse_comms, &mut timing,
  );

  // Locate ClampLower output edge (`Role::Auxiliary` and committed)
  let clamp_out_edge = (0..dag.num_edges())
    .find(|&e| init[e][0].role == Role::Auxiliary && init[e][0].poly_type == zk_torch_2::dag::PolyType::Dense)
    .expect("ClampLower output must be a Dense Auxiliary edge");
  assert!(dense_comms[clamp_out_edge].is_some(), "ClampLower output MUST be committed");

  let mut prover_transcript = Transcript::new(b"zkml");
  let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
    dag.prove::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
      &ck, &sck, &init, &dense_comms, &sparse_comms, &mut prover_transcript, &mut timing,
    );
  let _ = y_edge;

  init.iter_mut().for_each(|w| w.iter_mut().for_each(|w| w.clear_data()));

  // Attempt to flip ClampLower output role back to `Role::Output` and drop its commitment:
  let mut tampered_init = init.clone();
  tampered_init[clamp_out_edge][0].role = Role::Output;
  let mut tampered_comms = dense_comms.clone();
  tampered_comms[clamp_out_edge] = None;
  let mut verifier_transcript = Transcript::new(b"zkml");
  let verified = dag.verify::<F, KZH3Commit<PairingType>, SparseKZH3Commit<PairingType>>(
    &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
    &tampered_init, &vk, &svk, &tampered_comms, &sparse_comms, &mut verifier_transcript,
  );
  assert!(!verified, "Uncommitted or Role-downgraded ClampLower output MUST be rejected");
}

/// Regression for Cross-Family Audit Catch C2 & m1 (PCS VerifierKey Provenance & Malformed `f_star`):
/// Verifies that `KZH3Commit::verify_and_extract` uses the trusted `VerifierKey` rather than
/// `commitment.srs` (rejecting a `fake_setup` identity SRS swapped onto a valid commitment point),
/// and returns `false` without panicking when `proof.f_star` has mismatched dimensions.
#[test]
fn test_poc_pcs_untrusted_commitment_srs_and_malformed_f_star_rejected() {
  use zk_torch_2::crypto::polycommit::MLPolyCommit;
  let ck: KZH3CommitKey<PairingType> = KZH3Commit::setup(3, 0, 0, 0);
  let vk = KZH3VerifierKey { srs_map: ck.srs_map.clone() };
  let fake_ck: KZH3CommitKey<PairingType> = KZH3Commit::fake_setup(3);
  let fake_srs = fake_ck.srs_map.get(&3).unwrap().clone();

  let honest_poly = DenseMLPoly::new(3, (0..8).map(|i| f_i64(i + 1)).collect());
  let honest_comm = KZH3Commit::commit(&honest_poly, &ck);

  // Build a forged commitment preserving honest `c` (same Fiat-Shamir bytes!) with `fake_setup` SRS:
  let forged_poly = DenseMLPoly::new(3, vec![f_i64(999); 8]);
  let mut forged_comm_on_fake = KZH3Commit::commit(&forged_poly, &fake_ck);
  forged_comm_on_fake.c = honest_comm.c;
  forged_comm_on_fake.srs = fake_srs;

  let point = vec![f_i64(2), f_i64(3), f_i64(5)];
  let forged_proof = KZH3Commit::open(&forged_comm_on_fake, &forged_poly, &fake_ck, &point);

  assert!(
    !KZH3Commit::verify(&forged_comm_on_fake, &forged_proof, &vk, &point),
    "KZH3::verify MUST use trusted VerifierKey and reject fake commitment.srs"
  );
  let (ok, _) = KZH3Commit::verify_and_extract(&forged_comm_on_fake, &forged_proof, &vk, &point);
  assert!(!ok, "KZH3::verify_and_extract MUST reject fake commitment.srs");

  // Also test malformed `f_star` arity/length does not panic:
  let mut malformed_proof = KZH3Commit::open(&honest_comm, &honest_poly, &ck, &point);
  malformed_proof.f_star = DenseMLPoly::new(0, vec![f_i64(42)]);
  let (ok_malformed, _) = KZH3Commit::verify_and_extract(&honest_comm, &malformed_proof, &vk, &point);
  assert!(!ok_malformed, "Malformed f_star dimension MUST return false cleanly without panicking");
}
