// Field type selection based on backend and curve
#[cfg(all(feature = "arkworks", feature = "bls12_381"))]
use ark_bls12_381::Fr as F;
#[cfg(all(feature = "arkworks", feature = "bn254"))]
use ark_bn254::Fr as F;

#[cfg(all(feature = "icicle", feature = "bls12_381"))]
use icicle_bls12_381::curve::ScalarField as F;
#[cfg(all(feature = "icicle", feature = "bn254"))]
use icicle_bn254::curve::ScalarField as F;
#[cfg(all(feature = "icicle", feature = "goldilocks"))]
use icicle_goldilocks::field::ScalarField as F;

use zk_torch_2::crypto::{SparseBoolSumcheckProver, SumcheckVerifier};
use zk_torch_2::util::poly::{
  evaluate_range_structured, evaluate_two_pow_structured, range_dense, two_pow_dense,
  CryptoField, MLPoly, SelectionPolynomial,
};
use zk_torch_2::util::transcript::Transcript;

#[test]
fn test_structured_tables_match_dense_mle() {
  let r4 = vec![
    <F as CryptoField>::from_u32(3),
    <F as CryptoField>::from_u32(7),
    <F as CryptoField>::from_u32(11),
    <F as CryptoField>::from_u32(13),
  ];
  assert_eq!(evaluate_range_structured(&r4), range_dense::<F>(4).evaluate_at_point(&r4));
  assert_eq!(evaluate_two_pow_structured(&r4), two_pow_dense::<F>(4).evaluate_at_point(&r4));

  let r2 = vec![<F as CryptoField>::from_u32(5), <F as CryptoField>::from_u32(9)];
  assert_eq!(evaluate_range_structured(&r2), range_dense::<F>(2).evaluate_at_point(&r2));
}

#[test]
fn test_sparse_bool_sumcheck_completeness_and_terminal_eval() {
  let sel = SelectionPolynomial::<F>::new(2, 4, vec![(0, 3), (1, 7), (2, 0), (3, 15)]).to_sparse();
  let entries: Vec<(usize, F)> = sel.evaluations.iter().map(|(&k, &v)| (k, v)).collect();
  let w = <F as CryptoField>::from_u32(19);

  let mut p_tx = Transcript::new(b"sparse_bool_test");
  let mut prover = SparseBoolSumcheckProver::<F>::new(6, &mut p_tx);
  let eq_ch: Vec<F> = (0..6).map(|_| p_tx.challenge_scalar(b"challenge")).collect();
  let proof = prover.prove_with_entries(&[w], &[entries], &eq_ch, &mut p_tx);

  let mut v_tx = Transcript::new(b"sparse_bool_test");
  let mut verifier = SumcheckVerifier::new(6, 3, &mut v_tx);
  let v_eq_ch: Vec<F> = (0..6).map(|_| v_tx.challenge_scalar(b"challenge")).collect();
  let (res, r_bool) = verifier.verify(&mut v_tx, proof.round_messages, <F as CryptoField>::zero());
  let running_sum = res.expect("SparseBoolSumcheckProver honest proof must verify");

  let term_eval = prover.term_final_eval(0);
  assert_eq!(term_eval, sel.evaluate_at_point(&r_bool));

  let one = <F as CryptoField>::one();
  let eq_eval = v_eq_ch
    .iter()
    .zip(r_bool.iter())
    .fold(one, |acc, (&ec, &rb)| acc * (rb * ec + (one - rb) * (one - ec)));
  assert_eq!(running_sum, eq_eval * w * term_eval * (term_eval - one));
}
