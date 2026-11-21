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

use std::collections::{HashMap, VecDeque};
use zk_torch_2::util::poly::CryptoField;
use zk_torch_2::util::poly::{fix_variables_from_right, DenseMLPoly, SelectionPolynomial, SparseMLPoly};

fn make_sparse(n: usize, entries: Vec<(usize, F)>) -> SparseMLPoly<F> {
  let mut evaluations = HashMap::new();
  for (idx, val) in entries {
    evaluations.insert(idx, val);
  }
  let mut poly = SparseMLPoly::new(n, evaluations, VecDeque::new(), SelectionPolynomial::new(n, 0, vec![]));
  poly.build_sorted_indices();
  poly
}

#[test]
fn test_dense_fix_variables_single_variable() {
  let evaluations = vec![<F as CryptoField>::zero(), <F as CryptoField>::one()];
  let poly = DenseMLPoly::from_evaluations(evaluations);

  let partial_point = vec![<F as CryptoField>::zero()];
  let fixed_poly = poly.fix_variables(&partial_point);
  assert_eq!(fixed_poly.n, 0);
  assert_eq!(fixed_poly.evaluations.len(), 1);
  assert_eq!(fixed_poly.evaluations[0], <F as CryptoField>::zero());

  let partial_point = vec![<F as CryptoField>::one()];
  let fixed_poly = poly.fix_variables(&partial_point);
  assert_eq!(fixed_poly.n, 0);
  assert_eq!(fixed_poly.evaluations.len(), 1);
  assert_eq!(fixed_poly.evaluations[0], <F as CryptoField>::one());

  let half = <F as CryptoField>::from_u32(2u32).invert();
  let partial_point = vec![half];
  let fixed_poly = poly.fix_variables(&partial_point);
  assert_eq!(fixed_poly.n, 0);
  assert_eq!(fixed_poly.evaluations.len(), 1);
  assert_eq!(fixed_poly.evaluations[0], half);
}

#[test]
fn test_dense_fix_variables_two_variables() {
  let evaluations = vec![
    <F as CryptoField>::one(),
    <F as CryptoField>::from_u32(2u32),
    <F as CryptoField>::from_u32(3u32),
    <F as CryptoField>::from_u32(4u32),
  ];
  let poly = DenseMLPoly::from_evaluations(evaluations);

  let partial_point = vec![<F as CryptoField>::zero()];
  let fixed_poly = poly.fix_variables(&partial_point);
  assert_eq!(fixed_poly.n, 1);
  assert_eq!(fixed_poly.evaluations, vec![<F as CryptoField>::one(), <F as CryptoField>::from_u32(3u32)]);

  let partial_point = vec![<F as CryptoField>::one()];
  let fixed_poly = poly.fix_variables(&partial_point);
  assert_eq!(fixed_poly.n, 1);
  assert_eq!(fixed_poly.evaluations, vec![<F as CryptoField>::from_u32(2u32), <F as CryptoField>::from_u32(4u32)]);

  let partial_point = vec![<F as CryptoField>::zero(), <F as CryptoField>::one()];
  let fixed_poly = poly.fix_variables(&partial_point);
  assert_eq!(fixed_poly.n, 0);
  assert_eq!(fixed_poly.evaluations, vec![<F as CryptoField>::from_u32(3u32)]);
}

#[test]
fn test_dense_fix_variables_three_variables() {
  let evaluations: Vec<F> = (1..=8).map(|v| <F as CryptoField>::from_u32(v)).collect();
  let poly = DenseMLPoly::from_evaluations(evaluations);

  let fixed_poly = poly.fix_variables(&[<F as CryptoField>::zero()]);
  assert_eq!(fixed_poly.n, 2);
  assert_eq!(
    fixed_poly.evaluations,
    vec![
      <F as CryptoField>::from_u32(1),
      <F as CryptoField>::from_u32(3),
      <F as CryptoField>::from_u32(5),
      <F as CryptoField>::from_u32(7),
    ]
  );

  let fixed_poly = poly.fix_variables(&[<F as CryptoField>::zero(), <F as CryptoField>::one()]);
  assert_eq!(fixed_poly.n, 1);
  assert_eq!(fixed_poly.evaluations, vec![<F as CryptoField>::from_u32(3), <F as CryptoField>::from_u32(7)]);
}

#[test]
fn test_dense_fix_variables_empty_partial_point() {
  let evaluations: Vec<F> = (1..=4).map(|v| <F as CryptoField>::from_u32(v)).collect();
  let poly = DenseMLPoly::from_evaluations(evaluations.clone());
  let fixed_poly = poly.fix_variables(&[]);
  assert_eq!(fixed_poly.n, poly.n);
  assert_eq!(fixed_poly.evaluations, evaluations);
}

#[test]
fn test_sparse_fix_variables_single_variable() {
  let poly = make_sparse(1, vec![(1, <F as CryptoField>::one())]);

  let fixed0 = poly.fix_variables(&[<F as CryptoField>::zero()]).to_dense();
  assert_eq!(fixed0.n, 0);
  assert_eq!(fixed0.evaluations[0], <F as CryptoField>::zero());

  let fixed1 = poly.fix_variables(&[<F as CryptoField>::one()]).to_dense();
  assert_eq!(fixed1.n, 0);
  assert_eq!(fixed1.evaluations[0], <F as CryptoField>::one());
}

#[test]
fn test_sparse_fix_variables_two_variables() {
  let poly = make_sparse(2, vec![(0, <F as CryptoField>::from_u32(1)), (3, <F as CryptoField>::from_u32(4))]);

  let fixed0 = poly.fix_variables(&[<F as CryptoField>::zero()]).to_dense();
  assert_eq!(fixed0.evaluations, vec![<F as CryptoField>::one(), <F as CryptoField>::zero()]);

  let fixed1 = poly.fix_variables(&[<F as CryptoField>::one()]).to_dense();
  assert_eq!(fixed1.evaluations, vec![<F as CryptoField>::zero(), <F as CryptoField>::from_u32(4)]);
}

#[test]
fn test_sparse_fix_variables_interpolation() {
  let evaluations: Vec<F> = (1..=4).map(|v| <F as CryptoField>::from_u32(v)).collect();
  let dense_poly = DenseMLPoly::from_evaluations(evaluations.clone());
  let sparse_poly = make_sparse(2, evaluations.iter().copied().enumerate().collect());

  let partial_point = vec![<F as CryptoField>::from_u32(3)];
  let dense_fixed = dense_poly.fix_variables(&partial_point);
  let sparse_fixed = sparse_poly.fix_variables(&partial_point).to_dense();
  assert_eq!(dense_fixed.evaluations, sparse_fixed.evaluations);
}

#[test]
fn test_sparse_fix_variables_empty_polynomial() {
  let poly = make_sparse(2, vec![]);
  let fixed_poly = poly.fix_variables(&[<F as CryptoField>::from_u32(5)]);
  assert_eq!(fixed_poly.n, 1);
  assert!(fixed_poly.evaluations.is_empty());
}

#[test]
#[should_panic(expected = "invalid size of partial point")]
fn test_dense_fix_variables_invalid_partial_point_size() {
  let poly = DenseMLPoly::from_evaluations(vec![<F as CryptoField>::one(), <F as CryptoField>::from_u32(2)]);
  poly.fix_variables(&[<F as CryptoField>::one(), <F as CryptoField>::from_u32(2)]);
}

#[test]
#[should_panic(expected = "invalid partial point dimension")]
fn test_sparse_fix_variables_invalid_partial_point_size() {
  let poly = make_sparse(1, vec![(0, <F as CryptoField>::one())]);
  poly.fix_variables(&[<F as CryptoField>::one(), <F as CryptoField>::from_u32(2)]);
}

#[test]
fn test_dense_fix_variables_from_right() {
  let evaluations: Vec<F> = (1..=4).map(|v| <F as CryptoField>::from_u32(v)).collect();
  let poly = DenseMLPoly::from_evaluations(evaluations.clone());

  let partial_suffix = vec![<F as CryptoField>::from_u32(5), <F as CryptoField>::from_u32(6)];
  let fixed_poly = fix_variables_from_right(&poly, &partial_suffix);
  let left_poly = poly.fix_variables(&partial_suffix);
  assert_eq!(fixed_poly.evaluations[0], left_poly.evaluations[0]);

  let new_poly = fix_variables_from_right(&DenseMLPoly::from_evaluations(evaluations), &[<F as CryptoField>::from_u32(5)]);
  assert_eq!(
    new_poly.evaluations[0],
    (<F as CryptoField>::one() - <F as CryptoField>::from_u32(5)) * <F as CryptoField>::one()
      + <F as CryptoField>::from_u32(5) * <F as CryptoField>::from_u32(3)
  );
  assert_eq!(
    new_poly.evaluations[1],
    (<F as CryptoField>::one() - <F as CryptoField>::from_u32(5)) * <F as CryptoField>::from_u32(2)
      + <F as CryptoField>::from_u32(5) * <F as CryptoField>::from_u32(4)
  );
}
