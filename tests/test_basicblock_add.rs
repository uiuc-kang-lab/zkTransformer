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

use zk_torch_2::basicblock::{Add, BasicBlock};
use zk_torch_2::dag::{Claim, DataType, Role, Witness};
use zk_torch_2::util::poly::CryptoField;
use zk_torch_2::util::transcript::Transcript;

fn create_test_witness(shape: Vec<usize>, data: Vec<F>, sf: usize, role: Role) -> Witness<F> {
  Witness::new(shape, data, DataType::Float, sf, role)
}

#[test]
fn test_add_run_basic() {
  let add = Add;
  let a = create_test_witness(
    vec![2],
    vec![<F as CryptoField>::from_u32(1u32), <F as CryptoField>::from_u32(2u32)],
    1,
    Role::Input,
  );
  let b = create_test_witness(
    vec![2],
    vec![<F as CryptoField>::from_u32(3u32), <F as CryptoField>::from_u32(4u32)],
    1,
    Role::Input,
  );

  let results = add.run(&[&a, &b]);
  assert_eq!(results.len(), 1);
  let c = &results[0];
  assert_eq!(c.shape, vec![2]);
  assert_eq!(c.sf, 1);
  assert_eq!(c.role, Role::Output);
  assert_eq!(c.data.as_ref().unwrap().index(0), <F as CryptoField>::from_u32(4u32));
  assert_eq!(c.data.as_ref().unwrap().index(1), <F as CryptoField>::from_u32(6u32));
}

#[test]
fn test_add_run_zero_inputs() {
  let add = Add;
  let a = create_test_witness(vec![2], vec![<F as CryptoField>::zero(), <F as CryptoField>::zero()], 0, Role::Input);
  let b = create_test_witness(vec![2], vec![<F as CryptoField>::zero(), <F as CryptoField>::zero()], 0, Role::Input);
  let results = add.run(&[&a, &b]);
  let c = &results[0];
  assert_eq!(c.data.as_ref().unwrap().index(0), <F as CryptoField>::zero());
  assert_eq!(c.data.as_ref().unwrap().index(1), <F as CryptoField>::zero());
}

#[test]
fn test_add_run_single_element() {
  let add = Add;
  let a = create_test_witness(vec![1], vec![<F as CryptoField>::from_u32(5u32)], 2, Role::Input);
  let b = create_test_witness(vec![1], vec![<F as CryptoField>::from_u32(7u32)], 2, Role::Input);
  let results = add.run(&[&a, &b]);
  let c = &results[0];
  assert_eq!(c.shape, vec![1]);
  assert_eq!(c.data.as_ref().unwrap().index(0), <F as CryptoField>::from_u32(12u32));
}

#[test]
fn test_add_run_large_tensors() {
  let add = Add;
  let size = 8;
  let a_data: Vec<F> = (1..=size).map(|i| <F as CryptoField>::from_u32(i as u32)).collect();
  let b_data: Vec<F> = (1..=size).map(|i| <F as CryptoField>::from_u32((i * 2) as u32)).collect();
  let a = create_test_witness(vec![size], a_data, 1, Role::Input);
  let b = create_test_witness(vec![size], b_data, 1, Role::Input);
  let results = add.run(&[&a, &b]);
  let c = &results[0];
  for i in 0..size {
    assert_eq!(c.data.as_ref().unwrap().index(i), <F as CryptoField>::from_u32((3 * (i + 1)) as u32));
  }
}

#[test]
#[should_panic(expected = "Add expects inputs with the same scale factor")]
fn test_add_run_mismatched_scale_factors() {
  let add = Add;
  let a = create_test_witness(vec![2], vec![<F as CryptoField>::one(), <F as CryptoField>::from_u32(2)], 1, Role::Input);
  let b = create_test_witness(vec![2], vec![<F as CryptoField>::from_u32(3), <F as CryptoField>::from_u32(4)], 2, Role::Input);
  add.run(&[&a, &b]);
}

#[test]
#[should_panic]
fn test_add_run_mismatched_shapes() {
  let add = Add;
  let a = create_test_witness(vec![2], vec![<F as CryptoField>::one(), <F as CryptoField>::from_u32(2)], 1, Role::Input);
  let b = create_test_witness(
    vec![3],
    vec![
      <F as CryptoField>::one(),
      <F as CryptoField>::from_u32(2),
      <F as CryptoField>::from_u32(3),
      <F as CryptoField>::zero(),
    ],
    1,
    Role::Input,
  );
  add.run(&[&a, &b]);
}

#[test]
fn test_add_prove_verify_round_trip() {
  let add = Add;
  let a = create_test_witness(vec![2], vec![<F as CryptoField>::from_u32(1), <F as CryptoField>::from_u32(2)], 1, Role::Input);
  let b = create_test_witness(vec![2], vec![<F as CryptoField>::from_u32(3), <F as CryptoField>::from_u32(4)], 1, Role::Input);
  let c = create_test_witness(vec![2], vec![<F as CryptoField>::from_u32(4), <F as CryptoField>::from_u32(6)], 1, Role::Output);

  let witnesses = vec![&a, &b, &c];
  let edge_ids = vec![0, 1, 2];

  let out_claim = Claim {
    edge_id: 2,
    sparse_id: 0,
    point: vec![<F as CryptoField>::zero()],
    eval: <F as CryptoField>::from_u32(4),
  };
  let out_claims = vec![&out_claim];

  let mut transcript = Transcript::new(b"add_test");
  let (proofs, claims) = add.prove(&witnesses, &edge_ids, &out_claims, &mut transcript);
  assert!(proofs.is_empty(), "Add emits 0 sumcheck proofs");
  assert_eq!(claims.len(), 2, "Add emits 2 input claims");

  let mut verify_transcript = Transcript::new(b"add_test");
  let verify_claims = vec![&claims[0], &claims[1], &out_claim];
  let verified = add.verify(&witnesses, &verify_claims, &[], &mut verify_transcript);
  assert!(verified, "Verification should succeed for valid proof");
}

#[test]
fn test_add_verify_incorrect_sum_and_mismatched_points() {
  let add = Add;
  let a = Witness::new_wo_data(vec![2], DataType::Float, 1, Role::Input);
  let b = Witness::new_wo_data(vec![2], DataType::Float, 1, Role::Input);
  let c = Witness::new_wo_data(vec![2], DataType::Float, 1, Role::Output);
  let witnesses = vec![&a, &b, &c];

  let p1 = vec![<F as CryptoField>::one()];
  let p2 = vec![<F as CryptoField>::from_u32(2)];

  let a_claim = Claim { edge_id: 0, sparse_id: 0, point: p1.clone(), eval: <F as CryptoField>::from_u32(5) };
  let b_claim = Claim { edge_id: 1, sparse_id: 0, point: p1.clone(), eval: <F as CryptoField>::from_u32(7) };
  let c_good = Claim { edge_id: 2, sparse_id: 0, point: p1.clone(), eval: <F as CryptoField>::from_u32(12) };
  let c_bad = Claim { edge_id: 2, sparse_id: 0, point: p1.clone(), eval: <F as CryptoField>::from_u32(13) };
  let b_bad_pt = Claim { edge_id: 1, sparse_id: 0, point: p2, eval: <F as CryptoField>::from_u32(7) };

  let mut tx = Transcript::new(b"add_verify");
  assert!(add.verify(&witnesses, &[&a_claim, &b_claim, &c_good], &[], &mut tx));
  assert!(!add.verify(&witnesses, &[&a_claim, &b_claim, &c_bad], &[], &mut tx));
  assert!(!add.verify(&witnesses, &[&a_claim, &b_bad_pt, &c_good], &[], &mut tx));
}
