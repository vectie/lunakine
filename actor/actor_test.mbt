///|
fn fixture() -> @actor.Spec {
  {
    format: "lunakine.elu-actor.v1",
    history_frames: 1,
    observation_width: 1,
    encoder: [1, 1],
    decoder: [2, 1],
    observation_norm: { mean: [0.0], variance: [0.0], epsilon: 1.0, },
    goal_norm: { mean: [0.0], variance: [0.0], epsilon: 1.0, },
    action_mask: [1.0],
  }
}

///|
test "portable FP32 parameters preserve endian and actor branch ordering" {
  let spec = fixture()
  let blob = @actor.encode_parameters([1.0, 0.0, 2.0, 3.0, 0.5])
  assert_eq(blob[:4].to_owned(), b"\x00\x00\x80\x3f")
  let model = @actor.Reference::prepare(spec, blob[:])
  assert_eq(model.run([1.0], [0.0]), [3.5])
  let expected = 3.5 + 2.0 * 0.46211715726000974
  assert_true((model.run([1.0], [0.5])[0] - expected).abs() < 0.000001)
}

///|
test "action mask and normalization belong to the exported forward program" {
  let original = fixture()
  let spec = { ..original, action_mask: [0.0], }
  let model = @actor.Reference::prepare(
    spec,
    @actor.encode_parameters([1.0, 0.0, 2.0, 3.0, 0.5])[:],
  )
  assert_eq(model.run([7.0], [2.0]), [0.0])
}

///|
test "wrong parameter and observation shapes fail before evaluation" {
  let spec = fixture()
  assert_true(
    try @actor.Reference::prepare(spec, b""[:]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  let model = @actor.Reference::prepare(
    spec,
    @actor.encode_parameters([1.0, 0.0, 2.0, 3.0, 0.5])[:],
  )
  assert_true(
    try model.run([], [0.0]) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  assert_true(
    try { ..spec, decoder: [3, 1], }.validate() catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
