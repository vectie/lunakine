///|
fn test_spec() -> @actor.Spec {
  {
    format: "lunakine.elu-actor.v1",
    history_frames: 1,
    observation_width: 2,
    encoder: [1, 1],
    decoder: [3, 1],
    observation_norm: { mean: [0.0, 0.0], variance: [1.0, 1.0], epsilon: 0.01, },
    goal_norm: { mean: [0.0], variance: [1.0], epsilon: 0.01, },
    action_mask: [1.0],
  }
}

///|
test "absent AscendCL is an explicit error" {
  assert_true(
    try @ascend.Runtime::open("/no-such-lunakine-library.so") catch {
      @ascend.ExecutionError(message) =>
        message.has_prefix("AscendCL unavailable")
    } noraise {
      _ => false
    },
  )
}

///|
test "software ABI fixture verifies input mapping reuse and deterministic close" {
  guard @env.get_env_var("LUNAKINE_TEST_ASCENDCL") is Some(path) else {
    println(
      "SKIP software AscendCL ABI fixture: run scripts/test_ascend_abi.mbtx",
    )
    return
  }
  let runtime = @ascend.Runtime::open(path)
  let session = runtime.prepare("software-fixture.om", test_spec())
  assert_true(
    try runtime.close() catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  let history : FixedArray[Float] = [1.0, 2.0]
  let goals : FixedArray[Float] = [3.0]
  let actions : FixedArray[Float] = [0.0]
  for i in 0..<10 {
    history[0] = Float::from_int(i)
    session.run_into(history, goals, actions)
    assert_eq(actions[0], Float::from_int(i * 2 + 5))
  }
  session.close()
  session.close()
  assert_true(
    try session.run_into(history, goals, actions) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  // The test library's aclFinalize rejects any retained resource.
  runtime.close()
  runtime.close()
}

///|
test "software ABI fixture releases partial preparation and propagates execution errors" {
  guard @env.get_env_var("LUNAKINE_TEST_ASCENDCL") is Some(path) else { return }
  let runtime = @ascend.Runtime::open(path)
  let spec = test_spec()
  let wrong = { ..spec, history_frames: 2, decoder: [5, 1], }
  assert_true(
    try runtime.prepare("wrong-size.om", wrong) catch {
      _ => true
    } noraise {
      _ => false
    },
  )
  let session = runtime.prepare("fail-execute.om", spec)
  assert_true(
    try session.run_into([1.0, 2.0], [3.0], [0.0]) catch {
      @ascend.ExecutionError(message) => message == "aclmdlExecute: 77"
    } noraise {
      _ => false
    },
  )
  session.close()
  runtime.close()
}
