///|
fn normalization(
  graph : @buffer.Buffer,
  input : String,
  norm : @actor.Normalization,
  prefix : String,
) -> String raise @actor.ActorError {
  let width = norm.mean.length()
  initializer(
    graph,
    prefix + ".mean",
    [width],
    @actor.encode_parameters(norm.mean[:])[:],
  )
  initializer(
    graph,
    prefix + ".variance",
    [width],
    @actor.encode_parameters(norm.variance[:])[:],
  )
  initializer(
    graph,
    prefix + ".eps2",
    [1],
    @actor.encode_parameters([norm.epsilon * norm.epsilon])[:],
  )
  node(graph, "Sub", [input, prefix + ".mean"], prefix + ".centered")
  node(
    graph,
    "Add",
    [prefix + ".variance", prefix + ".eps2"],
    prefix + ".regularized",
  )
  node(graph, "Sqrt", [prefix + ".regularized"], prefix + ".std")
  node(
    graph,
    "Div",
    [prefix + ".centered", prefix + ".std"],
    prefix + ".normalized",
  )
  prefix + ".normalized"
}

///|
fn network(
  graph : @buffer.Buffer,
  widths : Array[Int32],
  parameters : BytesView,
  offset : Int32,
  input : String,
  prefix : String,
) -> (String, Int32) {
  let mut current = input
  let mut cursor = offset
  for i in 0..<(widths.length() - 1) {
    let nin = widths[i]
    let nout = widths[i + 1]
    let layer = prefix + ".\{i}"
    let weight_bytes = nin * nout * 4
    let bias_bytes = nout * 4
    initializer(
      graph,
      layer + ".weight",
      [nout, nin],
      parameters[cursor:cursor + weight_bytes],
    )
    cursor += weight_bytes
    initializer(
      graph,
      layer + ".bias",
      [nout],
      parameters[cursor:cursor + bias_bytes],
    )
    cursor += bias_bytes
    node(
      graph,
      "Gemm",
      [current, layer + ".weight", layer + ".bias"],
      layer + ".affine",
      attributes=[integer_attribute("transB", 1)],
    )
    current = layer + ".affine"
    if i < widths.length() - 2 {
      node(graph, "Elu", [current], layer + ".elu")
      current = layer + ".elu"
    }
  }
  (current, cursor)
}

///|
/// Emits a static batch-one ONNX model, IR 8/opset 13, with FP32 weights and
/// arithmetic. Input names are history and goals; output name is actions.
/// No Python, protobuf, or training framework is required by this exporter.
pub fn encode(
  spec : @actor.Spec,
  parameters : BytesView,
) -> Bytes raise @actor.ActorError {
  ignore(@actor.decode_parameters(spec, parameters))
  let graph = @buffer.Buffer()
  text(graph, 2, "lunakine_actor")
  message(
    graph,
    11,
    value_info("history", [1, spec.history_frames, spec.observation_width])[:],
  )
  message(graph, 11, value_info("goals", [1, spec.goal_width()])[:])
  message(graph, 12, value_info("actions", [1, spec.action_width()])[:])
  let goal_input = normalization(graph, "goals", spec.goal_norm, "goal")
  let history_input = normalization(
    graph,
    "history",
    spec.observation_norm,
    "observation",
  )
  let shape_bytes = @buffer.Buffer()
  shape_bytes.write_int64_le(1L)
  shape_bytes.write_int64_le(spec.history_width().to_int64())
  initializer(graph, "history_shape", [2], shape_bytes.to_bytes()[:], dtype=7)
  node(graph, "Reshape", [history_input, "history_shape"], "history_flat")
  let (encoded, cursor) = network(
    graph,
    spec.encoder,
    parameters,
    0,
    goal_input,
    "encoder",
  )
  node(graph, "Tanh", [encoded], "latent")
  node(graph, "Concat", ["latent", "history_flat"], "decoder_input", attributes=[
    integer_attribute("axis", 1),
  ])
  let (decoded, _) = network(
    graph,
    spec.decoder,
    parameters,
    cursor,
    "decoder_input",
    "decoder",
  )
  initializer(
    graph,
    "action_mask",
    [spec.action_width()],
    @actor.encode_parameters(spec.action_mask[:])[:],
  )
  node(graph, "Mul", [decoded, "action_mask"], "actions")
  let model = @buffer.Buffer()
  integer(model, 1, 8)
  text(model, 2, "lunakine")
  message(model, 7, graph.to_bytes()[:])
  let opset = @buffer.Buffer()
  integer(opset, 2, 13)
  message(model, 8, opset.to_bytes()[:])
  model.to_bytes()
}
