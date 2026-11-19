///|
/// Offline correctness reference. This is deliberately separate from the
/// production Ascend session and is never selected as an automatic fallback.
pub struct Reference {
  priv spec : Spec
  priv parameters : Array[Float]
}

///|
pub fn Reference::prepare(
  spec : Spec,
  parameters : BytesView,
) -> Reference raise ActorError {
  { spec, parameters: decode_parameters(spec, parameters), }
}

///|
fn normalize(input : ArrayView[Double], norm : Normalization) -> Array[Float] {
  let width = norm.mean.length()
  input.mapi((i, x) => {
    let mean = Float::from_double(norm.mean[i % width])
    let variance = Float::from_double(norm.variance[i % width])
    let eps2 = Float::from_double(norm.epsilon * norm.epsilon)
    (Float::from_double(x) - mean) / (variance + eps2).sqrt()
  })
}

///|
fn network(
  widths : Array[Int32],
  parameters : Array[Float],
  offset : Int32,
  input : Array[Float],
) -> (Array[Float], Int32) {
  let mut x = input
  let mut cursor = offset
  for layer in 0..<(widths.length() - 1) {
    let nin = widths[layer]
    let nout = widths[layer + 1]
    let base = cursor
    let biases = base + nin * nout
    let last = layer == widths.length() - 2
    x = Array::makei(nout, row => {
      let mut sum : Float = 0.0
      for col in 0..<nin {
        sum = sum + x[col] * parameters[base + row * nin + col]
      }
      let y = sum + parameters[biases + row]
      if last || y >= 0.0 {
        y
      } else {
        @math.expf(y) - 1.0
      }
    })
    cursor = biases + nout
  }
  (x, cursor)
}

///|
/// History is flattened oldest-to-newest, with features contiguous per frame.
pub fn Reference::run(
  self : Reference,
  history : ArrayView[Double],
  goals : ArrayView[Double],
) -> Array[Double] raise ActorError {
  let s = self.spec
  guard history.length() == s.history_width() &&
    goals.length() == s.goal_width() &&
    history.all(x => finite(Float::from_double(x).to_double())) &&
    goals.all(x => finite(Float::from_double(x).to_double())) else {
    raise InvalidActor("inference input size or finite-value mismatch")
  }
  let (latent, cursor) = network(
    s.encoder,
    self.parameters,
    0,
    normalize(goals, s.goal_norm),
  )
  let input = latent.map(@math.tanhf)
  input.append(normalize(history, s.observation_norm))
  let (actions, _) = network(s.decoder, self.parameters, cursor, input)
  let result = actions.mapi((i, value) => {
    (value * Float::from_double(s.action_mask[i])).to_double()
  })
  guard result.all(finite) else { raise InvalidActor("nonfinite actor result") }
  result
}
