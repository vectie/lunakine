///|
/// Row-major [out,in] weights followed by biases for every encoder layer,
/// followed by every decoder layer. Storage is little-endian IEEE FP32.
pub fn encode_parameters(values : ArrayView[Double]) -> Bytes raise ActorError {
  guard values.length() <= 536870911 else {
    raise InvalidActor("parameter buffer too large")
  }
  let bytes = FixedArray::make(values.length() * 4, b'\x00')
  for i, value in values {
    let narrowed = Float::from_double(value)
    guard finite(narrowed.to_double()) else {
      raise InvalidActor("parameter is not finite FP32")
    }
    bytes.unsafe_write_uint32_le(i * 4, narrowed.reinterpret_as_uint())
  }
  Bytes::from_array(bytes[:])
}

///|
pub fn decode_parameters(
  spec : Spec,
  bytes : BytesView,
) -> Array[Float] raise ActorError {
  spec.validate()
  let count = spec.parameter_count()
  guard bytes.length() == count * 4 else {
    raise InvalidActor("parameter blob size differs from actor specification")
  }
  Array::makei(count, i => {
    let p = i * 4
    let bits = bytes[p].to_uint() |
      (bytes[p + 1].to_uint() << 8) |
      (bytes[p + 2].to_uint() << 16) |
      (bytes[p + 3].to_uint() << 24)
    let value = Float::reinterpret_from_uint(bits)
    guard finite(value.to_double()) else {
      raise InvalidActor("nonfinite parameter")
    }
    value
  })
}
