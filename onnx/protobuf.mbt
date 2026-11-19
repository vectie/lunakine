///|
// Small writer for the ONNX fields we emit, not a general protobuf compiler.
fn integer(output : @buffer.Buffer, field : Int32, value : Int32) -> Unit {
  output.write_leb128((field * 8).reinterpret_as_uint())
  output.write_leb128(value.to_uint64())
}

///|
fn message(output : @buffer.Buffer, field : Int32, value : BytesView) -> Unit {
  output.write_leb128((field * 8 + 2).reinterpret_as_uint())
  output.write_leb128(value.length().to_uint64())
  output.write_bytes(value)
}

///|
fn text(output : @buffer.Buffer, field : Int32, value : String) -> Unit {
  message(output, field, @utf8.encode(value)[:])
}

///|
fn integer_attribute(name : String, value : Int32) -> Bytes {
  let attr = @buffer.Buffer()
  text(attr, 1, name)
  integer(attr, 20, 2) // AttributeProto.INT
  integer(attr, 3, value)
  attr.to_bytes()
}

///|
fn value_info(name : String, shape : Array[Int32]) -> Bytes {
  let dims = @buffer.Buffer()
  for extent in shape {
    let dim = @buffer.Buffer()
    integer(dim, 1, extent)
    message(dims, 1, dim.to_bytes()[:])
  }
  let tensor_type = @buffer.Buffer()
  integer(tensor_type, 1, 1) // FLOAT
  message(tensor_type, 2, dims.to_bytes()[:])
  let value_type = @buffer.Buffer()
  message(value_type, 1, tensor_type.to_bytes()[:])
  let value = @buffer.Buffer()
  text(value, 1, name)
  message(value, 2, value_type.to_bytes()[:])
  value.to_bytes()
}

///|
fn initializer(
  graph : @buffer.Buffer,
  name : String,
  shape : Array[Int32],
  bytes : BytesView,
  dtype~ : Int32 = 1,
) -> Unit {
  let tensor = @buffer.Buffer()
  for extent in shape {
    integer(tensor, 1, extent)
  }
  integer(tensor, 2, dtype)
  text(tensor, 8, name)
  message(tensor, 9, bytes)
  message(graph, 5, tensor.to_bytes()[:])
}

///|
fn node(
  graph : @buffer.Buffer,
  op : String,
  inputs : Array[String],
  output : String,
  attributes~ : Array[Bytes] = [],
) -> Unit {
  let value = @buffer.Buffer()
  for input in inputs {
    text(value, 1, input)
  }
  text(value, 2, output)
  text(value, 3, output)
  text(value, 4, op)
  for attr in attributes {
    message(value, 5, attr[:])
  }
  message(graph, 1, value.to_bytes()[:])
}
