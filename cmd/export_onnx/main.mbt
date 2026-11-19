///|
async fn main {
  let args = @env.args()
  guard args.length() == 3 else {
    fail("usage: export_onnx BUNDLE_DIRECTORY OUTPUT.onnx")
  }
  let spec : @actor.Spec = @json.from_json(
    @json.parse(@utf8.decode(@fs.read_file(args[1] + "/actor.json").binary())),
  )
  let parameters = @fs.read_file(args[1] + "/parameters.f32").binary()
  let bytes = @onnx.encode(spec, parameters[:])
  @fs.write_file(args[2], bytes, create_mode=CreateNew)
  println("exported static FP32 actor: " + args[2])
}
