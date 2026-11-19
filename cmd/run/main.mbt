///|
struct Input {
  history : Array[Double]
  goals : Array[Double]
} derive(FromJson)

///|
pub extend Input with @json.FromJson::{from_json}

///|
async fn main {
  let args = @env.args()
  guard args.length() == 5 else {
    fail("usage: run LIBASCENDCL MODEL.om ACTOR.json INPUT.json")
  }
  let spec : @actor.Spec = @json.from_json(
    @json.parse(@utf8.decode(@fs.read_file(args[3]).binary())),
  )
  spec.validate()
  let input : Input = @json.from_json(
    @json.parse(@utf8.decode(@fs.read_file(args[4]).binary())),
  )
  let runtime = @ascend.Runtime::open(args[1])
  defer (runtime.close() catch { _ => () })
  let session = runtime.prepare(args[2], spec)
  defer (session.close() catch { _ => () })
  let history = FixedArray::from_array(input.history.map(Float::from_double)[:])
  let goals = FixedArray::from_array(input.goals.map(Float::from_double)[:])
  let actions = FixedArray::make(spec.action_width(), 0.0F)
  session.run_into(history, goals, actions)
  println(Json(actions.map(x => x.to_double())).stringify())
  session.close()
  runtime.close()
}
