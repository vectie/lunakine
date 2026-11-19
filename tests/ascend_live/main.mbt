///|
struct Sample {
  history : Array[Array[Double]]
  goals : Array[Double]
  libtorch_actions : Array[Double]
  reference_actions : Array[Double]
} derive(FromJson)

///|
struct Fixture {
  cases : Array[Sample]
} derive(FromJson)

///|
pub extend Sample with @json.FromJson::{from_json}

///|
pub extend Fixture with @json.FromJson::{from_json}

///|
async fn main {
  let args = @env.args()
  guard args is [_, library, model, actor_spec, comparison, output] else {
    fail(
      "usage: ascend_live LIBASCENDCL MODEL.om ACTOR.json COMPARISON.json NEW_REPORT.json",
    )
  }
  guard !@fs.exists(output) else { fail("report must be new") }
  let spec : @actor.Spec = @json.from_json(@fs.read_file(actor_spec).json())
  spec.validate()
  let fixture : Fixture = @json.from_json(@fs.read_file(comparison).json())
  guard !fixture.cases.is_empty() else { fail("empty comparison fixture") }
  let inputs = fixture.cases.map(sample => {
    let flat : Array[Float] = []
    for frame in sample.history {
      for value in frame {
        flat.push(Float::from_double(value))
      }
    }
    (
      FixedArray::from_array(flat[:]),
      FixedArray::from_array(sample.goals.map(Float::from_double)[:]),
    )
  })
  let runtime = @ascend.Runtime::open(library)
  defer (runtime.close() catch { _ => () })
  let session = runtime.prepare(model, spec)
  defer (session.close() catch { _ => () })
  let actions = FixedArray::make(spec.action_width(), 0.0F)
  let tolerance = 0.001
  let mut maximum_libtorch_error = 0.0
  let mut maximum_reference_error = 0.0
  let mut compared = 0
  let times : Array[Double] = []
  // Every run checks every output. Timing covers the synchronous runtime call,
  // including host/device transfers, and excludes comparison/JSON allocation.
  for iteration in 0..<1000 {
    let index = iteration % fixture.cases.length()
    let (history, goals) = inputs[index]
    let sample = fixture.cases[index]
    guard sample.libtorch_actions.length() == actions.length() &&
      sample.reference_actions.length() == actions.length() else {
      fail("comparison action width mismatch")
    }
    let start = @bench.monotonic_clock_start()
    session.run_into(history, goals, actions)
    let elapsed_us = @bench.monotonic_clock_end(start)
    for column in 0..<actions.length() {
      let value = actions[column].to_double()
      let torch_error = (value - sample.libtorch_actions[column]).abs()
      let reference_error = (value - sample.reference_actions[column]).abs()
      guard torch_error <= tolerance && reference_error <= tolerance else {
        fail(
          "NPU action mismatch at sample \{index}, action \{column}: \{torch_error}, \{reference_error}",
        )
      }
      maximum_libtorch_error = maximum_libtorch_error.max(torch_error)
      maximum_reference_error = maximum_reference_error.max(reference_error)
      compared += 1
    }
    if iteration >= 10 {
      times.push(elapsed_us)
    }
  }
  session.close()
  runtime.close()
  times.sort()
  let mut total = 0.0
  for value in times {
    total += value
  }
  let report : Json = {
    "format": "lunakine.ascend-live-comparison.v1",
    "library": Json(library),
    "model": Json(model),
    "fixture_cases": Json(fixture.cases.length()),
    "runs": 1000,
    "warmup_runs_excluded_from_timing": 10,
    "compared_actions": Json(compared),
    "maximum_libtorch_error": Json(maximum_libtorch_error),
    "maximum_reference_error": Json(maximum_reference_error),
    "absolute_tolerance": Json(tolerance),
    "transfer_inclusive_us": {
      "minimum": Json(times[0]),
      "mean": Json(total / times.length().to_double()),
      "p95": Json(times[times.length() * 95 / 100]),
      "p99": Json(times[times.length() * 99 / 100]),
      "maximum": Json(times[times.length() - 1]),
    },
    "passed": true,
    "robot_control_qualified": false,
  }
  @fs.write_file(output, report.stringify(indent=2), create_mode=CreateNew)
  println(report.stringify())
}
