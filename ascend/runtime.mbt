///|
pub suberror AscendError {
  ExecutionError(String)
} derive(Debug)

///|
pub extend AscendError with @debug.Debug::{to_repr}

///|
priv type RuntimeHandle

///|
priv type SessionHandle

///|
#borrow(path)
extern "c" fn runtime_open(path : Bytes) -> RuntimeHandle = "lk_runtime_open"

///|
#borrow(handle)
extern "c" fn runtime_error(handle : RuntimeHandle) -> Bytes = "lk_runtime_error"

///|
#borrow(handle)
extern "c" fn runtime_close(handle : RuntimeHandle) -> Int32 = "lk_runtime_close"

///|
#borrow(runtime, model)
extern "c" fn session_prepare(
  runtime : RuntimeHandle,
  model : Bytes,
  device : Int32,
  history : Int32,
  goals : Int32,
  actions : Int32,
) -> SessionHandle = "lk_session_prepare"

///|
#borrow(handle)
extern "c" fn session_error(handle : SessionHandle) -> Bytes = "lk_session_error"

///|
#borrow(handle)
extern "c" fn session_close(handle : SessionHandle) -> Int32 = "lk_session_close"

///|
#borrow(handle, history, goals, actions)
extern "c" fn session_run(
  handle : SessionHandle,
  history : FixedArray[Float],
  goals : FixedArray[Float],
  actions : FixedArray[Float],
) -> Int32 = "lk_session_run"

///|
/// Owns AscendCL initialization and the explicitly selected shared library.
/// Open one runtime per process; share it across sessions on the same thread.
pub struct Runtime {
  priv handle : RuntimeHandle
}

///|
/// Opens libascendcl explicitly. Absence of CANN is an error, never a CPU fallback.
pub fn Runtime::open(library : String) -> Runtime raise AscendError {
  let handle = runtime_open(@utf8.encode(library))
  let message = @utf8.decode_lossy(runtime_error(handle))
  guard message.is_empty() else { raise ExecutionError(message) }
  { handle, }
}

///|
/// Close sessions first. Busy/error results retain native ownership for retry.
pub fn Runtime::close(self : Runtime) -> Unit raise AscendError {
  if runtime_close(self.handle) != 0 {
    raise ExecutionError(@utf8.decode_lossy(runtime_error(self.handle)))
  }
}

///|
/// Static batch-one model with FP32 host inputs/outputs. Device buffers and ACL
/// datasets are allocated once at preparation and reused by run_into.
pub struct Session {
  priv handle : SessionHandle
  priv history_width : Int32
  priv goal_width : Int32
  priv action_width : Int32
}

///|
pub fn Runtime::prepare(
  self : Runtime,
  model : String,
  spec : @actor.Spec,
  device~ : Int32 = 0,
) -> Session raise {
  spec.validate()
  guard device >= 0 else {
    raise ExecutionError("negative Ascend device index")
  }
  let handle = session_prepare(
    self.handle,
    @utf8.encode(model),
    device,
    spec.history_width(),
    spec.goal_width(),
    spec.action_width(),
  )
  let message = @utf8.decode_lossy(session_error(handle))
  guard message.is_empty() else { raise ExecutionError(message) }
  {
    handle,
    history_width: spec.history_width(),
    goal_width: spec.goal_width(),
    action_width: spec.action_width(),
  }
}

///|
/// Synchronous, single-caller execution. The caller retains all host arrays.
/// No host array allocation occurs on the successful native execution path.
pub fn Session::run_into(
  self : Session,
  history : FixedArray[Float],
  goals : FixedArray[Float],
  actions : FixedArray[Float],
) -> Unit raise AscendError {
  guard history.length() == self.history_width &&
    goals.length() == self.goal_width &&
    actions.length() == self.action_width else {
    raise ExecutionError("actor input/output size mismatch")
  }
  guard history.all(x => @actor.finite(x.to_double())) &&
    goals.all(x => @actor.finite(x.to_double())) else {
    raise ExecutionError("nonfinite actor input")
  }
  if session_run(self.handle, history, goals, actions) != 0 {
    raise ExecutionError(@utf8.decode_lossy(session_error(self.handle)))
  }
  guard actions.all(x => @actor.finite(x.to_double())) else {
    raise ExecutionError("nonfinite actor output")
  }
}

///|
pub fn Session::close(self : Session) -> Unit raise AscendError {
  if session_close(self.handle) != 0 {
    raise ExecutionError(@utf8.decode_lossy(session_error(self.handle)))
  }
}
