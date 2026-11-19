// Narrow dynamically loaded AscendCL ABI. No CUDA, Python, or toolkit headers.
// Signatures and enum values follow Huawei's public AscendCL C API.
#include <moonbit.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

typedef struct {
  void *library;
  int initialized, sessions, device;
  char error[256];
  int (*init)(const char *);
  int (*finalize)(void);
  int (*set_device)(int32_t);
  int (*reset_device)(int32_t);
  int (*create_context)(void **, int32_t);
  int (*destroy_context)(void *);
  int (*set_context)(void *);
  int (*allocate)(void **, size_t, int);
  int (*release)(void *);
  int (*copy)(void *, size_t, const void *, size_t, int);
  int (*load)(const char *, uint32_t *);
  int (*unload)(uint32_t);
  void *(*create_desc)(void);
  int (*get_desc)(void *, uint32_t);
  int (*destroy_desc)(void *);
  size_t (*num_inputs)(const void *);
  size_t (*num_outputs)(const void *);
  int (*input_index)(const void *, const char *, size_t *);
  size_t (*input_size)(const void *, size_t);
  size_t (*output_size)(const void *, size_t);
  int (*input_type)(const void *, size_t);
  int (*output_type)(const void *, size_t);
  void *(*create_dataset)(void);
  int (*destroy_dataset)(void *);
  void *(*create_buffer)(void *, size_t);
  int (*destroy_buffer)(void *);
  int (*add_buffer)(void *, void *);
  int (*execute)(uint32_t, const void *, void *);
} LKRuntime;

typedef struct {
  LKRuntime *runtime;
  void *context, *desc, *inputs, *outputs;
  void *device[3], *buffer[3];
  size_t bytes[3], input_index[2];
  uint32_t model;
  int loaded, retained, ready;
  char error[256];
} LKSession;

static moonbit_bytes_t error_bytes(const char *message) {
  size_t n = strlen(message);
  moonbit_bytes_t b = moonbit_make_bytes((int32_t)n, 0);
  memcpy(b, message, n);
  return b;
}

static int runtime_cleanup(LKRuntime *r) {
  r->error[0] = 0;
  if (r->sessions) {
    snprintf(r->error, sizeof(r->error), "close actor sessions before runtime");
    return -1;
  }
  if (r->initialized) {
    if (r->device >= 0) {
      int code = r->reset_device(r->device);
      if (code) { snprintf(r->error,sizeof(r->error),"aclrtResetDevice: %d",code); return -1; }
      r->device = -1;
    }
    int code = r->finalize();
    if (code) { snprintf(r->error, sizeof(r->error), "aclFinalize: %d", code); return -1; }
    r->initialized = 0;
  }
  if (r->library) { dlclose(r->library); r->library = NULL; }
  return 0;
}

static void runtime_finalizer(void *p) { (void)runtime_cleanup(p); }

MOONBIT_FFI_EXPORT LKRuntime *lk_runtime_open(moonbit_bytes_t path) {
  LKRuntime *r = moonbit_make_external_object(runtime_finalizer, sizeof(*r));
  memset(r, 0, sizeof(*r));
  r->device = -1;
  r->library = dlopen((const char *)path, RTLD_NOW | RTLD_LOCAL);
  if (!r->library) { snprintf(r->error, sizeof(r->error), "AscendCL unavailable: %s", dlerror()); return r; }
#define LOAD(field, symbol) do { \
  *(void **)(&r->field) = dlsym(r->library, symbol); \
  if (!r->field) { snprintf(r->error, sizeof(r->error), "missing AscendCL symbol: %s", symbol); return r; } \
} while (0)
  LOAD(init,"aclInit"); LOAD(finalize,"aclFinalize");
  LOAD(set_device,"aclrtSetDevice"); LOAD(reset_device,"aclrtResetDevice");
  LOAD(create_context,"aclrtCreateContext");
  LOAD(destroy_context,"aclrtDestroyContext"); LOAD(set_context,"aclrtSetCurrentContext");
  LOAD(allocate,"aclrtMalloc"); LOAD(release,"aclrtFree"); LOAD(copy,"aclrtMemcpy");
  LOAD(load,"aclmdlLoadFromFile"); LOAD(unload,"aclmdlUnload");
  LOAD(create_desc,"aclmdlCreateDesc"); LOAD(get_desc,"aclmdlGetDesc");
  LOAD(destroy_desc,"aclmdlDestroyDesc"); LOAD(num_inputs,"aclmdlGetNumInputs");
  LOAD(num_outputs,"aclmdlGetNumOutputs"); LOAD(input_index,"aclmdlGetInputIndexByName");
  LOAD(input_size,"aclmdlGetInputSizeByIndex"); LOAD(output_size,"aclmdlGetOutputSizeByIndex");
  LOAD(input_type,"aclmdlGetInputDataType"); LOAD(output_type,"aclmdlGetOutputDataType");
  LOAD(create_dataset,"aclmdlCreateDataset"); LOAD(destroy_dataset,"aclmdlDestroyDataset");
  LOAD(create_buffer,"aclCreateDataBuffer"); LOAD(destroy_buffer,"aclDestroyDataBuffer");
  LOAD(add_buffer,"aclmdlAddDatasetBuffer"); LOAD(execute,"aclmdlExecute");
#undef LOAD
  int code = r->init(NULL);
  if (code) { snprintf(r->error, sizeof(r->error), "aclInit: %d", code); return r; }
  r->initialized = 1;
  return r;
}

MOONBIT_FFI_EXPORT moonbit_bytes_t lk_runtime_error(LKRuntime *r) { return error_bytes(r->error); }
MOONBIT_FFI_EXPORT int lk_runtime_close(LKRuntime *r) { return runtime_cleanup(r); }

static int session_cleanup(LKSession *s) {
  s->ready = 0;
  if (!s->runtime) return 0;
  LKRuntime *r = s->runtime;
  if (s->context) {
    int code = r->set_context(s->context);
    if (code) { snprintf(s->error, sizeof(s->error), "close/set context: %d", code); return -1; }
  }
  // Synchronous execution means there is no outstanding device work here.
#define RELEASE(field, call) do { if (field) { \
  int code = (call); if (code) { snprintf(s->error,sizeof(s->error),"close resource: %d",code); return -1; } \
  field = NULL; } } while (0)
  RELEASE(s->inputs, r->destroy_dataset(s->inputs));
  RELEASE(s->outputs, r->destroy_dataset(s->outputs));
  for (int i=0;i<3;i++) {
    RELEASE(s->buffer[i], r->destroy_buffer(s->buffer[i]));
    RELEASE(s->device[i], r->release(s->device[i]));
  }
  RELEASE(s->desc, r->destroy_desc(s->desc));
  if (s->loaded) {
    int code = r->unload(s->model);
    if (code) { snprintf(s->error,sizeof(s->error),"aclmdlUnload: %d",code); return -1; }
    s->loaded = 0;
  }
  RELEASE(s->context, r->destroy_context(s->context));
#undef RELEASE
  if (s->retained) {
    r->sessions--; s->retained = 0; s->runtime = NULL; moonbit_decref(r);
  }
  return 0;
}

static void session_finalizer(void *p) { (void)session_cleanup(p); }

static int checked(LKSession *s, int code, const char *op) {
  if (!code) return 0;
  snprintf(s->error,sizeof(s->error),"%s: %d",op,code);
  return -1;
}

MOONBIT_FFI_EXPORT LKSession *lk_session_prepare(LKRuntime *r, moonbit_bytes_t path,
  int device, int history, int goals, int actions) {
  LKSession *s = moonbit_make_external_object(session_finalizer, sizeof(*s));
  memset(s,0,sizeof(*s));
  if (!r->initialized || history <= 0 || goals <= 0 || actions <= 0 ||
      history > INT_MAX/4 || goals > INT_MAX/4 || actions > INT_MAX/4) {
    snprintf(s->error,sizeof(s->error),"closed runtime or invalid input/output size"); return s;
  }
  if (r->device >= 0 && r->device != device) {
    snprintf(s->error,sizeof(s->error),"this runtime owns one selected Ascend device"); return s;
  }
  s->runtime = r; s->retained = 1; moonbit_incref(r); r->sessions++;
  s->bytes[0]=(size_t)history*4; s->bytes[1]=(size_t)goals*4; s->bytes[2]=(size_t)actions*4;
#define CHECK(call) do { if (checked(s,(call),#call)) goto failed; } while (0)
  if (r->device < 0) { CHECK(r->set_device(device)); r->device=device; }
  CHECK(r->create_context(&s->context,device));
  CHECK(r->set_context(s->context));
  CHECK(r->load((const char *)path,&s->model)); s->loaded=1;
  s->desc=r->create_desc();
  if (!s->desc) { snprintf(s->error,sizeof(s->error),"aclmdlCreateDesc failed"); goto failed; }
  CHECK(r->get_desc(s->desc,s->model));
  if (r->num_inputs(s->desc)!=2 || r->num_outputs(s->desc)!=1) {
    snprintf(s->error,sizeof(s->error),"actor model must have two inputs and one output"); goto failed;
  }
  CHECK(r->input_index(s->desc,"history",&s->input_index[0]));
  CHECK(r->input_index(s->desc,"goals",&s->input_index[1]));
  if (s->input_index[0]>1 || s->input_index[1]>1 || s->input_index[0]==s->input_index[1]) {
    snprintf(s->error,sizeof(s->error),"invalid history/goals model input mapping"); goto failed;
  }
  for (int i=0;i<2;i++) if (r->input_type(s->desc,s->input_index[i])!=0 ||
      r->input_size(s->desc,s->input_index[i])!=s->bytes[i]) {
    snprintf(s->error,sizeof(s->error),"model input must match static FP32 actor contract"); goto failed;
  }
  if (r->output_type(s->desc,0)!=0 || r->output_size(s->desc,0)!=s->bytes[2]) {
    snprintf(s->error,sizeof(s->error),"model output must match static FP32 actor contract"); goto failed;
  }
  s->inputs=r->create_dataset(); s->outputs=r->create_dataset();
  if (!s->inputs || !s->outputs) { snprintf(s->error,sizeof(s->error),"dataset creation failed"); goto failed; }
  for (int i=0;i<3;i++) {
    CHECK(r->allocate(&s->device[i],s->bytes[i],2)); // NORMAL_ONLY
    s->buffer[i]=r->create_buffer(s->device[i],s->bytes[i]);
    if (!s->buffer[i]) { snprintf(s->error,sizeof(s->error),"buffer creation failed"); goto failed; }
  }
  for (int index=0;index<2;index++) {
    int logical = s->input_index[0]==(size_t)index ? 0 : 1;
    CHECK(r->add_buffer(s->inputs,s->buffer[logical]));
  }
  CHECK(r->add_buffer(s->outputs,s->buffer[2]));
  s->ready=1;
  return s;
failed: {
    char original[256]; memcpy(original,s->error,sizeof(original));
    (void)session_cleanup(s);
    memcpy(s->error,original,sizeof(original));
    return s;
  }
#undef CHECK
}

MOONBIT_FFI_EXPORT moonbit_bytes_t lk_session_error(LKSession *s) { return error_bytes(s->error); }
MOONBIT_FFI_EXPORT int lk_session_close(LKSession *s) {
  s->error[0]=0;
  return session_cleanup(s);
}

MOONBIT_FFI_EXPORT int lk_session_run(LKSession *s, float *history, float *goals, float *actions) {
  s->error[0]=0;
  if (!s->ready || !s->runtime) { snprintf(s->error,sizeof(s->error),"actor session is closed"); return -1; }
  LKRuntime *r=s->runtime;
  if (checked(s,r->set_context(s->context),"aclrtSetCurrentContext") ||
      checked(s,r->copy(s->device[0],s->bytes[0],history,s->bytes[0],1),"history host-to-device") ||
      checked(s,r->copy(s->device[1],s->bytes[1],goals,s->bytes[1],1),"goals host-to-device") ||
      checked(s,r->execute(s->model,s->inputs,s->outputs),"aclmdlExecute") ||
      checked(s,r->copy(actions,s->bytes[2],s->device[2],s->bytes[2],2),"actions device-to-host")) return -1;
  return 0;
}
