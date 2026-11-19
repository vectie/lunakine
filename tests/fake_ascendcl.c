// Software-only AscendCL test double. No device or model math is simulated.
// Reversed model input order exercises name-based binding in the real loader.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int live;
typedef struct { void *data; size_t size; } Buffer;
typedef struct { Buffer *buffers[2]; int count; } Dataset;
static void *make(size_t n) { void *p=calloc(1,n); if(p)live++; return p; }
static int drop(void *p) { if(p){live--;free(p);}return 0; }
int aclInit(const char *unused) { (void)unused; return 0; }
int aclFinalize(void) { return live ? 91 : 0; }
int aclrtSetDevice(int32_t device) { return device==0 ? 0 : 92; }
int aclrtResetDevice(int32_t device) { return device==0 && live==0 ? 0 : 97; }
int aclrtCreateContext(void **p,int32_t device) { (void)device; *p=make(1);return *p?0:1; }
int aclrtDestroyContext(void *p) { return drop(p); }
int aclrtSetCurrentContext(void *p) { return p ? 0 : 93; }
int aclrtMalloc(void **p,size_t n,int policy) { (void)policy; *p=make(n);return *p?0:1; }
int aclrtFree(void *p) { return drop(p); }
int aclrtMemcpy(void *out,size_t capacity,const void *in,size_t n,int kind) {
  if(capacity<n || (kind!=1&&kind!=2))return 94; memcpy(out,in,n);return 0;
}
int aclmdlLoadFromFile(const char *path,uint32_t *id) { *id=strstr(path,"fail-execute")?2:1;live++;return 0; }
int aclmdlUnload(uint32_t id) { (void)id;live--;return 0; }
void *aclmdlCreateDesc(void) { return make(1); }
int aclmdlGetDesc(void *p,uint32_t id) { (void)id;return p?0:1; }
int aclmdlDestroyDesc(void *p) { return drop(p); }
size_t aclmdlGetNumInputs(const void *p) { (void)p;return 2; }
size_t aclmdlGetNumOutputs(const void *p) { (void)p;return 1; }
int aclmdlGetInputIndexByName(const void *p,const char *name,size_t *index) {
  (void)p;if(!strcmp(name,"history")){*index=1;return 0;}
  if(!strcmp(name,"goals")){*index=0;return 0;}return 95;
}
size_t aclmdlGetInputSizeByIndex(const void *p,size_t index) { (void)p;return index?8:4; }
size_t aclmdlGetOutputSizeByIndex(const void *p,size_t index) { (void)p;(void)index;return 4; }
int aclmdlGetInputDataType(const void *p,size_t index) { (void)p;(void)index;return 0; }
int aclmdlGetOutputDataType(const void *p,size_t index) { (void)p;(void)index;return 0; }
void *aclmdlCreateDataset(void) { return make(sizeof(Dataset)); }
int aclmdlDestroyDataset(void *p) { return drop(p); }
void *aclCreateDataBuffer(void *data,size_t n) { Buffer *b=make(sizeof(*b));b->data=data;b->size=n;return b; }
int aclDestroyDataBuffer(void *p) { return drop(p); }
int aclmdlAddDatasetBuffer(void *p,void *buffer) { Dataset *d=p;if(d->count==2)return 96;d->buffers[d->count++]=buffer;return 0; }
int aclmdlExecute(uint32_t model,const void *inputs,void *outputs) {
  if(model==2)return 77;
  const Dataset *in=inputs;Dataset *out=outputs;
  float *goal=in->buffers[0]->data,*history=in->buffers[1]->data;
  *(float *)out->buffers[0]->data=history[0]*2+history[1]+goal[0];
  return 0;
}
