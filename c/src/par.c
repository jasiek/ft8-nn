/* A pool of worker threads (internal.h par_*). The decode stages hand out their work items through an atomic counter,
 * and every item is computed exactly as the serial code did, so the results do not depend on the thread count.
 * POSIX threads here; the _WIN32 branch is for native Windows builds (MinGW-w64 gcc in CI), which use no pthreads. */
#include <stdatomic.h>
#include <stdlib.h>

#include "internal.h"

/* worker stack: fine sync keeps ~170 KB of arrays on the stack, and macOS gives other threads only 512 KB */
#define THREAD_STACK (8u << 20)
#define MAX_THREADS 64

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
typedef HANDLE thread_t;
typedef SRWLOCK mutex_t;
typedef CONDITION_VARIABLE cond_t;
static void mx_init(mutex_t *m) { InitializeSRWLock(m); }
static void mx_lock(mutex_t *m) { AcquireSRWLockExclusive(m); }
static void mx_unlock(mutex_t *m) { ReleaseSRWLockExclusive(m); }
static void mx_free(mutex_t *m) { (void)m; }
static void cv_init(cond_t *c) { InitializeConditionVariable(c); }
static void cv_wait(cond_t *c, mutex_t *m) { SleepConditionVariableSRW(c, m, INFINITE, 0); }
static void cv_signal(cond_t *c) { WakeConditionVariable(c); }
static void cv_broadcast(cond_t *c) { WakeAllConditionVariable(c); }
static void cv_free(cond_t *c) { (void)c; }
static void *worker(void *arg);
static DWORD WINAPI thr_main(LPVOID p) {
    worker(p);
    return 0;
}
static int thr_start(thread_t *t, void *arg) {
    return (*t = CreateThread(NULL, THREAD_STACK, thr_main, arg, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL)) != NULL;
}
static void thr_join(thread_t t) {
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}
static int cpu_count(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
}
#else
#include <pthread.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
typedef pthread_t thread_t;
typedef pthread_mutex_t mutex_t;
typedef pthread_cond_t cond_t;
static void mx_init(mutex_t *m) { pthread_mutex_init(m, NULL); }
static void mx_lock(mutex_t *m) { pthread_mutex_lock(m); }
static void mx_unlock(mutex_t *m) { pthread_mutex_unlock(m); }
static void mx_free(mutex_t *m) { pthread_mutex_destroy(m); }
static void cv_init(cond_t *c) { pthread_cond_init(c, NULL); }
static void cv_wait(cond_t *c, mutex_t *m) { pthread_cond_wait(c, m); }
static void cv_signal(cond_t *c) { pthread_cond_signal(c); }
static void cv_broadcast(cond_t *c) { pthread_cond_broadcast(c); }
static void cv_free(cond_t *c) { pthread_cond_destroy(c); }
static void *worker(void *arg);
static int thr_start(thread_t *t, void *arg) {
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, THREAD_STACK);
    int rc = pthread_create(t, &at, worker, arg);
    pthread_attr_destroy(&at);
    return rc == 0;
}
static void thr_join(thread_t t) { pthread_join(t, NULL); }
static int cpu_count(void) {
#ifdef __APPLE__ /* performance cores only: an efficiency core would hold up the last work items */
    int n = 0;
    size_t len = sizeof n;
    if (!sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, NULL, 0) && n > 0) return n;
#endif
    long n2 = sysconf(_SC_NPROCESSORS_ONLN);
    return n2 > 0 ? (int)n2 : 1;
}
#endif

typedef struct {
    par *p;
    int w;
} wslot;

struct par {
    int nt; /* threads including the caller */
    thread_t th[MAX_THREADS];
    wslot ws[MAX_THREADS];
    mutex_t m;
    cond_t go, done;
    void (*fn)(void *, int);
    void *arg;
    unsigned gen; /* bumped per par_run */
    int busy;     /* workers still in this run's fn */
    int quit;
};

static void *worker(void *arg) {
    wslot *ws = arg;
    par *p = ws->p;
    unsigned seen = 0;
    mx_lock(&p->m);
    for (;;) {
        while (p->gen == seen && !p->quit) cv_wait(&p->go, &p->m);
        if (p->quit) break;
        seen = p->gen;
        void (*fn)(void *, int) = p->fn;
        void *a = p->arg;
        mx_unlock(&p->m);
        fn(a, ws->w);
        mx_lock(&p->m);
        if (--p->busy == 0) cv_signal(&p->done);
    }
    mx_unlock(&p->m);
    return NULL;
}

par *par_new(int nthreads) {
    par *p = calloc(1, sizeof *p);
    int nt = nthreads > 0 ? nthreads : cpu_count();
    if (nt > MAX_THREADS) nt = MAX_THREADS;
    mx_init(&p->m);
    cv_init(&p->go);
    cv_init(&p->done);
    p->nt = 1;
    for (int w = 1; w < nt; w++) { /* a thread that cannot be started is simply not used */
        p->ws[p->nt] = (wslot){p, p->nt};
        if (!thr_start(&p->th[p->nt], &p->ws[p->nt])) break;
        p->nt++;
    }
    return p;
}

void par_free(par *p) {
    if (!p) return;
    mx_lock(&p->m);
    p->quit = 1;
    cv_broadcast(&p->go);
    mx_unlock(&p->m);
    for (int w = 1; w < p->nt; w++) thr_join(p->th[w]);
    mx_free(&p->m);
    cv_free(&p->go);
    cv_free(&p->done);
    free(p);
}

int par_threads(const par *p) { return p ? p->nt : 1; }

void par_run(par *p, void (*fn)(void *, int), void *arg) {
    if (!p || p->nt == 1) {
        fn(arg, 0);
        return;
    }
    mx_lock(&p->m);
    p->fn = fn;
    p->arg = arg;
    p->busy = p->nt - 1;
    p->gen++;
    cv_broadcast(&p->go);
    mx_unlock(&p->m);
    fn(arg, 0);
    mx_lock(&p->m);
    while (p->busy) cv_wait(&p->done, &p->m);
    mx_unlock(&p->m);
}

typedef struct {
    void (*body)(void *, int);
    void *arg;
    int n;
    atomic_int next;
} for_job;

static void for_worker(void *a, int w) {
    for_job *j = a;
    for (int i; (i = atomic_fetch_add(&j->next, 1)) < j->n;) j->body(j->arg, i);
}

void par_for(par *p, int n, void (*body)(void *, int), void *arg) {
    if (!p || p->nt == 1 || n <= 1) {
        for (int i = 0; i < n; i++) body(arg, i);
        return;
    }
    for_job j = {.body = body, .arg = arg, .n = n};
    atomic_init(&j.next, 0);
    par_run(p, for_worker, &j);
}
