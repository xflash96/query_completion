// vim: noai:ts=4:sw=4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <assert.h>
#include <signal.h>
#include <xmmintrin.h>
#include <time.h>
#ifndef __unix__
#include <sys/time.h>
#endif
#include <omp.h>

#include "mkl.h"
#include "model.h"

const char *charset = "abcdefghijklmnopqrstuvwxyz0123456789 .-%_:/\\$";
int char_index[256];

// some ancient compiler does not support C11, so we have to supply this function ourselves
void *aligned_alloc(size_t alignment, size_t size) {
    void *ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}

// The LSTM struct implementing Kera's LSTM
typedef struct lstm_t {
    float *kernel;        // concat of Kera's kernel and recurrent kernel
    float *bias;

    float **c, **h;
    float **z, **t;
    float **cbuf, **hbuf; // c and h buffer for beam search

    int n_in, n_hid, max_batch; // # of (input, hidden) units, and max batch size
} lstm_t;

// A softmax layer with linear input
typedef struct softmax_t {
    float *W, *bias;
    float **out;

    int n_in, n_out, max_batch;
} softmax_t;

// The NN model 
typedef struct model_t {
    float **in;             // The input
    lstm_t lstm_1, lstm_2;  // The two layer LSTM
    softmax_t softmax;      // THe output softmas layer (w/ linear unit)

    float dropout;          // The fraction of dropout
    int n_in, n_hid, max_batch;
} model_t;

// Struct for completion distance
typedef struct dist_t {
    int *seq;
    float *dist, *dist_new;
    int pos, len, extend;
} dist_t;

// Struct for omnicompletion
typedef struct qcomp_t {
    int **cand, **result, **buf;
    float *cand_score, *result_score, **new_score;
    int *rank, *next_char;
    dist_t *dist, *dist_buf;

    int max_batch, max_len;
} qcomp_t;

// The Trie
typedef struct list_t {
    struct list_t *next, *prev;
    struct list_t *child, *parent;
    struct list_t **top; // pointers to 16 most frequent strings in subtries
    int key, val;
    double weight;
} list_t;


/*********** time utils **********/

#define NS_PER_SEC 1000000000
int64_t wall_clock_ns()
{
#ifdef __unix__
	struct timespec tspec;
	int r = clock_gettime(CLOCK_MONOTONIC, &tspec);
	assert(r==0);
	return tspec.tv_sec*NS_PER_SEC + tspec.tv_nsec;
#else
    struct timeval tv;
	int r = gettimeofday( &tv, NULL );
	assert(r==0);
	return tv.tv_sec*NS_PER_SEC + tv.tv_usec*1000;
#endif
}

double wall_time_diff(int64_t ed, int64_t st)
{
    return (double)(ed-st)/(double)NS_PER_SEC;
}

/************* BLAS utils *************/

void mysgemm(int m, int n, int k, float alpha, const float *restrict A, const float *restrict B, float beta, float *restrict C)
{ cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, A, k, B, k, beta, C, n); }

void mysaxpy(int n, float a, const float *restrict x , float *restrict y)
{ cblas_saxpy(n, a, x, 1, y, 1); }

void myszero(int n, float *x)
{ memset(x, 0, n*sizeof(*x)); }

void myscopy(int n, const float *restrict x, float *restrict y)
{ memcpy(y, x, n*sizeof(*x)); }

void myclip(int n, float *x)
{
    x = __builtin_assume_aligned(x, 8*sizeof(float));

    const __m128 mzero = _mm_set1_ps(0);
    const __m128 mone  = _mm_set1_ps(1);
    for(int i=0; i<n; i+=8, x+=8){
        __m128 a = _mm_load_ps(x);
        __m128 b = _mm_load_ps(x+4);
        a = _mm_max_ps(a, mzero);
        b = _mm_max_ps(b, mzero);
        a = _mm_min_ps(a, mone);
        b = _mm_min_ps(b, mone);
        _mm_store_ps(x, a);
        _mm_store_ps(x+4, b);
    }
}

// allocate 2D matrix with aligned bytes
float **smat2d_init(int m, int n)
{
    float **a = calloc(m, sizeof(*a));
    *a = aligned_alloc(8*sizeof(float), m*n* sizeof(**a));
    for(int i=1; i<m; i++)
        a[i] = a[0] + i*n;
    return a;
}

// allocating integer 2d matrix
int **imat2d_init(int m, int n)
{
    int **a = calloc(m, sizeof(*a));
    *a = calloc(m*n, sizeof(**a));
    for(int i=1; i<m; i++) a[i] = a[0]+n*i;
    return a;
}

/************* lstm_t ***********/
lstm_t lstm_init(int max_batch, int n_in, int n_hid)
{
    lstm_t self = {
        .kernel = aligned_alloc(8*sizeof(float), 4*n_hid*(n_in+n_hid) * sizeof(float)),
        .bias   = aligned_alloc(8*sizeof(float), 4*n_hid * sizeof(float)),
        .c      = smat2d_init(max_batch, n_hid),
        .h      = smat2d_init(max_batch, n_hid),
        .z      = smat2d_init(max_batch, 4*n_hid),
        .t      = smat2d_init(max_batch, n_in+n_hid),
        .cbuf   = smat2d_init(max_batch, n_hid),
        .hbuf   = smat2d_init(max_batch, n_hid),
        .n_in   = n_in,
        .n_hid  = n_hid,
        .max_batch = max_batch,
    };

    return self;
}

// loading LSTM weights from model.h
// Note that we swapped z2 and z3 in the keras impl
void lstm_load(lstm_t self, const float *kernel, const float *rec_kernel, const float *bias)
{
    int line = self.n_in + self.n_hid;
    for(int ii=0; ii<4*self.n_hid; ii++){
        int i = ii;
        // swap z2 and z3
        if(ii >= 3*self.n_hid) i -= self.n_hid;
        else if(ii >= 2*self.n_hid) i += self.n_hid;

        for(int j=0; j<self.n_in; j++) self.kernel[ii*line+j] = kernel[j*4*self.n_hid+i];
        for(int j=0; j<self.n_hid; j++) self.kernel[ii*line+self.n_in+j] = rec_kernel[j*4*self.n_hid+i];
    }
    myscopy(2*self.n_hid, bias, self.bias);
    // swap z2 and z3
    myscopy(self.n_hid, bias+2*self.n_hid, self.bias+3*self.n_hid);
    myscopy(self.n_hid, bias+3*self.n_hid, self.bias+2*self.n_hid);
}

// The forward pass for LSTM
// Note that the input (in) is scaled by in_alpha
void lstm_forward(lstm_t self, int batch, float **in, float in_alpha)
{
    // Applying linear ops to the input (in)
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<batch; i++){
        myszero(self.n_in, self.t[i]);
        mysaxpy(self.n_in, in_alpha, in[i], self.t[i]);
        myscopy(self.n_hid, self.h[i], self.t[i]+self.n_in);
    }
    mysgemm(batch, 4*self.n_hid, self.n_in+self.n_hid, 1, *self.t, self.kernel, 0, *self.z);

    // Adding bias to input, then do hard-sigmoid
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<batch; i++){
        float *z = self.z[i];
        mysaxpy(4*self.n_hid, 1., self.bias, z);
        for(int j=0; j<3*self.n_hid; j++) z[j] = 0.2*z[j] + 0.5;
        myclip(3*self.n_hid, z);
    }

    // Eval next c and h
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<batch; i++){
        float *c = self.c[i], *h = self.h[i];
        float *z0 = self.z[i] + self.n_hid*0, *z1 = self.z[i] + self.n_hid*1;
        float *z2 = self.z[i] + self.n_hid*2, *z3 = self.z[i] + self.n_hid*3;
        for(int j=0; j<self.n_hid; j++){
            c[j] = z1[j]*c[j] + z0[j]*tanh(z3[j]);
            h[j] = z2[j]*tanh(c[j]);
        }
    }
}

// Backup current c and h to buffer
void lstm_backup(lstm_t self)
{
    myscopy(self.max_batch*self.n_hid, *self.c, *self.cbuf);
    myscopy(self.max_batch*self.n_hid, *self.h, *self.hbuf);
}

// Recover c and h from buffer
void lstm_recover(lstm_t self, int src, int dst)
{
    myscopy(self.n_hid, self.cbuf[src], self.c[dst]);
    myscopy(self.n_hid, self.hbuf[src], self.h[dst]);
}

// Reset all c and h to zero
void lstm_reset(lstm_t self)
{
    memset(*self.c, 0, self.max_batch*self.n_hid*sizeof(**self.c));
    memset(*self.h, 0, self.max_batch*self.n_hid*sizeof(**self.h));
}

/************* softmax_t **********/
softmax_t softmax_init(int max_batch, int n_in, int n_out)
{
    softmax_t self = {
        .W      = aligned_alloc(8*sizeof(float), n_in*n_out * sizeof(float)),
        .bias   = calloc(4*sizeof(float), n_out * sizeof(float)),
        .out    = smat2d_init(max_batch, n_out),
        .n_in   = n_in,
        .n_out  = n_out,
        .max_batch = max_batch,
    };
    return self;
}

// Loading the linear weights of softmax layer
void softmax_load(softmax_t self, const float *kernel, const float *bias)
{
    for(int i=0; i<self.n_out; i++)
        for(int j=0; j<self.n_in; j++) self.W[i*self.n_in+j] = kernel[j*self.n_out+i];
    myscopy(self.n_out, bias, self.bias);
}

// The forward pass of softmax
void softmax_forward(softmax_t self, int batch, float **in, float in_alpha)
{
    mysgemm(batch, self.n_out, self.n_in, in_alpha, *in, self.W, 0, *self.out);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<batch; i++){
        float *out = self.out[i];
        mysaxpy(self.n_out, 1, self.bias, out);

        // stable log-sum-exp
        float max = out[0];
        for(int j=1; j<self.n_out; j++) if(max<out[j]) max = out[j];
        float sum_exp = 0;
        for(int j=0; j<self.n_out; j++) sum_exp += exp(out[j]-max);
        float log_sum_exp = log(sum_exp);
        for(int j=0; j<self.n_out; j++) out[j] -= log_sum_exp+max;
    }
}


/************* model_t **********/
model_t model_init(int max_batch, int n_in, int n_hid)
{
    model_t self = {
        .in     = smat2d_init(max_batch, n_in),
        .lstm_1 = lstm_init(max_batch, n_in, n_hid),
        .lstm_2 = lstm_init(max_batch, n_hid, n_hid),
        .softmax = softmax_init(max_batch, n_hid, n_in),
        .dropout = 1,
        .n_in = n_in,
        .n_hid = n_hid,
        .max_batch = max_batch,
    };
    return self;
}

// Loading all the weights for the model
void model_load(model_t self)
{
    lstm_load(self.lstm_1, LSTM_1_KERNEL, LSTM_1_RECURRENT_KERNEL, LSTM_1_BIAS);
    lstm_load(self.lstm_2, LSTM_2_KERNEL, LSTM_2_RECURRENT_KERNEL, LSTM_2_BIAS);
    softmax_load(self.softmax, DENSE_1_KERNEL, DENSE_1_BIAS);
}

// The forward pass for the model
void model_forward(model_t self, int batch, int *x)
{
    // setup the onehot encodings for inputs
    memset(*self.in, 0, self.max_batch*self.n_in*sizeof(float));
    for(int i=0; i<batch; i++) self.in[i][x[i]] = 1;

    lstm_forward(self.lstm_1, batch, self.in, 1);
    lstm_forward(self.lstm_2, batch, self.lstm_1.h, 1/self.dropout);
    softmax_forward(self.softmax, batch, self.lstm_2.h, 1/self.dropout);
}

// Reset the recurrent units of the model
void model_reset(model_t self)
{
    lstm_reset(self.lstm_1);
    lstm_reset(self.lstm_2);
}

/************* dist_t **********/
dist_t dist_init(int max_len)
{
    dist_t self = {
        .seq = calloc(max_len+1, sizeof(int)),
        .dist = calloc(max_len+1, sizeof(float)),
        .dist_new = calloc(max_len+1, sizeof(float)),
        .pos = 0,
        .len = 0,
        .extend = 0,
    };
    return self;
}

// Setup the first column for the distance matrix
dist_t dist_build(dist_t self, int *a)
{
    int len = 0;
    for(; a[len]; len++)
        ;
    self.pos = 1;
    self.len = len;

    for(int j=0; j<=len; j++){
        self.dist[j] = j;
        self.seq[j] = a[j];
    }
    return self;
}

// Hard-coded index (charset + PAD/END) for space symbol
#define SPACE_SYMBOL (2+26+10)

// Bump the distance matrix by one column (DP-ish)
// and store the result in dist_new
dist_t dist_bump(dist_t self, int next_b)
{
    self.extend = 1;
    self.dist_new[0] = self.pos;
    for(int j=1; j<=self.len; j++){
        // white space or end
        int is_last = self.seq[j] == SPACE_SYMBOL || self.seq[j] == 0;
        int is_same = (next_b == self.seq[j-1]);
        float add = self.dist[j] + !is_last;
        float sub = self.dist[j-1] + !is_same;
        float del = self.dist_new[j-1] + 1;

        float cost = add;
        if(cost > sub) cost = sub;
        if(cost > del) cost = del;
        self.dist_new[j] = cost;
    }
    return self;
}

// Commit dist_new to dist
dist_t dist_commit(dist_t self)
{
    myscopy(self.len+1, self.dist_new, self.dist);
    self.pos++;

    return self;
}

// Copying (branching) the distance matirx
void dist_copy(dist_t *src, dist_t *dst)
{
    for(int i=0; i<=src->len; i++){
        dst->seq[i] = src->seq[i];
        dst->dist[i] = src->dist[i];
        dst->dist_new[i] = src->dist_new[i];
    }
    dst->len = src->len;
    dst->pos = src->pos;
    dst->extend = src->extend;
}

// The difference between dist_new and dist
float dist_diff(dist_t self)
{
    return self.dist_new[self.len] - self.dist[self.len];
}

/************* qcomp_t ***********/
qcomp_t qcomp_init(int max_batch, int max_len)
{
    int voc_size = strlen(charset)+2;
    qcomp_t self = {
        .cand   = imat2d_init(max_batch, max_len),
        .result = imat2d_init(max_batch, max_len),
        .buf    = imat2d_init(max_batch, max_len),
        .cand_score     = calloc(max_batch, sizeof(float)),
        .result_score   = calloc(max_batch, sizeof(float)),
        .new_score      = smat2d_init(max_batch, voc_size),
        .rank   = calloc(max_batch*voc_size, sizeof(int)),
        .next_char = calloc(max_batch, sizeof(int)),
        .dist   = calloc(max_batch, sizeof(dist_t)),
        .dist_buf = calloc(max_batch, sizeof(dist_t)),
        .max_batch = max_batch,
        .max_len = max_len,
    };

    for(int i=0; i<strlen(charset); i++)
        char_index[(unsigned)charset[i]] = i+2; // after PAD and END

    for(int i=0; i<max_batch; i++){
        self.dist[i] = dist_init(max_len);
        self.dist_buf[i] = dist_init(max_len);
    }

    return self;
}

// Encode string to zero-terminating index sequence
void qcomp_encode(qcomp_t self, const char *str, int *seq, int transparent)
{
    int i;
    if (transparent) 
        for(i=0; i<strlen(str); i++) seq[i] = ((unsigned char)str[i]) + 2;
    else
        for(i=0; i<strlen(str); i++) seq[i] = char_index[(unsigned)str[i]];
    seq[i] = 0;
}

// Decode index sequence to string
void qcomp_decode(qcomp_t self, const int *seq, char *str, int transparent)
{
    int i=0;
    if (transparent) 
        for(i=0; seq[i]>1 && i < 60; i++) str[i] = seq[i] - 2;
    else
        for(i=0; seq[i]>1 && i < 60; i++) str[i] = charset[seq[i]-2];
    str[i] = '\0';
}

// Comparator for two index by SCORE[index]
float *SCORE;
int argcmp(const void *x, const void *y)
{
    int ix = *((int*)x), iy = *((int*)y);
    float fx = SCORE[ix], fy = SCORE[iy];
    if(fx<fy) return 1;
    else if (fx>fy) return -1;
    else return 0;
}

// Argsort, returning index matrix sorted by score[index]
void argsort(float *score, int *rank, int n)
{
    for(int i=0; i<n; i++) rank[i] = i;
    SCORE=score;
    qsort(rank, n, sizeof(*rank), argcmp);
}

// Safely copying zero-terminating index sequence
void seq_copy(int *src, int *dst, int max)
{
    for(int i=0; src[i] && i<max; i++) dst[i] = src[i];
}

// Boltzman sampling with temperature (scaling)
int boltzman_sampling(int n, const float *log_softmax, float temp)
{
    float *prob = calloc(n, sizeof(float));
    for(int i=0; i<n; i++) prob[i] = log_softmax[i] / temp;
    float sum_exp = 0;
    for(int i=0; i<n; i++) sum_exp += exp(prob[i]);
    float log_sum_exp = log(sum_exp);
    for(int i=0; i<n; i++) prob[i] = exp(prob[i]-log_sum_exp);

    float t = ((float)random())/(RAND_MAX);
    int itval;
    float acc = prob[0];
    for(itval=1; itval<n; acc += prob[itval++]) if(acc > t) break;

    free(prob);
    return itval-1;
}

// Stochastic search
void qcomp_stocsearch(qcomp_t self, model_t model, int *prefix)
{
    int voc_size = model.n_in;
    for(int j=0; j<self.max_batch; j++){
        model_reset(model);
        int i;
        for(i=0; prefix[i]; i++){
            self.result[j][i] = self.next_char[0] = prefix[i];
            model_forward(model, 1, self.next_char);
        }
        float score = 0;
        for(; self.next_char[0]>1 && i < self.max_len; i++){
            int sample = boltzman_sampling(voc_size, model.softmax.out[0], 1);
            self.result[j][i] = self.next_char[0] = sample;
            score += model.softmax.out[0][sample];
            model_forward(model, 1, self.next_char);
        }
        self.result_score[j] = score;
    }
}

// Beam search
// inspired by https://gist.github.com/udibr/67be473cf053d8c38730
int qcomp_beamsearch(qcomp_t self, model_t model, int *prefix)
{
    int n_cand = 1, n_result = 0;
    int voc_size = model.n_in;

    seq_copy(prefix, self.cand[0], self.max_len);
    self.cand_score[0] = 0;

    model_reset(model);
    int pos=0;
    for(int i=0; prefix[i]; i++, pos++){
        self.cand[0][i] = self.next_char[0] = prefix[i];
        model_forward(model, 1, self.next_char);
    }

    for(; n_cand && pos < self.max_len; pos++){
        for(int i=0; i<n_cand; i++){
            for(int j=0; j<voc_size; j++)
                self.new_score[i][j] = self.cand_score[i] + model.softmax.out[i][j];
        }
        argsort(*self.new_score, self.rank, voc_size*n_cand);

        n_cand = self.max_batch - n_result;
        for(int i=0; i<n_cand; i++)
            seq_copy(self.cand[i], self.buf[i], self.max_len);
        lstm_backup(model.lstm_1);
        lstm_backup(model.lstm_2);

        int n_zombie = 0;
        for(int r=0; r<n_cand; r++){
            int idx = self.rank[r];
            int i = idx / voc_size, j = idx % voc_size;
            if (j <= 1) {
                self.result_score[n_result] = (*self.new_score)[idx];
                seq_copy(self.buf[i], self.result[n_result], self.max_len);
                self.result[n_result][pos] = j;
                n_result++;
                n_zombie++;
                continue;
            }
            int dst  = r - n_zombie;
            seq_copy(self.buf[i], self.cand[dst], self.max_len);
            lstm_recover(model.lstm_1, i, dst);
            lstm_recover(model.lstm_2, i, dst);

            self.cand[dst][pos] = self.next_char[dst] = j;
            self.cand_score[dst] = (*self.new_score)[idx];
        }
        n_cand -= n_zombie;
        model_forward(model, n_cand, self.next_char);
    }
    return n_result;
}

// Omni search
int qcomp_omnisearch(qcomp_t self, model_t model, int *prefix)
{
    int n_cand = 1, n_result = 0;
    int voc_size = model.n_in;

    double forward_time = 0;

    seq_copy(prefix, self.cand[0], self.max_len);
    self.cand_score[0] = 0;
    self.dist[0] = dist_build(self.dist[0], prefix); 

    model_reset(model);
    int pos=0;
    for(int i=0; prefix[i] && i<2; i++, pos++){
        self.cand[0][i] = self.next_char[0] = prefix[i];
        model_forward(model, 1, self.next_char);
        self.dist[0] = dist_bump(self.dist[0], prefix[i]);
        self.dist[0] = dist_commit(self.dist[0]);
    }
    self.cand_score[0] = -4*self.dist[0].dist[self.dist[0].len];

    for(; n_cand && pos < self.max_len; pos++){
        #pragma omp parallel for schedule(dynamic)
        for(int i=0; i<n_cand; i++){
            for(int j=0; j<voc_size; j++){
                self.dist[i] = dist_bump(self.dist[i], j);
                self.new_score[i][j] = self.cand_score[i] + model.softmax.out[i][j] * self.dist[i].extend;
                self.new_score[i][j] -= 4*dist_diff(self.dist[i]);
            }
        }
        argsort(*self.new_score, self.rank, voc_size*n_cand);

        n_cand = self.max_batch - n_result;
        for(int i=0; i<n_cand; i++)
            seq_copy(self.cand[i], self.buf[i], self.max_len);
        lstm_backup(model.lstm_1);
        lstm_backup(model.lstm_2);
        for(int i=0; i<n_cand; i++)
            dist_copy(self.dist+i, self.dist_buf+i);

        int n_zombie = 0;
        for(int r=0; r<n_cand; r++){
            int idx = self.rank[r];
            int i = idx / voc_size, j = idx % voc_size;
            if (j <= 1) {
                self.result_score[n_result] = (*self.new_score)[idx];
                seq_copy(self.buf[i], self.result[n_result], self.max_len);
                self.result[n_result][pos] = j;
                n_result++;
                n_zombie++;
                continue;
            }
            int dst  = r - n_zombie;
            seq_copy(self.buf[i], self.cand[dst], self.max_len);
            lstm_recover(model.lstm_1, i, dst);
            lstm_recover(model.lstm_2, i, dst);
            dist_copy(self.dist_buf+i, self.dist+dst);

            self.cand[dst][pos] = self.next_char[dst] = j;
            self.cand_score[dst] = (*self.new_score)[idx];
            self.dist[dst] = dist_bump(self.dist[dst], j);
            self.dist[dst] = dist_commit(self.dist[dst]);
        }
        n_cand -= n_zombie;
        int64_t time_st = wall_clock_ns();
        model_forward(model, n_cand, self.next_char);
        int64_t time_ed = wall_clock_ns();
        forward_time += wall_time_diff(time_ed, time_st);
    }
    fprintf(stderr, "forward time %f\n", forward_time);
    return n_result;
}

const int END=1;
enum{VERTICAL, HORIZONTAL};
/************* trie_t *************/
list_t* trie_init(const char *fname)
{
    char *buf = malloc(2048+20);
    FILE *fin = fopen(fname, "r");
    list_t *root = calloc(1, sizeof(list_t));
    int trie_size = 1;

    while(fgets(buf, 2048, fin) != NULL){
        // Handling oversize sequence, strip end-lines,
        // and appending END symbol
        int len = strlen(buf);
        if(len>60) len = 60, buf[len] = '\0';
        if(buf[len-1] == '\n') buf[--len] = '\0';
        buf[len++] = END;
        buf[len] = '\0';

        // Setup weights
        int count = 0;
        int pos = 0;
        // Counting # of tabs
        for (int i = 0; i < len; ++i) {
            if(buf[i] == '\t') {
                pos = i;
                count++;
            }
        }
        // If we find two tabs, after the second tab it is the weight
        // Otherwise the weight is 1
        float weight = 1.0f;
        if (count == 2) {
            buf[pos] = END;
            weight = atof(buf+pos+1);
        }
        buf[pos+1] = '\0';

        // Going down the Trie
        list_t *p = root;
        int link;
        char *s = buf;
        for(; *s; s++){
            for(; p->next != NULL && p->key != *s; p = p->next)
                ;

            if(p->key != *s){
                link = HORIZONTAL;
                break;
            }
            p->val++;
            p->weight += weight;
            if(p->child == NULL){
                link = VERTICAL;
                s++;
                break;
            }
            p = p->child;
        }

        // Create nodes if not matched
        for(; *s; s++){
            list_t *old = p;
            p = calloc(1, sizeof(list_t));
            trie_size++;
            p->key = *s;
            p->val = 1;
            p->weight = weight;
            if(link==VERTICAL)
                old->child = p, p->parent = old;
            else
                old->next = p, p->prev = old;
            link = VERTICAL;
        }
    }

    fprintf(stderr, "trie size = %d\n", trie_size);

    free(buf);

    return root;
}

int min(int a, int b)
{
    if(a<b) return a;
    else return b;
}

list_t *MERGE_BUF[16];

// Merging the most frequent pointers (the **top)
void trie_merge_top(list_t *a, list_t *b)
{
    int size_a = min(16, a->val);
    int size_b = min(16, b->val);

    int i=0, k=0;
    for(int j=0; i<size_a && j<size_b && k<size_a; k++){
        list_t *pa = a->top[i], *pb = b->top[j];
        if(!pa || (pb && pa->weight < pb->weight)) MERGE_BUF[k] = pb, j++;
        else MERGE_BUF[k] = pa, i++;
    }
    for(; k<size_a; i++, k++) MERGE_BUF[k] = a->top[i];

    for(int i=0; i<size_a; i++){
        a->top[i] = MERGE_BUF[i];
    }
}

// Build the top list by recursively calling merge
void trie_build(list_t *root)
{
    int top_size = min(16, root->val);
    root->top = calloc(top_size, sizeof(*root->top));

    // leave node
    if(root->key == END){
        root->top[0] = root;
    }
    
    if (root->child) {
        trie_build(root->child);
        for(list_t *p=root->child; p; p = p->next){
            trie_merge_top(root, p);
        }
    }
    if(root->next) trie_build(root->next); 
}

// Lookup s in the trie,
// returning trie node if found and null otherwise
list_t *trie_lookup(list_t *root, char *s)
{
    list_t *p = root;
    while(*s){
        for(; p && p->key != *s; p = p->next)
            ;
        if(!p) break;
        s++;
        if(!*s) break;
        p = p->child;
    }
    return p;
}

// Trie search
int qcomp_triesearch(qcomp_t qcomp, list_t *root, int *seq)
{
    char prefix[258], r[258], buf[258];
#ifdef TRANSPARENT_TRIE
    qcomp_decode(qcomp, seq, prefix, 1);
#else
    qcomp_decode(qcomp, seq, prefix, 0);
#endif
    list_t *q = trie_lookup(root, prefix);
    if(!q){
        for(int i=0; i<qcomp.max_batch; i++)
            qcomp.result[i][0] = qcomp.result_score[i] = 0;
        return 0;
    }
    for(int i=0; i<qcomp.max_batch; i++){
        if(i>=q->val || !q->top[i]){
            qcomp.result_score[i] = 0;
            qcomp.result[i][0] = 0;
            continue;
        }
        qcomp.result_score[i] = log(q->top[i]->weight)-log(q->weight);

        int len=0;
        for(list_t *p = q->top[i]; p != q; ){
            r[len++] = p->key;
            while(!p->parent)
                p = p->prev;
            p = p->parent;
        }
        for(int j=0, k=len-1, t; j<k; j++, k--) t = r[j], r[j] = r[k], r[k] = t;
        r[len-1] = 0;
        memset(buf, 0, 256);
        strcat(buf, prefix);
        strcat(buf, r);
#ifdef TRANSPARENT_TRIE
        qcomp_encode(qcomp, buf, qcomp.result[i], 1);
#else
        qcomp_encode(qcomp, buf, qcomp.result[i], 0);
#endif
    }
    return q->val;
}

enum{STOCSEARCH, BEAMSEARCH, OMNISEARCH, TRIESEARCH, TRIELOOKUP};
/*****************************/
int main(int argc, char **argv)
{
    srand((unsigned)0);

    omp_set_num_threads(8);
    int n_beam = 16, max_len = 60;
    int voc_size = LSTM_1_KERNEL_SHAPE_0;
    model_t model = model_init(n_beam, voc_size, 256);
    model_load(model);
    qcomp_t qcomp = qcomp_init(n_beam, max_len);

    char *prefix = malloc(2048), prev[62] = {0};
    char str[62];
    int seq[62], rank[16];
    
    const char *prog_name = argv[0];
    const char *base_name = prog_name+strlen(prog_name)-1;
    for(; base_name!=prog_name && isalnum(*base_name); base_name--)
        ;
    base_name += 1;

    int mode = 0;
    int transparent = 0;
    // Different program names would invoke different functionalities
    if(strcmp(base_name, "stocsearch") == 0)
        mode = STOCSEARCH;
    else if(strcmp(base_name, "beamsearch") == 0)
        mode = BEAMSEARCH;
    else if(strcmp(base_name, "omnisearch") == 0)
        mode = OMNISEARCH;
    else if(strcmp(base_name, "triesearch") == 0) {
        mode = TRIESEARCH;
#ifdef TRANSPARENT_TRIE
        transparent = 1;
#endif
    }
    else if(strcmp(base_name, "trielookup") == 0) {
        mode = TRIELOOKUP;
#ifdef TRANSPARENT_TRIE
        transparent = 1;
#endif
    }
    else
        fprintf(stderr, "ERROR CMD\n"), exit(1);
    fprintf(stderr, "%s mode %d\n", base_name, mode);

    list_t *root = NULL;
    if(mode == TRIESEARCH || mode == TRIELOOKUP){
        root = trie_init(argv[1]);
        trie_build(root);
    }
    int to_end = 0;
    if(argc >= 3 && strcmp(argv[2], "1") == 0) to_end = 1;
    signal(SIGPIPE, SIG_IGN);
    fprintf(stderr, "ready\n");
    while(NULL != fgets(prefix, 2000, stdin)){
        int len = strlen(prefix);
        if(len>60) len = 60, prefix[len] = '\0';
        if(prefix[len-1] == '\n') prefix[len-1] = '\0';
        qcomp_encode(qcomp, prefix, seq, transparent);


		int64_t st = wall_clock_ns();
        // if(strcmp(prefix, prev) == 0 && mode != TRIELOOKUP)
        //    fprintf(stderr, "repeat\n");
        if(mode==STOCSEARCH)
            qcomp_stocsearch(qcomp, model, seq);
        else if(mode==BEAMSEARCH)
            qcomp_beamsearch(qcomp, model, seq);
        else if(mode==OMNISEARCH)
            qcomp_omnisearch(qcomp, model, seq);
        else if(mode==TRIESEARCH) {
            int count = qcomp_triesearch(qcomp, root, seq);
            // also count and print to stderr
            fprintf(stderr, "%d\n", count);
            fflush(stderr);
        }
        else if(mode==TRIELOOKUP){
            int len = strlen(prefix);
            if(to_end) prefix[len++] = END;
            prefix[len] = '\0';

            list_t *p = trie_lookup(root, prefix);
            if(!p) printf("0\n");
            else printf("%d %f\n", p->val, p->weight);
            fflush(stdout);
            continue;
        }

        argsort(qcomp.result_score, rank, n_beam);

        for(int jj=0; jj<n_beam; jj++){
            int j = rank[jj];
            qcomp_decode(qcomp, qcomp.result[j], str, transparent);
            printf("%f:\t%s\n", qcomp.result_score[j], str);
        }
        fflush(stdout);
        strcpy(prev, prefix);
    }
    fprintf(stderr, "exit\n");

    return 0;
}
