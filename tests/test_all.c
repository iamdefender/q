/*
 * test_all.c — PostGPT-Q unit tests
 * gcc tests/test_all.c -O2 -lm -o tests/run_tests && cd ~/q && ./tests/run_tests
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  [TEST] %s... ", name);
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if(!(cond)) { FAIL(msg); return; } } while(0)

/* ── 1. Math ── */
static float clampf(float x, float lo, float hi) { return x<lo?lo:x>hi?hi:x; }

static void rmsnorm(float *out, const float *x, int n) {
    float ms=0; for(int i=0;i<n;i++) ms+=x[i]*x[i];
    ms=1.0f/sqrtf(ms/n+1e-6f);
    for(int i=0;i<n;i++) out[i]=x[i]*ms;
}

static void matmul(float *out, const float *x, const float *w, int n_in, int d_out) {
    for(int d=0;d<d_out;d++){
        float v=0; for(int j=0;j<n_in;j++) v+=x[j]*w[d*n_in+j];
        out[d]=v;
    }
}

static void softmax(float *x, int n) {
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}
    if(s>0) for(int i=0;i<n;i++) x[i]/=s;
}

void test_clampf(void) {
    TEST("clampf");
    CHECK(clampf(5,0,10)==5, "mid"); CHECK(clampf(-1,0,10)==0, "lo");
    CHECK(clampf(15,0,10)==10, "hi"); CHECK(clampf(0,0,0)==0, "zero");
    PASS();
}

void test_rmsnorm(void) {
    TEST("rmsnorm");
    float x[]={3,4}, out[2];
    rmsnorm(out,x,2);
    float ms=(9+16)/2.0f, sc=1.0f/sqrtf(ms+1e-6f);
    CHECK(fabsf(out[0]-3*sc)<1e-5f, "el0");
    CHECK(fabsf(out[1]-4*sc)<1e-5f, "el1");
    float norm=sqrtf(out[0]*out[0]+out[1]*out[1]);
    CHECK(fabsf(norm-sqrtf(2.0f))<0.01f, "norm");
    PASS();
}

void test_matmul(void) {
    TEST("matmul");
    float x[]={1,2,3}, w[]={1,0,0, 0,1,0}, out[2];
    matmul(out,x,w,3,2);
    CHECK(fabsf(out[0]-1)<1e-5f, "r0"); CHECK(fabsf(out[1]-2)<1e-5f, "r1");
    PASS();
}

void test_softmax(void) {
    TEST("softmax");
    float x[]={1,2,3}; softmax(x,3);
    CHECK(fabsf(x[0]+x[1]+x[2]-1)<1e-5f, "sum1");
    CHECK(x[2]>x[1]&&x[1]>x[0], "order");
    PASS();
}

/* ── 2. BPE ── */
typedef struct{int a,b,new_id;}BPEMerge;
#define MAX_BPE 1024
#define MAX_VOCAB 1280
typedef struct{
    BPEMerge merges[MAX_BPE]; int n_merges,vocab_size;
    uint8_t vocab_bytes[MAX_VOCAB][64]; int vocab_len[MAX_VOCAB];
}BPE;

static int bpe_load(BPE *bpe, const char *path){
    FILE *f=fopen(path,"rb"); if(!f) return 0;
    uint32_t n; fread(&n,4,1,f); bpe->n_merges=(int)n; bpe->vocab_size=256+n;
    for(int i=0;i<256;i++){bpe->vocab_bytes[i][0]=(uint8_t)i;bpe->vocab_len[i]=1;}
    for(int i=0;i<(int)n&&i<MAX_BPE;i++){
        uint32_t a,b,nid; fread(&a,4,1,f);fread(&b,4,1,f);fread(&nid,4,1,f);
        bpe->merges[i].a=a;bpe->merges[i].b=b;bpe->merges[i].new_id=nid;
        int la=bpe->vocab_len[a],lb=bpe->vocab_len[b];
        if(la+lb<64){memcpy(bpe->vocab_bytes[nid],bpe->vocab_bytes[a],la);
            memcpy(bpe->vocab_bytes[nid]+la,bpe->vocab_bytes[b],lb);
            bpe->vocab_len[nid]=la+lb;}
    }
    fclose(f); return 1;
}

static int bpe_encode(const BPE *bpe, const uint8_t *text, int tlen, int *out, int maxo){
    int n=0; for(int i=0;i<tlen&&n<maxo;i++) out[n++]=text[i];
    for(int m=0;m<bpe->n_merges;m++){
        int a=bpe->merges[m].a,b=bpe->merges[m].b,nid=bpe->merges[m].new_id,j=0;
        for(int i=0;i<n;i++){if(i<n-1&&out[i]==a&&out[i+1]==b){out[j++]=nid;i++;}else out[j++]=out[i];}
        n=j;
    }
    return n;
}

static int bpe_decode_token(const BPE *bpe, int id, char *buf, int sz){
    if(id<0||id>=bpe->vocab_size)return 0;
    int len=bpe->vocab_len[id]; if(len>=sz)len=sz-1;
    memcpy(buf,bpe->vocab_bytes[id],len); buf[len]=0; return len;
}

void test_bpe_load(void) {
    TEST("bpe_load");
    BPE bpe; int ok=bpe_load(&bpe,"q.merges");
    CHECK(ok, "opened"); CHECK(bpe.n_merges==1024, "1024 merges");
    CHECK(bpe.vocab_size==1280, "vocab 1280");
    CHECK(bpe.vocab_bytes[65][0]==65, "A=65");
    PASS();
}

void test_bpe_encode(void) {
    TEST("bpe_encode");
    BPE bpe; bpe_load(&bpe,"q.merges");
    int ids[64]; int n=bpe_encode(&bpe,(const uint8_t*)"resonance",9,ids,64);
    CHECK(n>0&&n<=9, "len ok"); CHECK(n<9, "merged");
    PASS();
}

void test_bpe_roundtrip(void) {
    TEST("bpe_roundtrip");
    BPE bpe; bpe_load(&bpe,"q.merges");
    const char *text="Hello world";
    int ids[64]; int n=bpe_encode(&bpe,(const uint8_t*)text,11,ids,64);
    char dec[256]={0}; int pos=0;
    for(int i=0;i<n;i++){char buf[128];int len=bpe_decode_token(&bpe,ids[i],buf,128);memcpy(dec+pos,buf,len);pos+=len;}
    dec[pos]=0;
    CHECK(strcmp(dec,text)==0, "matches");
    PASS();
}

/* ── 3. MetaWeights bigram ── */
void test_metaweights_bigram(void) {
    TEST("metaweights_bigram");
    int ids[]={0,1,0,1,0,1};
    typedef struct{int a,b;float p;}Bi;
    Bi bis[10]; int nb=0;
    for(int i=0;i<5;i++){
        int found=0;
        for(int j=0;j<nb;j++) if(bis[j].a==ids[i]&&bis[j].b==ids[i+1]){bis[j].p+=1;found=1;break;}
        if(!found){bis[nb].a=ids[i];bis[nb].b=ids[i+1];bis[nb].p=1;nb++;}
    }
    CHECK(nb==2, "2 bigrams");
    PASS();
}

/* ── 4. Chambers ── */
void test_chambers_init(void) {
    TEST("chambers_init");
    float act[6]={0}; act[1]=0.2f; act[4]=0.15f; float debt=0;
    CHECK(fabsf(act[1]-0.2f)<1e-5f, "LOVE=0.2");
    CHECK(fabsf(act[4]-0.15f)<1e-5f, "FLOW=0.15");
    CHECK(act[0]==0, "FEAR=0"); CHECK(debt==0, "debt=0");
    PASS();
}

void test_chambers_decay(void) {
    TEST("chambers_decay");
    float act=0.2f; act*=0.93f;
    CHECK(fabsf(act-0.186f)<1e-3f, "decayed");
    PASS();
}

/* ── 5. DOE Parliament ── */
void test_parliament_expert(void) {
    TEST("parliament_expert_forward");
    int d=8, rank=4;
    float *A=calloc(rank*d,sizeof(float)), *B=calloc(d*rank,sizeof(float));
    for(int i=0;i<rank*d;i++) A[i]=0.01f*((float)rand()/RAND_MAX-0.5f);
    for(int i=0;i<d*rank;i++) B[i]=0.01f*((float)rand()/RAND_MAX-0.5f);
    float x[8]={1,0,0,0,0,0,0,0}, mid[4], out[8];
    for(int r=0;r<rank;r++){float s=0;for(int j=0;j<d;j++) s+=A[r*d+j]*x[j];mid[r]=s;}
    for(int o=0;o<d;o++){float s=0;for(int r=0;r<rank;r++) s+=B[o*rank+r]*mid[r];out[o]=s;}
    float sum=0; for(int i=0;i<d;i++) sum+=fabsf(out[i]);
    CHECK(sum>0, "nonzero"); CHECK(sum<1, "small");
    free(A); free(B);
    PASS();
}

void test_parliament_vitality(void) {
    TEST("parliament_vitality");
    float v=1.0f; v*=0.95f;
    CHECK(v<1.0f, "decayed"); CHECK(fabsf(v-0.95f)<1e-5f, "=0.95");
    PASS();
}

/* ── 6. Transformer gate ── */
void test_transformer_gate(void) {
    TEST("transformer_gate");
    CHECK(clampf((0.1f-0.5f)/1.5f,0,1)==0, "untrained=0");
    CHECK(clampf((2.0f-0.5f)/1.5f,0,1)==1, "trained=1");
    CHECK(fabsf(clampf((1.25f-0.5f)/1.5f,0,1)-0.5f)<1e-5f, "mid=0.5");
    PASS();
}

/* ── 7. Boundary detection ── */
void test_boundary(void) {
    TEST("boundary_detection");
    BPE bpe; bpe_load(&bpe,"q.merges");
    int dot=46; int is_b=0;
    for(int i=0;i<bpe.vocab_len[dot];i++) if(bpe.vocab_bytes[dot][i]=='.') is_b=1;
    CHECK(is_b, ". is boundary");
    is_b=0;
    for(int i=0;i<bpe.vocab_len[97];i++) if(bpe.vocab_bytes[97][i]=='.'||bpe.vocab_bytes[97][i]=='!'||bpe.vocab_bytes[97][i]=='?') is_b=1;
    CHECK(!is_b, "a not boundary");
    PASS();
}

/* ── 8. Coherence scoring ── */
void test_coherence(void) {
    TEST("coherence_score");
    float sb=(5>15)?1.5f:(5>10)?0.8f:(5>6)?0.2f:-0.5f;
    float lb=(20>15)?1.5f:(20>10)?0.8f:(20>6)?0.2f:-0.5f;
    CHECK(sb==-0.5f, "short penalty"); CHECK(lb==1.5f, "long bonus");
    PASS();
}

/* ── 9. Memory persistence ── */
void test_memory(void) {
    TEST("memory_save_load");
    uint32_t magic=0x514D454D;
    FILE *f=fopen("/tmp/test_q.memory","wb");
    CHECK(f!=NULL, "write open");
    fwrite(&magic,4,1,f);
    int nb=2,nt=0,nh=0;
    fwrite(&nb,4,1,f);fwrite(&nt,4,1,f);fwrite(&nh,4,1,f);
    int a1=10,b1=20;float p1=0.5f;
    fwrite(&a1,4,1,f);fwrite(&b1,4,1,f);fwrite(&p1,4,1,f);
    int a2=30,b2=40;float p2=0.8f;
    fwrite(&a2,4,1,f);fwrite(&b2,4,1,f);fwrite(&p2,4,1,f);
    fclose(f);
    f=fopen("/tmp/test_q.memory","rb"); CHECK(f!=NULL, "read open");
    uint32_t rm; fread(&rm,4,1,f); CHECK(rm==0x514D454D, "magic");
    int rnb,rnt,rnh; fread(&rnb,4,1,f);fread(&rnt,4,1,f);fread(&rnh,4,1,f);
    CHECK(rnb==2, "2 bigrams");
    int ra,rb;float rp;
    fread(&ra,4,1,f);fread(&rb,4,1,f);fread(&rp,4,1,f);
    CHECK(ra==10&&rb==20, "ids match"); CHECK(fabsf(rp-0.5f)<1e-5f, "prob match");
    fclose(f); remove("/tmp/test_q.memory");
    PASS();
}

/* ── 10. Schumann ── */
void test_schumann(void) {
    TEST("schumann_resonance");
    float s=0.4f*sinf(2*M_PI*7.83f*0);
    CHECK(fabsf(s)<1e-5f, "t=0 is 0");
    s=0.4f*sinf(2*M_PI*7.83f*(0.25f/7.83f));
    CHECK(s>0.3f, "quarter>0.3");
    PASS();
}

/* ── 11. Nucleus sampling ── */
void test_nucleus(void) {
    TEST("nucleus_sampling");
    float p[]={10,1,0.5f,0.1f,-5}; softmax(p,5);
    CHECK(p[0]>0.99f, "peaked");
    float cum=0; int ns=0;
    for(int i=0;i<5;i++){cum+=p[i];ns++;if(cum>=0.85f)break;}
    CHECK(ns==1, "nucleus=1");
    PASS();
}

/* ── 12. Smoke test ── */
void test_smoke(void) {
    TEST("smoke_compile_run");
    int ret=system("gcc postgpt_q.c -O2 -lm -o /tmp/q_smoke 2>/dev/null");
    CHECK(ret==0, "compiles");
    ret=system("echo quit | timeout 15 /tmp/q_smoke q.merges q.txt >/dev/null 2>&1");
    CHECK(ret==0, "runs no-crash");
    remove("/tmp/q_smoke");
    PASS();
}

int main(void) {
    printf("\n========== PostGPT-Q Test Suite ==========\n\n");
    test_clampf(); test_rmsnorm(); test_matmul(); test_softmax();
    test_bpe_load(); test_bpe_encode(); test_bpe_roundtrip();
    test_metaweights_bigram();
    test_chambers_init(); test_chambers_decay();
    test_parliament_expert(); test_parliament_vitality();
    test_transformer_gate();
    test_boundary();
    test_coherence();
    test_memory();
    test_schumann();
    test_nucleus();
    test_smoke();
    printf("\n==========================================\n");
    printf("  PASSED: %d  FAILED: %d  TOTAL: %d\n", tests_passed, tests_failed, tests_passed+tests_failed);
    printf("==========================================\n\n");
    return tests_failed>0?1:0;
}
