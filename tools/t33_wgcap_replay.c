/* CPU-only replay of bounded real-Qwen T32 WGCAP v1 records. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D 2048
#define I 512
#define G 256
#define NG 8
#define RH 40
#define RB 1609800
#define NR 96
#define HB 64
#define M 3
#define NK 5
static const int ks[NK]={0,4,8,16,32};
static const char *mn[M]={"residual_magnitude","downnorm_proxy","effect_top64"};
typedef struct {double v;int i;} Score;
typedef struct {double v[NR],se,sr;int n;} Dist;
static int s4(const uint8_t*w,int row,int cols,int col){int v=(w[(size_t)row*(cols/2)+col/2]>>((col&1)*4))&15;return v&8?v-16:v;}
static float silu(float x){return x/(1.f+expf(-x));}
static float dsilu(float x){float s=1.f/(1.f+expf(-x));return s+x*s*(1.f-s);}
static int score_cmp(const void*a,const void*b){const Score*x=a,*y=b;return x->v<y->v?1:x->v>y->v?-1:x->i-y->i;}
static int double_cmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return x<y?-1:x>y;}
static void rank(const double*s,int n,int*out){Score*x=malloc((size_t)n*sizeof(*x));if(!x)exit(2);for(int i=0;i<n;i++)x[i]=(Score){s[i],i};qsort(x,n,sizeof(*x),score_cmp);for(int i=0;i<n;i++)out[i]=x[i].i;free(x);}
static void proj(float*out,const float*in,int rows,int cols,const uint8_t*w,const float*sc){
#pragma omp parallel for schedule(static)
 for(int r=0;r<rows;r++){float sum=0;for(int c=0;c<cols;c++)sum+=in[c]*(float)s4(w,r,cols,c);out[r]=sum*sc[r];}
}
static double rel(const float*a,const float*b,int n){double e=0,r=0;for(int i=0;i<n;i++){double d=(double)a[i]-b[i];e+=d*d;r+=(double)b[i]*b[i];}return sqrt(e/(r>1e-30?r:1e-30));}
static void add(Dist*d,double v,const float*a,const float*b,int n){d->v[d->n++]=v;for(int i=0;i<n;i++){double z=(double)a[i]-b[i];d->se+=z*z;d->sr+=(double)b[i]*b[i];}}
static double pct(const Dist*d,double p){double x[NR];memcpy(x,d->v,(size_t)d->n*sizeof(double));qsort(x,d->n,sizeof(double),double_cmp);int i=(int)ceil(p*d->n)-1;if(i<0)i=0;if(i>=d->n)i=d->n-1;return x[i];}
static void corrected(float*g,float*u,const float*bg,const float*bu,const int*ord,int k,const float*r,const uint8_t*gw,const uint8_t*uw,const float*gs,const float*us){
 memcpy(g,bg,I*4);memcpy(u,bu,I*4);for(int q=0;q<k;q++){int c=ord[q];for(int i=0;i<I;i++){g[i]+=r[c]*s4(gw,i,D,c)*gs[i];u[i]+=r[c]*s4(uw,i,D,c)*us[i];}}
}
int main(int argc,char**argv){
 if(argc!=2){fprintf(stderr,"usage: %s capture.wgcap\n",argv[0]);return 2;}FILE*f=fopen(argv[1],"rb");if(!f){perror(argv[1]);return 2;}
 unsigned char hdr[HB];if(fread(hdr,1,HB,f)!=HB||memcmp(hdr,"WGCAP01\0",8)||*(uint32_t*)(hdr+8)!=1||*(uint32_t*)(hdr+12)!=HB||*(uint32_t*)(hdr+20)!=D||*(uint32_t*)(hdr+24)!=I||*(uint32_t*)(hdr+32)!=RB||*(uint32_t*)(hdr+36)!=NR){fprintf(stderr,"invalid WGCAP header\n");return 2;}
 uint8_t*rec=malloc(RB);float *rg=malloc(I*4),*ru=malloc(I*4),*rh=malloc(I*4),*ry=malloc(D*4),*gq=malloc(I*4),*uq=malloc(I*4),*h=malloc(I*4),*y=malloc(D*4),*qi=malloc(D*4),*res=malloc(D*4),*dn=malloc(I*4);double *mag=malloc(D*sizeof(double)),*proxy=malloc(D*sizeof(double)),*effect=malloc(D*sizeof(double));int*orders=malloc((size_t)M*D*sizeof(int));
 if(!rec||!rg||!ru||!rh||!ry||!gq||!uq||!h||!y||!qi||!res||!dn||!mag||!proxy||!effect||!orders){fprintf(stderr,"allocation failure\n");return 2;}
 Dist raw[2]={0},hidden[2]={0},down[2]={0},dist[M][NK][2]={0};
 for(int ri=0;ri<NR;ri++){
  if(fread(rec,1,RB,f)!=RB){fprintf(stderr,"truncated record\n");return 2;}uint32_t*meta=(uint32_t*)rec;int layer=meta[0],step=meta[1],rank0=meta[4],split=step<2?0:1;if(!((layer==0||layer==20||layer==39)&&step<4&&rank0<8)){fprintf(stderr,"bad selector\n");return 2;}
  size_t o=RH;float*x=(float*)(rec+o);o+=D*4;int8_t*q=(int8_t*)(rec+o);o+=D;float*qs=(float*)(rec+o);o+=NG*4;float*cg=(float*)(rec+o);o+=I*4;float*cu=(float*)(rec+o);o+=I*4;float*ch=(float*)(rec+o);o+=I*4;float*cy=(float*)(rec+o);o+=D*4;uint8_t*gw=rec+o;o+=I*D/2;uint8_t*uw=rec+o;o+=I*D/2;uint8_t*dw=rec+o;o+=D*I/2;float*gs=(float*)(rec+o);o+=I*4;float*us=(float*)(rec+o);o+=I*4;float*ds=(float*)(rec+o);o+=D*4;if(o!=RB){fprintf(stderr,"layout error\n");return 2;}
  proj(rg,x,I,D,gw,gs);proj(ru,x,I,D,uw,us);add(&raw[split],rel(rg,cg,I),rg,cg,I);add(&raw[split],rel(ru,cu,I),ru,cu,I);for(int i=0;i<I;i++)rh[i]=silu(rg[i])*ru[i];add(&hidden[split],rel(rh,ch,I),rh,ch,I);proj(ry,ch,D,I,dw,ds);add(&down[split],rel(ry,cy,D),ry,cy,D);
  for(int c=0;c<D;c++){qi[c]=q[c]*qs[c/G];res[c]=x[c]-qi[c];mag[c]=fabsf(res[c]);}proj(gq,qi,I,D,gw,gs);proj(uq,qi,I,D,uw,us);
  for(int i=0;i<I;i++){double z=0;for(int out=0;out<D;out++){double w=s4(dw,out,I,i)*ds[out];z+=w*w;}dn[i]=sqrt(z);}
#pragma omp parallel for schedule(static)
  for(int c=0;c<D;c++){double z=0;for(int i=0;i<I;i++){float a=s4(gw,i,D,c)*gs[i],b=s4(uw,i,D,c)*us[i],j=cu[i]*dsilu(cg[i])*a+silu(cg[i])*b;z+=(double)dn[i]*dn[i]*j*j;}proxy[c]=fabsf(res[c])*sqrt(z);}
  rank(mag,D,orders);rank(proxy,D,orders+D);for(int c=0;c<D;c++)effect[c]=-1.;
  for(int v=0;v<64;v++){int c=orders[v];for(int i=0;i<I;i++){float a=gq[i]+res[c]*s4(gw,i,D,c)*gs[i],b=uq[i]+res[c]*s4(uw,i,D,c)*us[i];h[i]=silu(a)*b-silu(gq[i])*uq[i];}proj(y,h,D,I,dw,ds);double z=0;for(int out=0;out<D;out++)z+=(double)y[out]*y[out];effect[c]=sqrt(z);}rank(effect,D,orders+2*D);
  for(int m=0;m<M;m++)for(int ki=0;ki<NK;ki++){corrected(rg,ru,gq,uq,orders+m*D,ks[ki],res,gw,uw,gs,us);for(int i=0;i<I;i++)h[i]=silu(rg[i])*ru[i];proj(y,h,D,I,dw,ds);add(&dist[m][ki][split],rel(y,cy,D),y,cy,D);}
 }
 fclose(f);for(int s=0;s<2;s++){const char*n=s?"holdout":"calibration";printf("[T33-reconstruction] stage=gate_or_up split=%s aggregate_rel_l2=%.9g p99=%.9g max=%.9g records=%d\n",n,sqrt(raw[s].se/raw[s].sr),pct(&raw[s],.99),pct(&raw[s],1),raw[s].n);printf("[T33-reconstruction] stage=hidden split=%s aggregate_rel_l2=%.9g p99=%.9g max=%.9g records=%d\n",n,sqrt(hidden[s].se/hidden[s].sr),pct(&hidden[s],.99),pct(&hidden[s],1),hidden[s].n);printf("[T33-reconstruction] stage=down split=%s aggregate_rel_l2=%.9g p99=%.9g max=%.9g records=%d\n",n,sqrt(down[s].se/down[s].sr),pct(&down[s],.99),pct(&down[s],1),down[s].n);}
 for(int m=0;m<M;m++)for(int k=0;k<NK;k++)for(int s=0;s<2;s++){Dist*d=&dist[m][k][s];int bad=0;for(int i=0;i<d->n;i++)if(d->v[i]>.05)bad++;printf("[T33] method=%s k=%d split=%s aggregate_rel_l2=%.9g median=%.9g p95=%.9g p99=%.9g max=%.9g above_5pct=%d records=%d\n",mn[m],ks[k],s?"holdout":"calibration",sqrt(d->se/d->sr),pct(d,.5),pct(d,.95),pct(d,.99),pct(d,1),bad,d->n);}
 return 0;
}
