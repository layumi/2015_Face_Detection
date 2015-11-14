#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float *data;
    float *outData;
    int *x1;
    int *x2;
    int *y1;
    int *y2;
    int *resize;
    int M,N,s1,s2,s3;
    int i,j,t,ss,tmp,tmp_x1,tmp_x2,tmp_y1,tmp_y2,count;
    data = mxGetPr(prhs[0]);
    M=mxGetM(prhs[0]); 
    N=mxGetN(prhs[0]); 
    x1=mxGetPr(prhs[1]);
    x2=mxGetPr(prhs[2]); 
    y1=mxGetPr(prhs[3]);
    y2=mxGetPr(prhs[4]); 
    s1 = mxGetScalar(prhs[5]);
    s2 = mxGetScalar(prhs[6]);
    s3 = mxGetScalar(prhs[7]);
    plhs[0]=mxCreateNumericMatrix(s1,s2*3*s3,mxSINGLE_CLASS,mxREAL); 
    outData=mxGetPr(plhs[0]);
    count=0; 
    for (ss=0;ss<s3;ss++){
      tmp_x1 = x1[ss];
      tmp_x2 = x2[ss];
      tmp_y1 = y1[ss];
      tmp_y2 = y2[ss];
      for (t=0;t<3;t++){
        tmp = t * M * N /3;
        for(i=tmp_y1;i<=tmp_y2;i++) {
            for(j=tmp_x1;j<=tmp_x2;j++) {
             outData[count++]=data[tmp+M*i+j]; 
            } 
        }
      }
    }
}
