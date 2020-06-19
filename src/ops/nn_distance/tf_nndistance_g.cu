#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "float.h"
#define INNER_BATCH_SIZE 4
#define BATCH 512

__global__ void NmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
	__shared__ float buf[BATCH*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=BATCH){
			int end_k=min(m,k2+BATCH)-k2;
			for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*3+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				float x1=xyz[(i*n+j)*3+0];
				float y1=xyz[(i*n+j)*3+1];
				float z1=xyz[(i*n+j)*3+2];
				int best_i=0;
				float best=FLT_MAX;
				int end_ka=end_k-(end_k & (INNER_BATCH_SIZE - 1));
                for (int k=0;k<end_ka;k+=INNER_BATCH_SIZE){
                    #pragma unroll
                    for (int u=0; u < INNER_BATCH_SIZE; u++) {
                        float x2=buf[k*3+u*3]-x1;
                        float y2=buf[k*3+u*3+1]-y1;
                        float z2=buf[k*3+u*3+2]-z1;
                        float d=x2*x2+y2*y2+z2*z2;
                        if (d<best){
                            best=d;
                            best_i=k+k2+u;
                        }
                    }
                }
				for (int k=end_ka;k<end_k;k++){
					float x2=buf[k*3+0]-x1;
					float y2=buf[k*3+1]-y1;
					float z2=buf[k*3+2]-z1;
					float d=x2*x2+y2*y2+z2*z2;
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}
void NmDistanceKernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i){
	NmDistanceKernel<<<dim3(4,128,1),512>>>(b,n,xyz,m,xyz2,result,result_i);
	NmDistanceKernel<<<dim3(4,128,1),512>>>(b,m,xyz2,n,xyz,result2,result2_i);
}
__global__ void NmDistanceGradKernel(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			float x1=xyz1[(i*n+j)*3+0];
			float y1=xyz1[(i*n+j)*3+1];
			float z1=xyz1[(i*n+j)*3+2];
			int j2=idx1[i*n+j];
			float x2=xyz2[(i*m+j2)*3+0];
			float y2=xyz2[(i*m+j2)*3+1];
			float z2=xyz2[(i*m+j2)*3+2];
			float g=grad_dist1[i*n+j]*2;
			atomicAdd(&(grad_xyz1[(i*n+j)*3+0]),g*(x1-x2));
			atomicAdd(&(grad_xyz1[(i*n+j)*3+1]),g*(y1-y2));
			atomicAdd(&(grad_xyz1[(i*n+j)*3+2]),g*(z1-z2));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+0]),-(g*(x1-x2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+1]),-(g*(y1-y2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+2]),-(g*(z1-z2)));
		}
	}
}
void NmDistanceGradKernelLauncher(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2){
	cudaMemset(grad_xyz1,0,b*n*3*4);
	cudaMemset(grad_xyz2,0,b*m*3*4);
	NmDistanceGradKernel<<<dim3(1,64,1),512>>>(b,n,xyz1,m,xyz2,grad_dist1,idx1,grad_xyz1,grad_xyz2);
	NmDistanceGradKernel<<<dim3(1,64,1),512>>>(b,m,xyz2,n,xyz1,grad_dist2,idx2,grad_xyz2,grad_xyz1);
}

#endif
