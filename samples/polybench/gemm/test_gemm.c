#define N 8192
void test_gemm(float C[N][N], float A[N][N],
               float B[N][N]) {
#pragma scop
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
#pragma endscop
}
