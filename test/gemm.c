#define N 200
#define M 300
#define K 400
#define DATA_TYPE float

void add(int a, int b, int *c) {
  *c = a + b;
}

void matmul(DATA_TYPE A[N][K], DATA_TYPE B[K][M], DATA_TYPE C[N][M]) {
  int i, j, k;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

int max(int x, int y) {
  return x > y ? x : y;
}
