#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int n_samples, m_features, k_threads;

void print_args(){
    printf("\nCreating a dataset X(n x m) made of n=%d samples and m=%d features, using k=%d threads.", n_samples, m_features, k_threads);
}

void generate_true_beta(double* true_beta) {
    srand(10);
    for (int i = 0; i < m_features; i++) 
        true_beta[i] = ((double)rand() / RAND_MAX) * 4.0 - 2.0;
}

void print_beta(double* true_beta){
    int elems_to_show = (m_features < 5) ? m_features : 5;
    printf("\nFirst %d elements of true_beta: \n[  ", elems_to_show);
    for (int i = 0; i < elems_to_show; i++) 
        printf("%f  ", true_beta[i]);
    if(elems_to_show < m_features)
        printf("...  ");
    printf("]\n");
}

double** generate_matrix_X() {
    double **X = malloc(n_samples*sizeof(double*)); // Allocate rows
    for(int i = 0; i < n_samples; i++) // For each row, allocate columns 
        X[i] = malloc(m_features*sizeof(double));   

    for (int i = 0; i < n_samples; i++) { // Fill with random values (from 0 to 10)
        for (int j = 0; j < m_features; j++) 
            X[i][j] = ((double)rand() / RAND_MAX) * 10.0;
    }
    return X;
}

void print_matrix_X(double **X) {    
    int rows_to_show = (n_samples < 5) ? n_samples : 5;
    int cols_to_show = (m_features < 5) ? m_features : 5;
    
    printf("\nMatrix X (first %d rows, first %d columns):\n", rows_to_show, cols_to_show);
    
    for (int i = 0; i < rows_to_show; i++) {
        printf("Row %d: ", i);
        for (int j = 0; j < cols_to_show; j++) 
            printf("%.3f ", X[i][j]);
        
        if (m_features > 5) 
            printf("...");
        printf("\n");
    }
    if (n_samples > 5) 
        printf("... (%d more rows)\n", n_samples - 5);
}

double* generate_vector_y(double **X, double *true_beta){
    double *y = malloc(n_samples*sizeof(double));
    for(int i = 0; i < n_samples; i++){
        y[i] = 0.0;

        // Dot product X * true_beta
        for(int j = 0; j < m_features; j++)
            y[i] += X[i][j]*true_beta[j];
        
        // This random noise gives simulated "instability" to the vector y. 
        // It's useful to compute the error in a second moment.
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        y[i] += noise;
    }
    return y;
}

void print_vector_y(double *y) {
    printf("\nVector y (first 10 values):\n");
    
    int show = (n_samples < 10) ? n_samples : 10;
    for (int i = 0; i < show; i++) 
        printf("    y[%d] = %.6f\n", i, y[i]);
    
    if (n_samples > 10) 
        printf("    ... (%d more values)\n", n_samples - 10);
}

double** compute_XTX(double** X){
    double** XTX = malloc(m_features * sizeof(double*));
    for(int i = 0; i < m_features; i++) 
        XTX[i] = malloc(m_features * sizeof(double));

    
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i = 0; i < m_features; i++) {
        for(int j = 0; j < m_features; j++) {
            double sum = 0.0; // *
            for(int k = 0; k < n_samples; k++)
                sum += X[k][i] * X[k][j];
            XTX[i][j] = sum;
        }
    }
    return XTX;
}

// *sum is added because XTX is shared. XTX[i][j] += ... would be a race condition 

double* compute_XTy(double** X, double* y){
    double* XTy = malloc(m_features * sizeof(double));

    #pragma omp parallel for
    for(int i = 0; i < m_features; i++){
        double sum = 0.0; // *
        for(int k = 0; k < n_samples; k++)
            sum += X[k][i] * y[k];
        XTy[i] = sum;
    }
    return XTy;
}

// Transforms the system in superior triangular
void forward_elimination(double** A, double* b) {
    // For every column (except the last one)
    for (int i = 0; i < m_features - 1; i++) { 
        
        // PARTIAL PIVOTING: find the row with maximum pivot
        int max_row = i;
        for (int k = i + 1; k < m_features; k++) {
            if (fabs(A[k][i]) > fabs(A[max_row][i])) 
                max_row = k;
        }
        
        if (max_row != i) {
            // Swap rows in A
            double* temp_row = A[i];
            A[i] = A[max_row];
            A[max_row] = temp_row;
            
            // Swap rows in B
            double temp_b = b[i];
            b[i] = b[max_row];
            b[max_row] = temp_b;
        }
        
        if (fabs(A[i][i]) < 1e-10) {
            printf("Error: Matrix is singular or nearly singular at row %d!\n", i);
            return;
        }
        
        // Elimination
        for (int k = i + 1; k < m_features; k++) {
            double factor = A[k][i] / A[i][i];
            
            // row_k -= factor * row_i
            for (int j = i; j < m_features; j++) {
                A[k][j] -= factor * A[i][j];
            }
            
            // Same for vector b
            b[k] -= factor * b[i];
        }
    }
}

double* back_substitution(double** A, double* b) {
    double* beta = malloc(m_features * sizeof(double));
    
    // Solve the last equation
    if (fabs(A[m_features-1][m_features-1]) < 1e-10) {
        printf("Error: Division by zero in back substitution!\n");
        free(beta);
        return NULL;
    }
    beta[m_features - 1] = b[m_features - 1] / A[m_features - 1][m_features - 1];
    
    // Go through the top
    for (int i = m_features - 2; i >= 0; i--) {
        double sum = 0.0;
        
        // Subtract the contributes of already-solved variables
        for (int j = i + 1; j < m_features; j++) {
            sum += A[i][j] * beta[j];
        }
        
        // Solve for beta[i]
        if (fabs(A[i][i]) < 1e-10) {
            printf("Error: Division by zero in back substitution at row %d!\n", i);
            free(beta);
            return NULL;
        }
        beta[i] = (b[i] - sum) / A[i][i];
    }
    
    return beta;
}

double* gaussian_elimination(double** XTX, double* XTy) {
    // We first copy matrix X (A) and vector y (b) to perform
    // Gaussian Elimination. This is useful for simplifying the system.
    double** A = malloc(m_features * sizeof(double*));
    for (int i = 0; i < m_features; i++) {
        A[i] = malloc(m_features * sizeof(double));
        for (int j = 0; j < m_features; j++) 
            A[i][j] = XTX[i][j];
    }

    double* b = malloc(m_features * sizeof(double));
    for (int i = 0; i < m_features; i++) 
        b[i] = XTy[i];
    
    forward_elimination(A, b);
    double* beta = back_substitution(A, b);
    
    // Free temporary memory
    for (int i = 0; i < m_features; i++) 
        free(A[i]);
    free(A);
    free(b);

    return beta;
}

void validate_results(double* true_beta, double* computed_beta) {
    printf("\n=== NUMERICAL STABILITY ANALYSIS ===\n");
    
    double max_error = 0.0;
    double total_error = 0.0;
    double relative_error_sum = 0.0;
    
    printf("\nCoefficient Comparison (first 10):\n");
    printf("%-6s %-12s %-12s %-12s %-12s\n", 
           "Index", "True", "Computed", "Abs Error", "Rel Error");
    printf("---------------------------------------------------------------\n");
    
    int show = (m_features < 10) ? m_features : 10;
    for (int i = 0; i < show; i++) {
        double abs_error = fabs(true_beta[i] - computed_beta[i]);
        double rel_error = (fabs(true_beta[i]) > 1e-10) ? 
                           (abs_error / fabs(true_beta[i])) : 0.0;
        
        printf("%-6d %12.8f %12.8f %12.8f %12.6f%%\n", 
               i, true_beta[i], computed_beta[i], abs_error, rel_error * 100);
        
        if (abs_error > max_error) 
            max_error = abs_error;
        total_error += abs_error;
        relative_error_sum += rel_error;
    }
    
    // Compute errors for each coefficient
    for (int i = show; i < m_features; i++) {
        double abs_error = fabs(true_beta[i] - computed_beta[i]);
        double rel_error = (fabs(true_beta[i]) > 1e-10) ? 
                           (abs_error / fabs(true_beta[i])) : 0.0;
        
        if (abs_error > max_error) max_error = abs_error;
        total_error += abs_error;
        relative_error_sum += rel_error;
    }
    
    if (m_features > 10) 
        printf("... (%d more coefficients)\n", m_features - 10);
    
    double avg_abs_error = total_error / m_features;
    double avg_rel_error = relative_error_sum / m_features;
    
    printf("\n--- Error Statistics ---\n");
    printf("Maximum absolute error:  %.10e\n", max_error);
    printf("Average absolute error:  %.10e\n", avg_abs_error);
    printf("Average relative error:  %.6f%%\n", avg_rel_error * 100);
    
    // Compute the norm of the error
    double error_norm = 0.0;
    double true_norm = 0.0;
    for (int i = 0; i < m_features; i++) {
        double diff = true_beta[i] - computed_beta[i];
        error_norm += diff * diff;
        true_norm += true_beta[i] * true_beta[i];
    }
    error_norm = sqrt(error_norm);
    true_norm = sqrt(true_norm);
    double normalized_error = error_norm / true_norm;
    
    printf("L2 norm of error:        %.10e\n", error_norm);
    printf("Normalized error:        %.10e\n", normalized_error);
    
    printf("\n--- Quality Assessment ---\n");
    if (max_error < 1e-6)       printf("EXCELLENT: Very high precision (error < 1e-6)\n");
    else if (max_error < 1e-4)  printf("GOOD: High precision (error < 1e-4)\n");
    else if (max_error < 1e-2)  printf("ACCEPTABLE: Moderate precision (error < 1e-2)\n");
    else if (max_error < 1.0)   printf("WARNING: Low precision (error < 1.0)\n");
    else {
        printf("POOR: Very low precision (error >= 1.0)\n");
        printf("   Check for numerical instability or implementation errors.\n");
    }
    
    // Check per ill-conditioned matrix
    if (avg_rel_error > 0.01) {
        printf("Note: Relatively high errors may indicate:\n");
        printf("   - Ill-conditioned matrix (X^T X)\n");
        printf("   - Need for regularization\n");
        printf("   - Numerical precision limitations\n");
    }
}

void free_matrix_X(double **X) {
    for (int i = 0; i < n_samples; i++)
        free(X[i]);
    free(X);
}

int main(int argc, char* argv[]) {
    if(argc < 4 || argc > 4) {
        printf("\nYou must enter arguments: <n_samples> <m_features> <k_threads>!\n");
        return -1;
    }

    /// Step 1: get arguments from terminal
    n_samples = atoi(argv[1]);
    m_features = atoi(argv[2]);
    k_threads = atoi(argv[3]);
    omp_set_num_threads(k_threads);
    if(n_samples <= 0 || m_features <= 0 || k_threads <= 0){
        printf("n_samples, n_features and k_threads must be larger than 0!\n");
        return -1;
    }
    print_args();
    
    /// Step 2: generate true_beta[m] and X[n][m]
    double *true_beta = malloc(m_features*sizeof(double));
    generate_true_beta(true_beta);
    double **X = generate_matrix_X();
    
    print_beta(true_beta);
    print_matrix_X(X);
    
    /// Step 3: compute y = X * true_beta + noise 
    double* y = generate_vector_y(X, true_beta);
    print_vector_y(y);

    /// Step 4: compute XTX and XTy 
    // [=== THIS PART IS PARALLELIZED ===]
    double total_start = omp_get_wtime();

    printf("\nComputing X^T X and X^T y...\n");
    double xtx_start = omp_get_wtime();
    double** XTX = compute_XTX(X);
    double xtx_end = omp_get_wtime();
    double time_xtx = xtx_end - xtx_start;

    double xty_start = omp_get_wtime();
    double* XTy = compute_XTy(X, y);
    double xty_end = omp_get_wtime();
    double time_xty = xty_end - xty_start;

    // [=== END OF PARALLELIZATION ===]

    /// Step 5: extrapolate computed_beta -> Gaussian Elimination
    printf("Performing Gaussian Elimination...\n");
    double gauss_start = omp_get_wtime();
    double* computed_beta = gaussian_elimination(XTX, XTy);
    double gauss_end = omp_get_wtime();

    double time_gauss = gauss_end - gauss_start;
    double total_end = omp_get_wtime();
    double time_total = total_end - total_start;

    printf("\n=== PERFORMANCE RESULTS ===\n");
    printf("Dataset: n=%d samples, m=%d features\n", n_samples, m_features);
    printf("Time for X^T X:           %.6f seconds (%.2f%%)\n", 
           time_xtx, (time_xtx/time_total)*100);
    printf("Time for X^T y:           %.6f seconds (%.2f%%)\n", 
           time_xty, (time_xty/time_total)*100);
    printf("Time for Gaussian Elim:   %.6f seconds (%.2f%%)\n", 
           time_gauss, (time_gauss/time_total)*100);
    printf("----------------------------------------\n");
    printf("TOTAL TIME:               %.6f seconds\n", time_total);
    
    validate_results(true_beta, computed_beta);
    
    free(true_beta);
    free(computed_beta);
    free_matrix_X(X);
    free(y);

    for(int i = 0; i < m_features; i++)
        free(XTX[i]);
    
    free(XTX);
    free(XTy);
        
    return 0;
}