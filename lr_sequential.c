#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

int n_samples, m_features;

void print_args(){
    printf("\nStarting to create a dataset X(n x m) made of n=%d samples and m=%d features.", n_samples, m_features);
}

void fill_real_beta(double* true_beta) {
    srand(42);
    for (int i = 0; i < m_features; i++) 
        true_beta[i] = ((double)rand() / RAND_MAX) * 4.0 - 2.0;
}

void print_beta(double* true_beta){
    for (int i = 0; i < m_features; i++) 
        printf("%f ,", true_beta[i]);
    printf("\n");
}

double** generate_matrix_X() {
    printf("\nGenerating matrix X...\n");
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
    printf("\nMatrix X (first 5 rows, first 5 columns):\n");
    
    int rows_to_show = (n_samples < 5) ? n_samples : 5;
    int cols_to_show = (m_features < 5) ? m_features : 5;
    
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

        // Dot product X*beta
        for(int j = 0; j < m_features; j++)
            y[i] += X[i][j]*true_beta[j];
        
        // Add a random noise
        double noise = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        y[i] += noise;
    }
    return y;
}

void print_vector_y(double *y) {
    printf("\nVector y (first 10 values):\n");
    
    int show = (n_samples < 10) ? n_samples : 10;
    for (int i = 0; i < show; i++) 
        printf("y[%d] = %.6f\n", i, y[i]);
    
    if (n_samples > 10) 
        printf("... (%d more values)\n", n_samples - 10);
}

double** compute_XTX(double** X){
    double** XTX = malloc(m_features * sizeof(double*));
    for(int i = 0; i < m_features; i++) 
        XTX[i] = malloc(m_features * sizeof(double));

    for(int i = 0; i < m_features; i++){
        for(int j = 0; j < m_features; j++){
            XTX[i][j] = 0.0;  
            for(int k = 0; k < n_samples; k++)
                XTX[i][j] += X[k][i] * X[k][j];
            
        }
    }

    return XTX;
}

double* compute_XTy(double** X, double* y){
    double* XTy = malloc(m_features * sizeof(double));
    for(int i = 0; i < m_features; i++){
        XTy[i] = 0.0;
        for(int k = 0; k < n_samples; k++)
            XTy[i] += X[k][i] * y[k];
    }

    return XTy;
}

// Transforms the system in superior triangular
void forward_elimination(double** A, double* b) {
    for (int i = 0; i < m_features - 1; i++) { // For every column (except the last one)
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
    // Copy matrix A
    double** A = malloc(m_features * sizeof(double*));
    for (int i = 0; i < m_features; i++) {
        A[i] = malloc(m_features * sizeof(double));
        for (int j = 0; j < m_features; j++) 
            A[i][j] = XTX[i][j];
    }
    
    // Copy vector b
    double* b = malloc(m_features * sizeof(double));
    for (int i = 0; i < m_features; i++) 
        b[i] = XTy[i];
    
    
    printf("Performing forward elimination...\n");
    forward_elimination(A, b);
    
    printf("Performing back substitution...\n");
    double* beta = back_substitution(A, b);
    
    // Free temporary memory
    for (int i = 0; i < m_features; i++) {
        free(A[i]);
    }
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
        
        if (abs_error > max_error) max_error = abs_error;
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
    
    if (m_features > 10) {
        printf("... (%d more coefficients)\n", m_features - 10);
    }
    
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
    if (max_error < 1e-6) {
        printf("✓ EXCELLENT: Very high precision (error < 1e-6)\n");
    } else if (max_error < 1e-4) {
        printf("✓ GOOD: High precision (error < 1e-4)\n");
    } else if (max_error < 1e-2) {
        printf("⚠ ACCEPTABLE: Moderate precision (error < 1e-2)\n");
    } else if (max_error < 1.0) {
        printf("⚠ WARNING: Low precision (error < 1.0)\n");
    } else {
        printf("✗ POOR: Very low precision (error >= 1.0)\n");
        printf("   Check for numerical instability or implementation errors.\n");
    }
    
    // Check per ill-conditioned matrix
    if (avg_rel_error > 0.01) {
        printf("⚠ Note: Relatively high errors may indicate:\n");
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
    if(argc < 3 || argc > 3){
        printf("\nYou must enter arguments: <n_samples> <m_features>!\n");
        return -1;   
    }

    /*  FIRST PART: SETUP
        This part of code is simply the generation of our dataset.
        Each time it's randomly computed. The output of this dataset
        is y = X*true_beta + noise

        The linear regression should find the computed_beta, and it
        should be as closer as possible to true_beta, in order to
        have a successful regression.
    */

    /// Step 1: get arguments from terminal
    n_samples = atoi(argv[1]);
    m_features = atoi(argv[2]);
    print_args();

    /// Step 2: generate randomly the true_beta array[m] and the matrix X[n,m]
    double *true_beta = malloc(sizeof(double)*m_features);
    fill_real_beta(true_beta);
    //print_beta(true_beta);

    double **X = generate_matrix_X();
    print_matrix_X(X);

    /// Step 3: compute y = X * true_beta + noise -> noise is just a further random value  
    double* y = generate_vector_y(X, true_beta);
    print_vector_y(y);

    /*  SECOND PART: REGRESSION ALGORITHM
        This part is the processing of data. Our final objective is to
        go back to true_data given the samples, features and the values
        of y.
        - Input: X matrix, y vector
        - Actions:
            1) Compute X^T X (m x m matrix)
            2) Compute X^T y (m vector)
            3) Solve X^T X * computed_beta = X^T y
                for computed_beta
        - Output: computed_beta
    */

    printf("\n=== PERFORMANCE MEASUREMENT === \n");
    clock_t total_start = clock();

    /// Step 4: compute XTX and XTy
    printf("\nComputing X^T X...\n");
    clock_t xtx_start = clock();
    double** XTX = compute_XTX(X);
    clock_t xtx_end = clock();
    double time_xtx = ((double)(xtx_end - xtx_start)) / CLOCKS_PER_SEC;

    printf("Computing X^T y...\n");
    clock_t xty_start = clock();
    double* XTy = compute_XTy(X, y);
    clock_t xty_end = clock();
    double time_xty = ((double)(xty_end - xty_start)) / CLOCKS_PER_SEC;

    /// Step 5: extrapolate computed_beta -> Gaussian Elimination
    printf("\nSolving linear system...\n");
    clock_t gauss_start = clock();
    double* computed_beta = gaussian_elimination(XTX, XTy);
    clock_t gauss_end = clock();
    double time_gauss = ((double)(gauss_end - gauss_start)) / CLOCKS_PER_SEC;

    clock_t total_end = clock();
    double time_total = ((double)(total_end - total_start)) / CLOCKS_PER_SEC;

    /// Step 6: report performance
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
    
    /// Step 7: validate
    validate_results(true_beta, computed_beta);
    
    /// Final cleanup
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