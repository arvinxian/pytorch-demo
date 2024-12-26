import torch
import numpy as np
from demo import BinaryConvolution

class MatrixSolver:
    def __init__(self):
        """Initialize solver with the specified 5x5 kernel"""
        self.kernel = torch.tensor([
            [0.0, 0.0, 0.05, 0.0, 0.0],
            [0.0, 0.05, 0.125, 0.05, 0.0],
            [0.05, 0.125, 0.311, 0.125, 0.05],
            [0.0, 0.05, 0.125, 0.05, 0.0],
            [0.0, 0.0, 0.05, 0.0, 0.0]
        ], dtype=torch.float32)
        
    def validate_dimensions(self, inner_matrix, n, m):
        """Validate matrix dimensions"""
        inner = np.array(inner_matrix)
        rows, cols = inner.shape
        
        expected_rows = m-4  # For a 6x7 matrix, we expect 2 rows
        expected_cols = n-4  # For a 6x7 matrix, we expect 3 cols
        
        if rows != expected_rows or cols != expected_cols:
            # If dimensions don't match but are transposed, transpose the matrix
            if rows == expected_cols and cols == expected_rows:
                inner = inner.T
            else:
                raise ValueError(
                    f"Inner matrix shape {inner.shape} doesn't match expected shape "
                    f"({expected_rows}, {expected_cols}) for full matrix size ({n}, {m})"
                )
        
        if n < 6 or m < 6:
            raise ValueError("Full matrix dimensions must be at least 6x6")
            
        return inner
        
    def create_full_target(self, inner_matrix, n, m):
        """Create full target matrix with zero padding"""
        # Validate and get numpy array
        inner = self.validate_dimensions(inner_matrix, n, m)
        
        # Create full zero matrix
        full = np.zeros((m, n))
        
        # Place inner matrix in the center
        full[2:m-2, 2:n-2] = inner
        
        return torch.tensor(full, dtype=torch.float32)

    def create_random_input(self, n, m):
        """Create random input matrix with zero borders"""
        # Create full zero matrix
        matrix = np.zeros((m, n))
        
        # Fill inner part with random binary values
        inner = np.random.randint(0, 2, size=(m-4, n-4))
        matrix[2:m-2, 2:n-2] = inner
        
        return torch.tensor(matrix, dtype=torch.float32)

    def calculate_score(self, result_matrix, target_matrix):
        """Calculate score using (N1 - 0.6*N2)/N0"""
        result = result_matrix.bool()
        target = target_matrix.bool()
        
        N1 = torch.sum((result & target).float())
        N2 = torch.sum(result.float()) - N1
        N0 = torch.sum(target.float())
        
        if N0 == 0:
            return 0.0
        
        score = (N1 - 0.6 * N2) / N0
        return score.item()

    def process_matrix(self, input_matrix):
        """Apply convolution and binarization"""
        model = BinaryConvolution(self.kernel)
        with torch.no_grad():
            output = model(input_matrix)
        return output

    def solve(self, target_inner, n, m, iterations=1000):
        """Find optimal input matrix"""
        try:
            # Create full target matrix
            target_matrix = self.create_full_target(target_inner, n, m)
            
            best_score = float('-inf')
            best_input = None
            best_result = None
            
            for i in range(iterations):
                # Generate random input
                input_matrix = self.create_random_input(n, m)
                
                # Process matrix
                result_matrix = self.process_matrix(input_matrix)
                
                # Calculate score
                score = self.calculate_score(result_matrix, target_matrix)
                
                if score > best_score:
                    best_score = score
                    best_input = input_matrix.clone()
                    best_result = result_matrix.clone()
                
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i+1}/{iterations}, Best score so far: {best_score:.4f}")
            
            return best_input, best_result, best_score
            
        except Exception as e:
            print(f"Error during solving: {str(e)}")
            raise

def print_matrix(name, matrix):
    """Helper function to print matrices nicely"""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
    print(f"\n{name}:")
    print(matrix.astype(int))

def main():
    try:
        # Pre-defined test case
        n, m = 7, 6  # Full matrix will be 6x7
        target_inner = np.array([
            [0, 0, 1],  # 2 rows (m-4 = 2)
            [1, 0, 1]   # 3 cols (n-4 = 3)
        ])  # 2x3 matrix representing the inner part
        
        print(f"Problem dimensions: {n}x{m}")
        print(f"Inner matrix dimensions: {target_inner.shape}")
        print("\nTarget Inner Matrix (before processing):")
        print(target_inner)
        
        # Create solver and find solution
        solver = MatrixSolver()
        best_input, best_result, best_score = solver.solve(target_inner, n, m)
        
        # Print results
        print("\n=== Results ===")
        print_matrix("Target Inner Matrix", target_inner)
        print(f"\nBest Score: {best_score:.4f}")
        print_matrix("Best Input Matrix", best_input)
        print_matrix("Result Matrix", best_result)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 