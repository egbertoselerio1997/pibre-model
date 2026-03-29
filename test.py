import numpy as np
from scipy.optimize import nnls

if __name__ == "__main__":
    
    # 1. Biological Ground Truth (from your image)
    fraction_names = [
        "In_S_S", "In_S_I", "In_S_NH4_N", "In_S_NO3_N", 
        "In_S_PO4_P", "In_S_O2", "In_S_Alkalinity", "In_X_I", "In_X_S"
    ]
    x_true = np.array([
        219.3121, 50.7215, 49.7681, 2.0921, 
        2.9418, 0.4878, 217.0051, 98.6064, 103.0605
    ])
    
    composite_names = ["Total COD", "TKN", "Total P", "TSS"]
    
    # 2. Stoichiometric mapping matrix (4 Composites -> 9 Fractions)
    I = np.array([
        # S_S, S_I, NH4, NO3, PO4, O2, Alk, X_I, X_S
        [ 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 ],  # Total COD 
        [0.05, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.05 ],  # TKN 
        [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.01 ],  # Total P 
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.75,0.75 ]   # TSS 
    ])
    
    # 3. Calculate measured macroscopic composites (C_in will now have 4 elements)
    C_in = I @ x_true
    
    print("--- Simulated Measured Influent (C_in) ---")
    for name, val in zip(composite_names, C_in):
        print(f"{name:>10}: {val:.2f} mg/L")
        
    # 4. Solve using Standard NNLS (x >= 0)
    x_recovered, residual = nnls(I, C_in)
    
    # 5. Output comparison
    print(f"\n--- Pure NNLS Optimization (x >= 0) ---")
    print(f"Reconstruction Residual Error: {residual:.2e}\n")
    print(f"{'Fraction':<15} | {'x_true (Image)':<18} | {'x_recovered'}")
    print("-" * 55)
    for name, true_val, rec_val in zip(fraction_names, x_true, x_recovered):
        print(f"{name:<15} | {true_val:<18.4f} | {rec_val:.4f}")