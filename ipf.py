def ipf(seed, targets, max_iter=1000, tol=1e-6):
    """
     Perform Iterative Proportional Fitting (IPF) on a seed matrix.
     
     Parameter:
     
     - seed: inital joint matrix.
     - targets: list of tuples (target_array, axes),
        * the target array is also called constraits, it sum the overall axes, the axes do not include the target variable axes.
     - max_iter: int, Default:1000, the maximum numebr of iterations.
     - tol:numerical, Default:1e-6, tolerance for convergence.
     
     Returns:
     
      The final joint matrix after IPF adjustment

     Dependencies:
        Numpy
    """
    # Dependecies
    import numpy as np


    # Create a copy of the seed matrix and convert it to float for updates.
    fitted = seed.copy().astype(float)
    
    # Loop for a maximum number of iterations.
    for i in range(max_iter):
        # Copy the current fitted matrix to check convergence later.
        prev = fitted.copy()
        
        # Loop through each target constraint and its corresponding axes.
        for (target, axes) in targets:
            # Calculate the current marginal sum along the specified axes.
            current_marginal = fitted.sum(axis=axes)
            
            # Compute the ratio between target and current marginal.
            with np.errstate(divide='ignore', invalid='ignore'):
                # For non-zero marginal sums, compute the ratio; otherwise, use 0.
                ratio = np.where(current_marginal != 0, target / current_marginal, 0)
                # Expand dimensions of ratio to align with the fitted array.
                ratio = np.expand_dims(ratio, axis=axes)
                # Adjust fitted values by multiplying with the ratio.
                fitted *= ratio
        
        # Check for convergence: if the maximum change is less than tolerance, stop iterations.
        if np.max(np.abs(fitted - prev)) < tol:
            break
    
    return fitted 
