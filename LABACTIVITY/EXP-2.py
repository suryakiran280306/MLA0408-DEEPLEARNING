import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Actual parameters
actual_mean = 50
actual_std = 10
sample_size = 1000

# Generate synthetic data
data = np.random.normal(actual_mean, actual_std, sample_size)
# MLE Estimates
estimated_mean = np.mean(data)
estimated_variance = np.mean((data - estimated_mean) ** 2)
# Actual variance
actual_variance = actual_std ** 2
# Display results

print("Actual Mean:", actual_mean)
print("Estimated Mean (MLE):", round(estimated_mean, 2))
print("\nActual Variance:", actual_variance)
print("Estimated Variance (MLE):", round(estimated_variance, 2))
