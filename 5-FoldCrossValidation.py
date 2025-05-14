import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
x = np.random.uniform(0, 2*np.pi, 100) 
noise = np.random.normal(0, 0.1, 100)   
y = np.sin(x) + noise                   

def Coefficient_val(x, y, degree):
    
    # Create matrix X with all values : [x^d, x^{d-1}, ..., 1]
    X = np.column_stack([x**i for i in range(degree, -1, -1)])
    # Solve (X^T X)^{-1} X^T y
    Ans = np.linalg.inv(X.T @ X) @ X.T @ y
    return Ans

def True_Value_func(coeffs, x):
    #actual value of the fucntion 
    result = 0
    for coeff in coeffs:
        result = result * x + coeff
    return result

#perform 5 folds 
def cross_validate(x, y, degree):
    f_size = 20
    index = np.arange(len(x))
    np.random.shuffle(index)  
    errors = []
    
    for i in range(5):
        # Split into train/validation folds
        val_indices = index[i*f_size : (i+1)*f_size]
        train_indices = np.concatenate([index[:i*f_size], 
                                      index[(i+1)*f_size:]])
        
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]
        
        
        coeffs = Coefficient_val(x_train, y_train, degree)
        
        y_pred = np.array([True_Value_func(coeffs, xi) for xi in x_val])
        
        errors.append(np.mean((y_val - y_pred)**2))
    
    return np.mean(errors)


degrees = range(1, 5)
cv_errors = [cross_validate(x, y, d) for d in degrees]
print(cv_errors)
best_degree = degrees[np.argmin(cv_errors)]
print(f"Best degree: {best_degree} ")
for val in cv_errors:
    print(val)

print(f"Lowest CV Error:{ min(cv_errors):.4f}")    


plt.figure(figsize=(10, 6))
x_plot = np.linspace(0, 2*np.pi, 200)  
plt.plot(x_plot, np.sin(x_plot), label='sin(x)', linewidth=2)
plt.scatter(x, y, color='green', alpha=0.5, label='Noisy observations')


coeffs = Coefficient_val(x, y, best_degree)
y_pred = np.array([True_Value_func(coeffs, xi) for xi in x_plot])
plt.plot(x_plot, y_pred, '--', label=f'Degree {best_degree} polynomial', linewidth=2)

plt.legend()
plt.title("5-Fold CV")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()