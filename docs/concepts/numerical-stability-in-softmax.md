# Numerical Stability in Softmax

Softmax is a great example of why “correct math on paper” is not automatically “correct computation on hardware.”

The usual stable pattern is:

1. subtract the row maximum
2. exponentiate
3. sum the exponentials
4. divide

Subtracting the maximum prevents the exponentials from blowing up. This is one of the first places where you must think about correctness, numerics, and performance at the same time.
