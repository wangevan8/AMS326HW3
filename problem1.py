# Code modified from GeeksforGeeks and numpy.org reference
# Graph created with the assistance of Claude.ai

import numpy as np
import matplotlib.pyplot as plt

# Parameters
tosses = 4444444
w = 1  # Distance between parallel lines
diameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0]

max_lines = 6  # Allow up to 5+ crossings

# Results
results = {d: np.zeros(max_lines) for d in diameters}

# Simulation for each diameter
for d in diameters:
    radius = d / 2

    size = 100000
    num_batches = tosses // size
    remaining = tosses % size

    crossings_count = np.zeros(max_lines, dtype=int)

    for _ in range(num_batches):
        x_positions = np.random.random(size) * w

        for i in range(size):
            x = x_positions[i]

            left_cross = (x < radius)
            right_cross = ((w - x) < radius)

            crosses = 0
            if left_cross:
                crosses += 1
            if right_cross:
                crosses += 1

            if d > w:
                additional_left = int(np.floor((radius - x) / w))
                if additional_left > 0:
                    crosses += additional_left
                additional_right = int(np.floor((radius - (w - x)) / w))
                if additional_right > 0:
                    crosses += additional_right

            if crosses >= max_lines:
                crossings_count[-1] += 1
            else:
                crossings_count[crosses] += 1

    if remaining > 0:
        x_positions = np.random.random(remaining) * w
        for i in range(remaining):
            x = x_positions[i]

            left_cross = (x < radius)
            right_cross = ((w - x) < radius)

            crosses = 0
            if left_cross:
                crosses += 1
            if right_cross:
                crosses += 1

            if d > w:
                additional_left = int(np.floor((radius - x) / w))
                if additional_left > 0:
                    crosses += additional_left
                additional_right = int(np.floor((radius - (w - x)) / w))
                if additional_right > 0:
                    crosses += additional_right

            if crosses >= max_lines:
                crossings_count[-1] += 1
            else:
                crossings_count[crosses] += 1

    results[d] = crossings_count / tosses

# Print results
for d in diameters:
    print(f"Diameter d = {d:.2f}")
    for i in range(max_lines):
        if i == 0:
            print(f"  P(0 lines) = {results[d][0]:.6f}")
        elif i < max_lines - 1:
            print(f"  P({i} lines) = {results[d][i]:.6f}")
        else:
            print(f"  P({i}+ lines) = {results[d][i]:.6f}")
    print()

# Calculate P(at least 1 line) = 1 - P(0 lines)
at_least_one = {d: 1 - results[d][0] for d in diameters}

# Data for plotting
diameters_list = sorted(diameters)
prob_at_least_one = [at_least_one[d] for d in diameters_list]

# Plot probabilities as a function of diameter
plt.figure(figsize=(12, 8))

# Plot P(at least 1 line)
plt.plot(diameters_list, prob_at_least_one, marker='o', linewidth=2, label='P(at least 1 line)')

# Plot P(exactly k lines) for k=1,2,3,4
for k in range(1, 5):
    probs = [results[d][k] if k < max_lines - 1 else results[d][-1] for d in diameters_list]
    label = f'P(exactly {k} lines)' if k < 4 else f'P({k}+ lines)'
    plt.plot(diameters_list, probs, marker='s', linewidth=2, label=label)

plt.xlabel('Disc Diameter (d)')
plt.ylabel('Probability')
plt.title("Buffon's Disc: Probabilities of Crossing Lines")
plt.grid(True)
plt.legend()

# Theoretical values for small discs (d ≤ w)
theoretical_x = np.linspace(0, 1, 100)
theoretical_y = theoretical_x
plt.plot(theoretical_x, theoretical_y, 'r--', label='Theoretical (d/w)')

plt.legend()
plt.tight_layout()
plt.show()

