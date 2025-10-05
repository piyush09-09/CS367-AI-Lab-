import numpy as np
import random
import math
from skimage import io, color
from skimage.transform import resize
import matplotlib.pyplot as plt

# -------------------------
# Load image
# -------------------------
image_path = "C:\\Users\\User\\OneDrive\\Desktop\\jigsaw.jpg"
img = io.imread(image_path)
if len(img.shape) == 3:
    img = color.rgb2gray(img)
img = resize(img, (512, 512))
PATCH_SIZE = 128
GRID_SIZE = 4
NUM_PATCHES = GRID_SIZE * GRID_SIZE

# -------------------------
# Split into patches
# -------------------------
def get_patches(image):
    patches = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            patch = image[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE]
            patches.append(patch)
    return patches

patches = get_patches(img)

# Scramble
scrambled_order = list(range(NUM_PATCHES))
random.shuffle(scrambled_order)

# Reconstruct image
def reconstruct_image(order, patches):
    image = np.zeros_like(img)
    for idx, patch_idx in enumerate(order):
        i = idx // GRID_SIZE
        j = idx % GRID_SIZE
        image[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE] = patches[patch_idx]
    return image

scrambled_img = reconstruct_image(scrambled_order, patches)

# -------------------------
# Edge-based scoring function
# -------------------------
def edge_score(order, patches):
    score = 0
    grid = [[None]*GRID_SIZE for _ in range(GRID_SIZE)]
    for idx, patch_idx in enumerate(order):
        row = idx // GRID_SIZE
        col = idx % GRID_SIZE
        grid[row][col] = patches[patch_idx]

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            patch = grid[i][j]
            # Compare with right neighbor
            if j < GRID_SIZE - 1:
                right_patch = grid[i][j+1]
                score += np.sum(np.abs(patch[:, -1] - right_patch[:, 0]))
            # Compare with bottom neighbor
            if i < GRID_SIZE - 1:
                bottom_patch = grid[i+1][j]
                score += np.sum(np.abs(patch[-1, :] - bottom_patch[0, :]))
    return score

# -------------------------
# Simulated Annealing
# -------------------------
def simulated_annealing(patches, initial_temp=500, alpha=0.995, iterations=10000):
    current_order = list(range(NUM_PATCHES))
    random.shuffle(current_order)
    current_score = edge_score(current_order, patches)
    
    best_order = current_order.copy()
    best_score = current_score
    T = initial_temp

    for it in range(iterations):
        i, j = random.sample(range(NUM_PATCHES), 2)
        new_order = current_order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]

        new_score = edge_score(new_order, patches)
        delta = new_score - current_score

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_order = new_order
            current_score = new_score
            if current_score < best_score:
                best_order = current_order.copy()
                best_score = current_score
        
        T *= alpha
    
    return best_order, best_score

# -------------------------
# Solve the puzzle
# -------------------------
best_order, best_score = simulated_annealing(patches)
solved_img = reconstruct_image(best_order, patches)

# -------------------------
# Show results
# -------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(scrambled_img, cmap='gray')
plt.title("Scrambled Image")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(solved_img, cmap='gray')
plt.title(f"Solved Image\nScore: {best_score:.2f}")
plt.axis('off')
plt.show()
