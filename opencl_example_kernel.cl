// OpenCL Kernel for 3x3 grid multiplication
// Multiplies x and y position coordinates by stored constants

typedef struct {
  float multiplier_x;
  float multiplier_y;
} Constants;

__kernel void compute_grid(__constant Constants *constants,
                           __global float *results) {
  // Get the global work item ID (0-8 for 3x3 grid)
  int index = get_global_id(0);

  // Bounds check
  if (index >= 9) {
    return;
  }

  // Calculate x and y positions for 3x3 grid
  float x = (float)(index % 3);
  float y = (float)(index / 3);

  // Multiply positions by the constants
  float result_value =
      x * constants->multiplier_x + y * constants->multiplier_y;

  // Store the result
  results[index] = result_value;
}
