__kernel void iterate_gpu(__global const float *last,
                          __global const float *grid, __global float *next,
                          const int gridWidth, const float ratio,
                          __global uchar4 *colorGrid) {

  // ratio = v^2 * dt^2, v: velocity, dt: time step
  int id = get_global_id(0);
  // printf(id, " ");
  int x = id % gridWidth;
  int y = id / gridWidth;
  if (x != 0 && y != 0 && x != gridWidth - 1 && y != gridWidth - 1) {
    float sum_around = grid[id - 1] + grid[id + 1] + grid[id - gridWidth] +
                       grid[id + gridWidth];
    next[id] = -last[id] + (2.f - 4.f * ratio) * grid[id] + ratio * sum_around;
  }
  uchar4 color;
  float value = grid[id];
  value = fmax(fmin(value, 1.0f), -1.0f);
  float r = 0.5 * (fabs(value) + value);
  float b = 0.5 * (fabs(value) - value);
  float g = 0.5 * (r + b);

  color.x = (uchar)(r * 255.f);
  color.y = (uchar)(g * 255.f);
  color.z = (uchar)(b * 255.f);
  color.w = 255;
  colorGrid[id] = color;
}