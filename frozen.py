import taichi as ti
quality = 1 # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality ** 2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

x = ti.Vector(2, dt=ti.f32, shape=n_particles) # position
v = ti.Vector(2, dt=ti.f32, shape=n_particles) # velocity
C = ti.Matrix(2, 2, dt=ti.f32, shape=n_particles) # affine velocity field
F = ti.Matrix(2, 2, dt=ti.f32, shape=n_particles) # deformation gradient
material = ti.var(dt=ti.i32, shape=n_particles) # material id
Jp = ti.var(dt=ti.f32, shape=n_particles) # plastic deformation
grid_v = ti.Vector(2, dt=ti.f32, shape=(n_grid, n_grid)) # grid node momemtum/velocity
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid)) # grid node mass
ti.cfg.arch = ti.cuda # Try to run on GPU

