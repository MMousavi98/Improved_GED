# FEniCS code Variational Fracture Mechanics
################################################################################
#
# Modified GED
#
# (For plane-strain cases)
#
# Authors: Mohammad Mousavi
# Email: sm2652@cornell.edu
# date: 06/01/2024
#
################################################################################
from __future__ import division
from dolfin import *
from mshr import *
import os
import sympy
import numpy as np
import matplotlib.pyplot as plt

# Parameters for DOLFIN and SOLVER
# ----------------------------------------------------------------------------
set_log_level(LogLevel.WARNING)  # 20, // information of general interest

# set some dolfin specific parameters
parameters["form_compiler"]["representation"]="uflacs"
parameters["form_compiler"]["optimize"]=True
parameters["form_compiler"]["cpp_optimize"]=True
parameters["form_compiler"]["quadrature_degree"]=2
info(parameters,True)

# -----------------------------------------------------------------------------
# parameters of the solvers
solver_u_parameters   = {"nonlinear_solver": "snes",
                         "symmetric": True,
                         "snes_solver": {"linear_solver": "mumps", # lu mumps
                                         "method" : "newtontr",
                                         "line_search": "cp",
                                         "preconditioner" : "hypre_amg",
                                         "maximum_iterations": 300,
                                         "absolute_tolerance": 1e-6,
                                         "relative_tolerance": 1e-6,
                                         "solution_tolerance": 1e-6,
                                         "report": True,
                                         "error_on_nonconvergence": False}}


solver_Lmbda_parameters = {"nonlinear_solver": "snes",
                          "symmetric": True,
                          "snes_solver": {"maximum_iterations": 300,
                                          "report": True,
                                          "linear_solver": "mumps",
                                          "method": "vinewtonssls",  # if we want to use bounded solver, this should be "vinewtonssls" instead of "newtontr"
                                          "absolute_tolerance": 1e-6,
                                          "relative_tolerance": 1e-6,
                                          "error_on_nonconvergence": False}}


# Element-wise projection using LocalSolver
def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

# Define boundary sets for boundary conditions
# ----------------------------------------------------------------------------
class bot_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, 0.01*hsize) # near functuin takes three values: 1. given value 2. target value 3. tolerance

class top_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H, 0.01*hsize)

class pin_point(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L/2, hsize) and near(x[1], 0, 0.01*hsize)

# Convert all boundary classes for visualization
bot_boundary = bot_boundary()
top_boundary = top_boundary()
pin_point = pin_point()

# Damage function
lambda_i = 1.2          # damage threshold
Lmbda_ch_max = 2.7
alpha_damage = 1             # damage coeff: max damage
beta_damage = 20                # damage coeff: determine the sharpness of the damage evolution

# Material parameters
mu    = 1                       # Shear Modulus
kappa = 1000                    # Bulk Modulus
k_ell = 1e-5                    # Residual stiffness
eta = 0                         # Viscosity coeff

# Forces
body_force = Constant((0, 0))
load_min = 0
load_max = 0.6
load_steps = 100

# Geometry paramaters
L, H = 1, 1                    # Length (x) and height (y-direction)
resolution = 40
segment = 20
hsize = 0.02
ell_multi = 2
ell = Constant(ell_multi*hsize) # Length parameter

# Numerical parameters of the alternate minimization
maxiteration = 100
AM_tolerance = 2e-3

# Radius of the outer (2) and inner (1) circles for calculating J integral
r2 = 0.47
r1 = 0.45
cirle_center = 0.5 # for calculating J integral

crack_thickness = 0.04
crack_tip = 0.2

g_power = 0.28

# Naming parameters for saving output
simulation_params = "eta_%0.1f_ell_%0.2f_Lmbda_ch_max_%0.2f_g_%0.2f_lb_1.5" % (eta, ell, Lmbda_ch_max, g_power)
savedir   = simulation_params + "/"

# Define the geometric domains
domain = Rectangle(Point(0.0, 0.0), Point(L, H))
domain.set_subdomain(1, Circle(Point(cirle_center, 0.5*H), r2) - Circle(Point(cirle_center, 0.5*H), r1))

domain.set_subdomain(2, Rectangle(Point(0, 0.47*H), Point(L, 0.475*H)))
domain.set_subdomain(3, Rectangle(Point(0, 0.475*H), Point(L, 0.48*H)))
domain.set_subdomain(4, Rectangle(Point(0, 0.48*H), Point(L, 0.485*H)))
domain.set_subdomain(5, Rectangle(Point(0, 0.485*H), Point(L, 0.49*H)))
domain.set_subdomain(6, Rectangle(Point(0, 0.49*H), Point(L, 0.495*H)))
domain.set_subdomain(7, Rectangle(Point(0, 0.495*H), Point(L, 0.5*H)))
domain.set_subdomain(8, Rectangle(Point(0, 0.5*H), Point(L, 0.505*H)))
domain.set_subdomain(9, Rectangle(Point(0, 0.505*H), Point(L, 0.51*H)))
domain.set_subdomain(10, Rectangle(Point(0, 0.51*H), Point(L, 0.515*H)))
domain.set_subdomain(11, Rectangle(Point(0, 0.515*H), Point(L, 0.52*H)))
domain.set_subdomain(12, Rectangle(Point(0, 0.52*H), Point(L, 0.525*H)))
domain.set_subdomain(13, Rectangle(Point(0, 0.525*H), Point(L, 0.53*H)))

mesh = generate_mesh(domain, resolution)
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, mesh.domains())
top_boundary.mark(boundary_markers, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
mf = MeshFunction("size_t", mesh, 2, mesh.domains())
dx = Measure("dx", domain=mesh, subdomain_data=mf)

# Plot and save mesh
if not os.path.exists(savedir):
    os.makedirs(savedir)
plt.figure()
plot(mesh)
plt.savefig(savedir + "mesh.pdf")
plt.close()

#-----------------------------------------------------------------------------
# Define lines and points
lines = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
points = MeshFunction("size_t", mesh, mesh.topology().dim() - 2)

# show lines of interest
lines.set_all(0)
bot_boundary.mark(lines, 1)
top_boundary.mark(lines, 1)
file_results = XDMFFile(savedir + "/" + "lines.xdmf")
file_results.write(lines)

# Show points of interest
points.set_all(0)
pin_point.mark(points, 1)
file_results = XDMFFile(savedir + "/" + "points.xdmf")
file_results.write(points)

# Variational formulation
# ----------------------------------------------------------------------------
# Tensor space for projection of stress
T_DG0 = TensorFunctionSpace(mesh,'DG',0)
DG0   = FunctionSpace(mesh,'DG',0)
# Create mixed function space for elasticity
V_CG2 = VectorFunctionSpace(mesh, "Lagrange", 2)
# CG1 also defines the function space for damage
CG1 = FunctionSpace(mesh, "Lagrange", 1)
V_CG2elem = V_CG2.ufl_element()
CG1elem = CG1.ufl_element()
# Stabilized mixed FEM for incompressible elasticity
MixElem = MixedElement([V_CG2elem, CG1elem])
# Define function spaces for displacement and pressure
V = FunctionSpace(mesh, MixElem)

# Define the function, test and trial fields
w_p = Function(V)
u_p = TrialFunction(V)
v_q = TestFunction(V)
(u, p) = split(w_p)     # Displacement, pressure
(v, q) = split(v_q)   # Test functions for u, p
# Define the function, test and trial fields for damage problem
Lmbda  = Function(CG1)
Lmbda_trial = TrialFunction(CG1)
beta   = TestFunction(CG1)
damage = Function(CG1)
damage_previous = Function(CG1)

# Define functions to save
PTensor = Function(T_DG0, name="Nominal Stress")
FTensor = Function(T_DG0, name="Deformation Gradient")
JScalar = Function(CG1, name="Volume Ratio")

#--------------------------------------------------------------------
q_func_space = FunctionSpace(mesh, 'CG', 1)
q_expr = Expression('sqrt(pow(x[0]-cirle_center, 2) + pow(x[1]-0.5*H, 2)) < r1 ? \
                     1.0 : (sqrt(pow(x[0]-cirle_center, 2) + pow(x[1]-0.5*H, 2)) > r2 ? 0.0 : \
                     (r2 - sqrt(pow(x[0]-cirle_center, 2) + pow(x[1]-0.5*H, 2))) / (r2 - r1))', \
                          degree=1, r1=r1, r2=r2, cirle_center=cirle_center, H=H)
q_J_integral = interpolate(q_expr, q_func_space)
q_output = File(savedir+"q_J_integral.pvd")
q_output << q_J_integral

# Dirichlet boundary condition
# --------------------------------------------------------------------
u1 = Expression([0, "(-t/L)*x[0] + t"], t=0.0, L=L, degree=1)
u2 = Expression([0, "(t/L)*x[0] - t"], t=0.0, L=L, degree=1)
bc_u1 = DirichletBC(V.sub(0), u1, top_boundary)
bc_u2 = DirichletBC(V.sub(0), u2, bot_boundary)
bc_u = [bc_u1, bc_u2]

# bc - alpha (zero damage)
bc_Lmbda_B = DirichletBC(CG1, 1, bot_boundary)
bc_Lmbda_T = DirichletBC(CG1, 1, top_boundary)
bc_Lmbda = [bc_Lmbda_B, bc_Lmbda_T]

# Define the energy functional of damage problem
# --------------------------------------------------------------------
# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
J  = det(F)
Ic = tr(C) + 1
Lmbda_ch = sqrt(Ic/3)
Lmbda_ch = conditional(sqrt(Ic/3) > Lmbda_ch_max, Lmbda_ch_max, sqrt(Ic/3))

# Define the energy functional of the elasticity problem
# ----------------------------------------------------------------------------
# Constitutive functions of the damage model
def w(damage):           # Specific energy dissipation per unit volume
    return damage

def a(damage):           # Modulation function
    return (1.0-damage)**2

def b_sq(damage):        # b(alpha) = (1-alpha)^6 therefore we define b squared
    return (1.0-damage)**3

def P(u, damage):        # Nominal stress tensor
    return a(damage)*mu*(F - inv(F.T)) - b_sq(damage)*p*J*inv(F.T)

def energy_density_function(u, damage):
    return a(damage)*(mu/2.0)*(Ic-3.0-2.0*ln(J)) - b_sq(damage)*p*(J-1.0) - (1./(2.*kappa))*(p**2)

# Elastic energy, additional terms enforce material incompressibility and regularizes the Lagrange Multiplier:
elastic_energy    = ((1-k_ell)*a(damage)+k_ell)*(mu/2.0)*(Ic-3.0-2.0*ln(J))*dx \
                    - b_sq(damage)*p*(J-1.0)*dx - 1./(2.*kappa)*p**2*dx 

external_work     = dot(body_force, u)*dx
elastic_potential = elastic_energy - external_work

# Compute directional derivative about w_p in the direction of v (Gradient)
F_u = derivative(elastic_potential, w_p, v_q)
J_u = derivative(F_u, w_p, u_p)
# Variational problem for the displacement
problem_u = NonlinearVariationalProblem(F_u, w_p, bc_u, J=J_u)
solver_u  = NonlinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)

# Define the energy functional of damage problem
# --------------------------------------------------------------------
Lmbda_0 = interpolate(Expression("1", degree=0), CG1)  # initial (known) Lambda
Lmbda_previous = interpolate(Expression("1", degree=0), CG1)

def Relaxation(damage):
    return (1-damage)**g_power

def Viscous_Relaxation(Lmbda, Lmbda_previous):
    return conditional(Lmbda > Lmbda_previous, 1, 0)

dt = 1

Lmbda_WF = dt*(Lmbda*beta)*dx + Relaxation(damage)*dt*(ell**2)*inner(grad(Lmbda), grad(beta))*dx \
        - dt*Lmbda_ch*beta*dx + Viscous_Relaxation(Lmbda, Lmbda_previous)*eta*(Lmbda - Lmbda_previous)*beta*dx

Lmbda_Jacobian = derivative(Lmbda_WF, Lmbda, Lmbda_trial)

# Lower and upper bound
Lmbda_lb = interpolate(Expression("x[0]>=0 & x[0]<=L/5 & near(x[1], H/2, 0.01*hsize) ? 1.5 : 1", \
                       hsize = hsize, L=L, H=H, degree=0), CG1)
Lmbda_ub = interpolate(Expression("100", degree=0), CG1)

# Set up the solvers
problem_Lmbda = NonlinearVariationalProblem(Lmbda_WF, Lmbda, bc_Lmbda, J=Lmbda_Jacobian)
problem_Lmbda.set_bounds(Lmbda_lb, Lmbda_ub)
solver_Lmbda = NonlinearVariationalSolver(problem_Lmbda)
solver_Lmbda.parameters.update(solver_Lmbda_parameters)


load_multipliers = np.linspace(load_min, load_max, load_steps)
print("load:", load_multipliers)

# initialization of vectors to store data of interest
energies   = np.zeros((len(load_multipliers), 5))
iterations = np.zeros((len(load_multipliers), 2))

# Split solutions
(u, p) = w_p.split()
# Data file name
file_tot = XDMFFile(MPI.comm_world, savedir + "/results.xdmf")
# Saves the file in case of interruption
file_tot.parameters["rewrite_function_mesh"] = False
file_tot.parameters["functions_share_mesh"]  = True
file_tot.parameters["flush_output"]          = True

J_integral_list = []
crack_length = []
traction_x_list = []
traction_y_list = []
# ----------------------------------------------------------------------------
for (i_t, t) in enumerate(load_multipliers):
    # Structure used for one printout of the statement
    if MPI.rank(MPI.comm_world) == 0:
        print("\033[1;32m--- Starting of Time step {0:2d}: t = {1:4f} ---\033[1;m".format(i_t, t))

    # Alternate Mininimization Scheme
    # -------------------------------------------------------------------------
    # Solve for u holding alpha constant then solve for alpha holding u constant
    iteration = 1           # Initialization of iteration loop
    err_Lmbda = 1

    # Conditions for iteration
    while err_Lmbda > AM_tolerance and iteration < maxiteration:
        # solve elastic problem
        solver_u.solve()
        # solve damage problem with box constraint
        solver_Lmbda.solve()
        
        Lmbda_numeric = project(Lmbda, CG1)
        damage_values = np.zeros_like(Lmbda_numeric.vector().get_local())
        for i, value in enumerate(Lmbda_numeric.vector().get_local()):
            if value < lambda_i:
                # print(f"Index: {i}, Value: {value}")
                new_damage = 0
                damage_values[i] = max(new_damage, damage_previous.vector().get_local()[i])
            else:
                new_damage = 1 - ((lambda_i-1)/(value-1)) * (1 - alpha_damage + alpha_damage * np.exp(-beta_damage * (value - lambda_i)))
                damage_values[i] = max(new_damage, damage_previous.vector().get_local()[i])  # Ensure new damage doesn't go below previous damage
                # print(f"YYYYYYYYYYYYYYYYYYYYYYYYIndex: {i}, Value: {value}, and damage is {damage_values[i]}")
        damage.vector().set_local(damage_values)
                
        Lmbda_error = Lmbda.vector() - Lmbda_0.vector()
        err_Lmbda = Lmbda_error.norm('linf')
        print ("AM Iteration: {0:3d},  Lmbda_error: {1:>14.8f}".format(iteration, err_Lmbda))
        Lmbda_0.assign(Lmbda)


        volume_ratio = assemble(J/(L*H)*dx)
        iteration = iteration + 1


    # updating the lower bound to account for the irreversibility
    Lmbda_previous.assign(Lmbda)
    damage_previous.assign(damage)

    # Project
    local_project(P(u, damage), T_DG0, PTensor)
    local_project(F, T_DG0, FTensor)
    local_project(J, CG1, JScalar)

    # Rename for paraview
    u.rename("Displacement", "u")
    p.rename("Pressure", "p")
    Lmbda.rename("Stretch", "Lmbda")
    damage.rename("Damage", "damage")

    # Write solution to file
    Lmbda_ch_projected = project(Lmbda_ch, CG1)
    Lmbda_ch_projected.rename("Lambda_ch", "Lmbda_ch_projected")
    file_tot.write(u, t)
    file_tot.write(p, t)
    file_tot.write(Lmbda, t)
    file_tot.write(Lmbda_ch_projected, t)
    file_tot.write(damage, t)
    file_tot.write(PTensor, t)
    file_tot.write(FTensor, t)
    file_tot.write(JScalar,t)

    # Define the expression for the gradient term
    Vct_space = VectorFunctionSpace(mesh, "Lagrange", 1)
    laplacian_Lmbda = grad(Lmbda)
    grad_term = (ell**2) * laplacian_Lmbda
    grad_term_proj = project(grad_term, Vct_space)
    grad_term_proj.rename("Grad_Lambda_Term", "grad_term_proj")
    file_tot.write(grad_term_proj,t)

    # Update the displacement with each iteration
    u1.t = t
    u2.t = t

    # Post-processing
    # ----------------------------------------
    damage_values = damage.compute_vertex_values(mesh)
    # Find the rightmost node with damage >= 0.9
    rightmost_node = None
    rightmost_x = crack_tip
    for vertex in vertices(mesh):
        if damage_values[vertex.index()] >= 0.95:
            if vertex.point().x() > rightmost_x:
                rightmost_vertex = vertex
                rightmost_x = vertex.point().x()

    crack_length.append(rightmost_x)
    print("\ncrack_length:", crack_length)

    # J integral calculation
    F_1 = F[0, 0]
    F_2 = F[1, 0]
    F_1_vector = as_vector([F_1, F_2])
    J_expression = -1*(energy_density_function(u, damage)*grad(q_J_integral)[0] \
                        - inner(P(u, damage), outer(F_1_vector, grad(q_J_integral))))
    J_integral = assemble(J_expression*dx(1))
    J_integral_list.append(J_integral)
    print("\nJ Integral List:", J_integral_list)

    # Calculating the total force on the top boundary
    P_12 = P(u, damage)[0, 1]
    P_22 = P(u, damage)[1, 1]
    traction_y = assemble(P_22*ds(1))
    traction_y_list.append(traction_y)
    print("\ntraction y is:", traction_y_list)
    

# ----------------------------------------------------------------------------
crack_length_arr = np.array(crack_length)
J_integral_arr = np.array(J_integral_list)
traction_y_arr = np.array(traction_y_list)
np.savetxt(savedir + f'/J_disp.txt', np.column_stack((crack_length_arr, J_integral_arr)), \
            header='Crack Length | J Integral List', fmt='%f', delimiter=' | ')
np.savetxt(savedir + f'/traction_disp.txt', np.column_stack((load_multipliers, traction_y_arr)), \
            header='displacament | traction', fmt='%f', delimiter=' | ')


num_plot = load_steps
plt.figure(1)
plt.plot(crack_length[1:num_plot], J_integral_list[1:num_plot], label='total')
plt.xlabel('Crack length')
plt.ylabel('J')
plt.title('J Integral')
plt.legend()
plt.savefig(savedir + '/J_Integral.pdf', transparent=True)
plt.show()

plt.figure(2)
plt.plot(load_multipliers[1:num_plot], traction_y_list[1:num_plot])
plt.xlabel('Displacement')
plt.ylabel('Total force')
plt.title('Traction')
plt.savefig(savedir + '/traction_disp.pdf', transparent=True)






