import time
import numpy as np

def u_init(x):
    u_init = np.sin(3*np.pi*(x - 1/6))
    return u_init

def u_left(t):
    u_left = -1.
    return u_left

def u_right(t):
    u_right = 1.
    return u_right

start_time = time.perf_counter()

eps = 10**(-1.5)
a = 0.0; b = 1.0
t_0 = 0.0

T = 6.0
N = 800;    M = 300_000
# T = 0.02
# N = 8_000;  M = 100_000

h = (b - a) / N
tau = (T - t_0) / M

x = np.linspace(a, b, N+1)
t = np.linspace(t_0, T, M+1)

u = np.empty((M + 1, N + 1))

u[0, :] = u_init(x)

u[1:, 0] = u_left(t[1:])
u[1:, N] = u_right(t[1:])

eps_tau_h2 = eps*tau/h**2
tau_2h = tau/(2*h)

for m in range(M):
    u[m+1, 1:-1] = u[m,1:-1] + eps_tau_h2*(u[m,2:] - 2*u[m,1:-1] + u[m,:-2]) + \
        tau_2h*u[m,1:-1]*(u[m,2:] - u[m,:-2]) + tau*u[m,1:-1]**3

end_time = time.perf_counter()

print('Elapsed time is {:.4f} sec'.format(end_time-start_time))

# Если нужно сохранить данные в файл, откомментируйте команду ниже:
#np.savez('Example-08-0_Results', x=x, u=u)
