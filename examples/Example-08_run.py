#------------------------------------------------------------------
# Скрипт для запуска всех примеров решения одномерного (1D)
# дифференциального уравнения в частных производных (ДУЧП)
# параболического типа с использованием "явной" схемы
# метода конечных разностей.
#
# Конкретно, это примеры 8.0 (последовательный алгоритм),
# 8.1 (очень неэффективный параллельный алгоритм), и 8.2-8.3
# (наиболее эффективные параллельные алгоритмы).
#------------------------------------------------------------------

import subprocess


#------------------------------------------------------------------
def run_file(py_file, P, N, M, slow=False):
    cmd = f'py -3.13 {py_file} -N {N} -M {M} --noheader'
    if slow:
        cmd += ' --slow'
    if P > 1:
        cmd = f'mpiexec -n {P} ' + cmd

    try:
        result = subprocess.run(cmd.split(), # shell=True,
                                capture_output=True, text=True, check=True)

        # print("\nProgram output:")
        # print(result.stdout)

        res = result.stdout.split()

        if N != int(res[0]) or M != int(res[1]) or P != int(res[2]):
            raise subprocess.CalledProcessError(result.stdout)

        time_tot = float(res[3])
        time_sol = float(res[4])
        return time_tot, time_sol
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: '{cmd}'")
        print(f"Stdout: {e}")
        print(f"Stderr: {e.stderr}")
        exit()


#------------------------------------------------------------------
if __name__ == "__main__":
    example_sequential = 'Example-08-0.py'
    examples_parallel = ['Example-08-1.py', 'Example-08-2.py', 'Example-08-3.py']

    slow = False
    N = 200
    M = 10_000

    speed = 'slow' if slow else 'fast'
    save_file = f'Results_N{N}_M{M}_{speed}.dat'
    info_file = save_file.replace('.dat', '.info')

    msg = 'N\t M\t Procs\t time_tot\t time_sol\t R_tot\t R_sol\t E_tot\t E_sol\t File'
    print(msg)
    with open(save_file, 'a') as f:
        f.write(msg + '\n')

    # Расчёты без параллелизации:
    py_file = example_sequential
    time_tot0, time_sol0 = run_file(py_file, P=1, N=N, M=M, slow=slow)
    msg = f'{N}\t {M}\t 1\t {time_tot0:.6f}\t {time_sol0:.6f}\t 1\t 1\t 1\t 1\t {py_file}'
    print(msg)
    with open(save_file, 'a') as f:
        f.write(msg + '\n')

    # Расчёты с параллелизацией:
    for py_file in examples_parallel:
        for P in range(2, 9):
            time_tot, time_sol = run_file(py_file, P=P, N=N, M=M, slow=slow)
            R_tot = time_tot0 / time_tot
            R_sol = time_sol0 / time_sol
            E_tot = R_tot / P
            E_sol = R_sol / P
            msg = f'{N}\t {M}\t {P}\t {time_tot:.6f}\t {time_sol:.6f}' + \
                  f'\t {R_tot:.4f}\t {R_sol:.4f}\t {E_tot:.4f}\t {E_sol:.4f}\t {py_file}'
            print(msg)
            with open(save_file, 'a') as f:
                f.write(msg + '\n')

#------------------------------------------------------------------
