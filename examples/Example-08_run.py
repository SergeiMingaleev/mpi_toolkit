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

import argparse
import subprocess


#------------------------------------------------------------------
def run_file(py_file, P):
    global args
    N, M = int(args.N), int(args.M)

    hosts = f'hosts_{args.hosts}.txt'

    if args.pwd is None:
        folder = ''
    else:
        folder = '\\\\10.183.0.240\\MPI_Share\\examples\\'

    cmd = f'{args.python} {folder}{py_file} -N {N} -M {M} --noheader'
    cmd_prefix = cmd_suffix = ''

    if args.slow:
        cmd_suffix += ' --slow'
    if P > 1:
        cmd_prefix += f'mpiexec.exe -n {P} '
        if args.pwd is not None:
            cmd_prefix += f'-machinefile {folder}{hosts}.txt ' \
                        + f'-pwd {args.pwd} -wdir {folder} '
    cmd = cmd_prefix + cmd + cmd_suffix

    try:
        result = subprocess.run(cmd.split(), # shell=True,
                                capture_output=True, text=True, check=True)

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
def parse_arguments():
    parser = argparse.ArgumentParser(
                prog='python Example-08_run.py',
                description='Решение 1D ДУЧП параболического типа с использованием '
                            '"явной" разностной схемы. Мы сравниваем эффективность '
                            'разных подходов к MPI параллелизации этой схемы.',
    )
    parser.add_argument('-N', default=800,
                        help='Число `N` интервалов сетки по координате `x`. '
                             'По умолчанию равно 800.')
    parser.add_argument('-M', default=100_000,
                        help='Число `M` интервалов сетки по времени `t`. '
                             'По умолчанию равно 100_000.')
    parser.add_argument('-T', default=2.0,
                        help='Максимальное время `T`, до которого должны проводиться '
                             'расчёты. По умолчанию равно 2.0.')
    parser.add_argument('--slow', action="store_true",
                        help='Использовать медленную реализацию решения, '
                             'с вызовом функции `consecutive_tridiagonal_matrix_algorithm()` '
                             'вместо `scipy.linalg.solve_banded()` для решения трёхдиагональной '
                             'системы линейных уравнений.')
    parser.add_argument('-python', default='python.exe',
                        help='Имя программы для запуска в качестве `python.exe`.')
    parser.add_argument('-pwd', default=None,
                        help='Пароль пользователя для MPI запуска на кластере.')
    parser.add_argument('-hosts', default='2cores',
                        help='На каких машинах и скольки процессорах нужно запускать '
                             'MPI параллелизацию? Возможные варианты: `all_logical`, '
                             '`all_cores`, `half_cores`, `4cores`, `3cores`, `2cores`, '
                             '`1cores`')

    args = parser.parse_args()
    return args

#------------------------------------------------------------------
if __name__ == "__main__":
    example_sequential = 'Example-08-0.py'
    examples_parallel = ['Example-08-3.py', 'Example-08-2.py', 'Example-08-1.py']
    #examples_parallel = ['Example-08-2.py', 'Example-08-1.py']
    #examples_parallel = ['Example-08-1.py']

    procs_for_hosts = {
        'all_logical': range(4, 97, 4),
        'all_cores': range(4, 49, 4),
        'half_cores': range(2, 25, 2),
        '4cores': range(4, 41, 4),
        '3cores': range(3, 31, 3),
        '2cores': range(2, 21, 2),
        '1cores': range(1, 11, 1)
    }

    args = parse_arguments()
    N, M = int(args.N), int(args.M)

    if args.pwd is None:
        procs = range(2, 9)
    else:
        procs = procs_for_hosts[args.hosts]

    speed = 'slow' if args.slow else 'fast'
    save_file = f'Results08_N{N}_M{M}_{speed}'
    if args.pwd is not None:
        save_file += f'_hosts_{args.hosts}'
    save_file += '.dat'

    msg = 'N\t M\t Procs\t time_tot\t time_sol\t R_tot\t R_sol\t E_tot\t E_sol\t File'
    print(msg)
    with open(save_file, 'a') as f:
        f.write(msg + '\n')

    # Расчёты без параллелизации:
    py_file = example_sequential
    time_tot0, time_sol0 = run_file(py_file, P=1)
    msg = f'{N}\t {M}\t 1\t {time_tot0:.6f}\t {time_sol0:.6f}\t 1\t 1\t 1\t 1\t {py_file}'
    print(msg)
    with open(save_file, 'a') as f:
        f.write(msg + '\n')

    # Расчёты с параллелизацией:
    for py_file in examples_parallel:
        for P in procs:
            time_tot, time_sol = run_file(py_file, P=P)
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
