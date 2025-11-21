#------------------------------------------------------------------
# Скрипт для запуска всех примеров решения двумерного (2D)
# дифференциального уравнения в частных производных (ДУЧП)
# параболического типа с использованием "явной" схемы
# метода конечных разностей.
#
# Конкретно, это примеры 10.1 (последовательный алгоритм),
# 10.2 (неэффективный параллельный алгоритм), и
# 10.3 (наиболее эффективный параллельный алгоритм).
#------------------------------------------------------------------

import argparse
import subprocess


#------------------------------------------------------------------
def run_file(py_file, P):
    global args
    Nx, Ny, M = int(args.Nx), int(args.Ny), int(args.M)

    hosts = f'hosts_{args.hosts}.txt'

    if args.pwd is None:
        folder = ''
    else:
        folder = '\\\\10.183.0.240\\MPI_Share\\examples\\'

    cmd = f'{args.python} {folder}{py_file} -Nx {Nx} -Ny {Ny} -M {M} --noheader'
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

        if Nx != int(res[0]) or Ny != int(res[1]) or \
            M != int(res[2]) or P != int(res[3]):
            raise subprocess.CalledProcessError(result.stdout)

        Px = int(res[4])
        Py = int(res[5])
        time_tot = float(res[6])
        time_sol = float(res[7])
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
    parser.add_argument('-Nx', default=100,
                        help='Число `Nx` интервалов сетки по координате `x`. '
                             'По умолчанию равно 100.')
    parser.add_argument('-Ny', default=100,
                        help='Число `Ny` интервалов сетки по координате `y`. '
                             'По умолчанию равно 100.')
    parser.add_argument('-M', default=2000,
                        help='Число `M` интервалов сетки по времени `t`. '
                             'По умолчанию равно 2000.')
    parser.add_argument('-T', default=5.0,
                        help='Максимальное время `T`, до которого должны проводиться '
                             'расчёты. По умолчанию равно 5.0.')
    parser.add_argument('--slow', action="store_true",
                        help='Использовать медленную реализацию решения.')
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
    example_sequential = 'Example-10-1.py'
    examples_parallel = ['Example-10-3.py', 'Example-10-2.py']

    procs_for_hosts = {
        'all_logical': [4, 9, 16, 25, 36, 49, 64, 81], # 96 max
        'all_cores': [4, 9, 16, 25, 36, 49], # 48 max
        'half_cores':[4, 9, 16, 25], # 24 max
        '4cores': [4, 9, 16, 25, 36], # 40 max
        '3cores': [4, 9, 16, 25], # 30 max
        '2cores': [4, 9, 16], # 20 max
        '1cores': [4, 9] # 10 max
    }

    args = parse_arguments()
    Nx, Ny, M = int(args.Nx), int(args.Ny), int(args.M)

    if args.pwd is None:
        procs = [4, 9]
    else:
        procs = procs_for_hosts[args.hosts]

    speed = 'slow' if args.slow else 'fast'
    save_file = f'Results10_Nx{Nx}_Ny{Ny}_M{M}_{speed}'
    if args.pwd is not None:
        save_file += f'_hosts_{args.hosts}'
    save_file += '.dat'

    msg = 'Nx\t Ny\t M\t Procs\t time_tot\t time_sol\t R_tot\t R_sol\t E_tot\t E_sol\t File'
    print(msg)
    with open(save_file, 'a') as f:
        f.write(msg + '\n')

    # Расчёты без параллелизации:
    py_file = example_sequential
    time_tot0, time_sol0 = run_file(py_file, P=1)
    msg = f'{Nx}\t {Ny}\t {M}\t 1\t {time_tot0:.6f}\t {time_sol0:.6f}\t 1\t 1\t 1\t 1\t {py_file}'
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
            msg = f'{Nx}\t {Ny}\t {M}\t {P}\t {time_tot:.6f}\t {time_sol:.6f}' + \
                  f'\t {R_tot:.4f}\t {R_sol:.4f}\t {E_tot:.4f}\t {E_sol:.4f}\t {py_file}'
            print(msg)
            with open(save_file, 'a') as f:
                f.write(msg + '\n')

#------------------------------------------------------------------
