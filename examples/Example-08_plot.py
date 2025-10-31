#------------------------------------------------------------------
# Программа для создания анимации решения дифференциального
# уравнения в частных производных параболического типа,
# найденного в примерах 8.0, 8.1, и 9.1.
#
# Пример запуска программы:
#    python Example-08_plot.py Example-08-1_Results.npz
# где "Example-08-1_Results.npz" - это имя файла с данными
# решения, которые нужно конвертировать в анимацию.
# Анимация создаётся в формате MPEG-4 и записывается в файл
# с тем же именем, но с расширением ".mp4" вместо ".npz"
# (в примере выше будет создан файл "Example-08-1_Results.mp4").
#
# Для работы программы нужно установить программы FFmpeg:
# https://www.ffmpeg.org/download.html
# (добавив в переменную среды окружения "Path" путь к файлу
# "ffmpeg.exe")
# Установить библиотеку для работы с FFmpeg через Python:
#   pip install ffmpeg-python
# И установить библиотеку, облегчающую создание анимаций из
# matplotlib рисунков:
#   pip install celluloid
#
#------------------------------------------------------------------
# Эта программа (в её оригинальном виде) обсуждается в лекциях
# Д.В. Лукьяненко "8. Решение задач для уравнений в частных
# производных.Ч.1" начиная со времени 18:09 и "9. Решение задач
# для уравнений в частных производных.Ч.2" начиная со времени 26:48:
# https://youtu.be/6SYN28B_iyE&t=1089
# https://youtu.be/QuPp0AMiBfU&t=1609
#------------------------------------------------------------------

import os
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera


#------------------------------------------------------------------
# Примем как аргументы запуска программы имя файла данных и
# (опционально) шаг по моментам времени, которые нужно включить
# в анимацию:

parser = argparse.ArgumentParser(
            prog='python Example-08_plot.py',
            description='Конвертер файла данных в анимацию для '
                        'Примеров 8.0, 8.1, и 9.1.',
            #usage='%(prog)s [options]'
)
parser.add_argument('filename',
                    help='Имя файла данных, для которых нужно создать анимацию. '
                         'Должен быть записан в формате ".npz".')
parser.add_argument('-s', '--step', default=1,
                    help='Шаг по моментам времени, которые будут включены '
                         'в анимацию. По умолчанию равно 1 - включать '
                         'каждый записанный момент времени.')

args = parser.parse_args()

filename = args.filename
filename_base, filename_ext = os.path.splitext(filename)

if filename_ext != ".npz":
    raise IOError(f"Ожидался файл с расширением '.npz', а получен: '{filename}'")

if not os.path.isfile(filename):
    raise FileNotFoundError(f"Не могу найти файл '{filename}'")

frame_step = int(args.step)

print(f"Конвертируем файл данных '{filename}' в файл анимации '{filename_base}.mp4' "
      f"используя шаг {frame_step} по записанным моментам времени.\n")

#------------------------------------------------------------------
# Зачитаем файл данных и извлечём из него массивы `x`, `t`, и `u`:

results = np.load(filename)

x = results['x']
t = results['t']
u = results['u']

# Диапазон изменений значений по оси `x`:
xmin = np.min(x)
xmax = np.max(x)

#------------------------------------------------------------------
# Поскольку создание видео занимает время, добавим простой
# progress bar, который будет использоваться внутри цикла
# по `m` и внутри вызова `animation.save()`:
progress_bar_prefix = '    Рисуем:'
M = int(len(u) / frame_step)

def progress_bar(current_frame: int, total_frames: int):
    global progress_bar_prefix, M
    # Есть проблема - в нашем случае matplotlib не заполняет
    # правильно значение `total_frames` - see comment in:
    # https://github.com/matplotlib/matplotlib/blob/4aeb773422464799998d900198b35cb80e94b3e1/lib/matplotlib/animation.py#L1111
    if total_frames is None:
        total_frames = M
    prefix = progress_bar_prefix
    suffix = 'Complete'; fill = '█'
    bar_length = 60
    percent = ("{0:.1f}").format(100 * current_frame / total_frames)
    filled_length = int(round(bar_length * current_frame / total_frames))
    bar = fill * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}   ')
    sys.stdout.flush()

#------------------------------------------------------------------
# Готовимся рисовать найденное решение для
# нужных моментов времени:
plt.style.use('dark_background')
fig = plt.figure()
camera = Camera(fig)
ax = plt.axes(xlim=(xmin, xmax), ylim=(-2,2))
ax.set_xlabel('x')
ax.set_ylabel('u')

# Цикл по моментам времени, для которых записан файл
# данных - будем рисовать кадры с шагом `frame_step`:
for m in range(0, len(u), frame_step):
    # Рисуем кадр анимации:
    ax.plot(x, u[m], color='y', ls='-', lw=2)
    ax.text(0.8, -1.6, f't = {t[m]:.3f}')
    # Добавляем его в анимацию:
    camera.snap()
    progress_bar(m // frame_step, M)
# Анимируем набранные кадры:
animation = camera.animate(interval=15, repeat=False, blit=True)
# NOTE: Объект `animation` является экземпляром класса
# `matplotlib.animation.ArtistAnimation`, и он может записывать
# анимации в разных форматах - читайте его документацию:
# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.ArtistAnimation.html

# Наконец, запишем анимацию в файл в формате MPEG-4 (но можете поиграться
# с другими форматами - используйте расширение файла '.avi' и т.п.).
progress_bar_prefix = 'Записываем:'
animation.save(filename_base + '.mp4',
               progress_callback=progress_bar)

#------------------------------------------------------------------
