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
import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera


#------------------------------------------------------------------
# Примем как аргумент запуска программы имя файла данных
# и проверим его:

if len(sys.argv) not in [2, 3]:
    print("\n"
          "Please run the program with the name of the data file\n"
          "which should be converted to the animation added as an argument.\n"
          "EXAMPLE: python Example-08_plot.py Example-08-1_Results.npz\n"
          "\n"
          "Optionally, you can also add as the last argument an integer\n"
          "indicating the frame step that should be used for creating\n"
          "the animation (by default, the frame step is 1, that is every\n"
          "saved time frame is included into the animation).\n"
          "EXAMPLE: python Example-08_plot.py Example-08-1_Results.npz 30"
          "\n")
    exit()

filename = sys.argv[1]

filename_base, filename_ext = os.path.splitext(filename)

if filename_ext != ".npz":
    raise IOError(f"Expected a file with the extension '.npz', but got: '{filename}'")

if not os.path.isfile(filename):
    raise FileNotFoundError(f"Cannot find the file '{filename}'")

frame_step = 1
if len(sys.argv) == 3:
    frame_step = int(sys.argv[2])

print(f"Converting data file '{filename}' to animation file '{filename_base}.mp4'\n"
      f"using the time frame step {frame_step}.\n")
print("Please wait...")

#------------------------------------------------------------------
# Зачитаем файл данных и извлечём из него массивы `x`, `t`, и `u`:

results = np.load(filename)

x = results['x']
t = results['t']
u = results['u']

# Диапазон изменений значений по оси `x`:
xmin = np.min(x)
xmax = np.max(x)

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
# Анимируем набранные кадры:
animation = camera.animate(interval=15, repeat=False, blit=True)
# И записываем их в файл в формате MPEG-4 (но можете поиграться
# с другими форматами - используйте расширение файла '.avi' и т.п.):
animation.save(filename_base + '.mp4')
# NOTE: Объект `animation` является экземпляром класса
# `matplotlib.animation.ArtistAnimation`, и он может записывать
# анимации в разных форматах - читайте его документацию:
# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.ArtistAnimation.html
print("Done.")
#------------------------------------------------------------------
