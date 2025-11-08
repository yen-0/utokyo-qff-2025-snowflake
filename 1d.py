import random
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from PIL import Image
import tempfile
import glob
import os

backend = AerSimulator()

N = 20
size = 10
scale = 50
parent_length = scale * (N + 1)

DEPTH = 30
qc = random_circuit(N, DEPTH, measure = True)
qc = transpile(qc, backend, optimization_level =  0)
transpiled_qc = generate_preset_pass_manager(backend=backend).run(qc)
sampling_job = Sampler(mode=backend).run([transpiled_qc])
sampling_result = sampling_job.result()

sample_counts = sampling_result[0].data.c.get_counts()
plot_histogram(sample_counts)

conditions = list(sample_counts.keys())
l = len(conditions)
lengths = [sum([sample_counts[j] for j in conditions if j[N-i-1] == "1"])  for i in range(N)]
plt.bar(range(N), lengths, 0.35)
print(lengths)

max_len = max(lengths)


fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.axis('off')

y_shift = -8

def R(x, y):
    return (x*cos_r - y*sin_r, x*sin_r + y*cos_r+ y_shift)

for i in range(6):
    rot = i * (math.pi / 3)
    
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)
    x0, y0 = 0, 0
    x1, y1 =size*(N+1), 0
    rx0, ry0 = R(x0, y0)
    rx1, ry1 = R(x1, y1)
    ax.plot([rx0, rx1], [ry0, ry1], linewidth=3, color='black')
    ax.text(0, (N+1)*size*0.9, f"{lengths}", ha="center")

    angle_up =  60 * math.pi / 180
    angle_dn = -60 * math.pi / 180
    for n, seg_len in enumerate(lengths):
        px = size * n
        py = 0
        h = (seg_len / max_len) * scale
        cx_up = px + h * math.cos(angle_up)
        cy_up = py + h * math.sin(angle_up)
        cx_dn = px + h * math.cos(angle_dn)
        cy_dn = py + h * math.sin(angle_dn)
        
        rpx, rpy = R(px, py)
        rcxu, rcyu = R(cx_up, cy_up)
        rcxd, rcyd = R(cx_dn, cy_dn)

        ax.plot([rpx, rcxu], [rpy, rcyu], linewidth=3, color='black')
        ax.plot([rpx, rcxd], [rpy, rcyd], linewidth=3, color='black')


Re = (N+1) * size
ax.set_xlim(-Re, Re)
ax.set_ylim(-Re+y_shift, Re)
#plt.show()
FILENAME = f"00{DEPTH}"+  datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(FILENAME+".png", dpi=120, bbox_inches='tight', pad_inches=0.1)

qc.draw(output="mpl", filename = "circuit"+ FILENAME + ".jpg")
