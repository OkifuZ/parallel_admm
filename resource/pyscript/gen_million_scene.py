import random


scene_file_part1 = r'''

scene_name = "one million"
out_file = 'C:\codebase\admm_elasticity\result\one million'
seperate_out = false
show_windows = true

[global]
dt = 0.02
pin = []
g = 2.98
# density = 10.0

[solver]
admm_max_iter = 30
DCD_interval = 5
GS_max_iter = 5
contact.use_CCD = false
admm.warmstart_Ue = true
admm.warmstart_Uc = true
contact.enable = true
contact.w_scale = 250.0
contact.kappa = 1e6
contact.unique = false
contact.beta = 1e5
contact.mu = 0.1
damp.kl = 1e-4
damp.km = 1e-3

[DCD]
narrow.max_collision = 40000
narrow.thickness = 3e-2
broad.radius = 3e-2
broad.strategy = 'rebuild'

'''

scene_file_part2 = r'''

[[mesh]]
f.path = 'flat\flat_v3452.obj'
f.type = 'static'
f.color = 6
t.scale = 15.0
t.translate = [0.0, -2.5, 0]
t.rotate = [0, 0, 0]
k.stretch = 8000.0
k.min_limit = 0.98
k.max_limit = 1.02
k.bending = 0.001
# k.use_limit = true
k.use_limit = false
density = 2.5
animate = ""

[mesh_post]
center_ids = []

[[collider]]
type = 'plane'
center = [0, -0.6, 0]
halfside = 5.0
normal = [0, 1, 0]

[viewer]
scale = 10
color.background = [0.1, 0.1, 0.1]
enable_ground = false
'''

mesh_desc = '''
[[mesh]]
f.path = 'tetmesh/mydonut_999.msh'
f.type = 'tet'
f.color = 1
t.scale = 1
t.translate = [{tx}, {ty}, {tz}]
t.rotate = [{rx}, {ry}, {rz}]
k.stretch = {strech}
k.min_limit = 0.99
k.max_limit = 1.01
k.use_limit = false
density = {density}

'''
widthx = 8
widthz = 8
layer = 5
num = widthx * widthz * layer
density = [50.0] * num
strech = [10000.0] * num

maj_radius = 1.3
min_radius = 0.3

mesh_desc_list = []

posx_list = []
posy_list = []
posz_list = []

for k in range(layer):
    for i in range(widthx):
        for j in range(widthz):
            hori_gap = 0.15
            vert_gap = 0.3
            rdx = (random.uniform(-0.5, 0.5)) * hori_gap
            rdz = (random.uniform(-0.5, 0.5)) * hori_gap
            print(rdx, rdz)
            posx = i * (maj_radius * 2 + hori_gap) + rdx
            posz = j * (maj_radius * 2 + hori_gap) + rdz
            posy = k * (min_radius * 2 + vert_gap)
            posx_list.append(posx)
            posy_list.append(posy)
            posz_list.append(posz)

px_total = 0
py_total = 0
pz_total = 0
for i in range(num):
    px_total += posx_list[i]
    py_total += posy_list[i]
    pz_total += posz_list[i]
px_total /= num
py_total /= num
pz_total /= num

for i in range(num):
    posx_list[i] -= px_total
    posy_list[i] -= py_total
    posz_list[i] -= pz_total

for i in range(num):
    posy = 0
    rotx = roty = rotz = 0
    mesh_desc_list.append(
        mesh_desc.format(
        tx=posx_list[i], ty=posy_list[i], tz=posz_list[i],
        rx = rotx, ry = roty, rz=rotz,
        strech = strech[i],
        density = density[i]
    ))

with open('./resource/scene/one_million.toml', 'w') as fp:
    fp.write(scene_file_part1)
    for msc in mesh_desc_list:
        fp.write(msc)
    fp.write(scene_file_part2)

    fp.close()

