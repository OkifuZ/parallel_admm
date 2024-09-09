
import numpy as np

pre = '''
scene_name = "ball"


[global]
dt = 0.0333
pin = [
    156, 171, 172, 187, 188, 203, 204, 219, 220, 235, 236, 238, 241, 
    245, 250, 256, 263, 264, 266, 269, 273, 278, 284, 291, 292, 294, 
    297, 301, 306, 312, 319, 320, 322, 325, 329, 334, 340, 347, 348, 
    350, 353, 357, 362, 368, 375, 376, 377, 378, 379, 380, 381, 382, 
    460, 461, 462, 463, 464, 465, 466, 544, 545, 546, 547, 548, 549, 
    550, 628, 629, 630, 631, 632, 633, 634, 712, 713, 714, 715, 716, 
    717, 718]
g = 0.98
density = 1.1

[solver]
admm_max_iter = 25
DCD_interval = 5
GS_max_iter = 5
admm.warmstart_Ue = true
admm.warmstart_Uc = true
contact.enable = true
contact.w_scale = 2000.0
contact.kappa = 1e6
contact.beta = 1e5
contact.mu = 0.5

damp.kl = 1e-3
damp.km = 1e-4

[DCD]
narrow.max_collision = 20000
narrow.thickness = 5e-2
broad.radius = 5e-2
broad.strategy = 'rebuild'


[[mesh]]
f.path = 'trimesh/bowl.obj'
f.type = 'tri'
f.color = 0
t.scale = {}
t.translate = [0, 2.0, 0]
t.rotate = [0, 0, 0]
k.stretch = 10000.0
k.bending = 0.2
k.min_limit = 0.90
k.max_limit = 1.1
k.use_limit = true

'''

post = '''

[[collider]]
type = 'plane'
center = [0, 0, 0]
halfside = 5.0
normal = [0, 1, 0]

[viewer]
scale = 5
color.background = [0.1, 0.1, 0.1]
enable_ground = true


'''

ball_temp = '''
[[mesh]]
f.path = 'tetmesh/coarse_ball.msh'
f.type = 'tet'
f.color = {}
t.scale = {}
t.translate = [{}, {}, {}]
t.rotate = [0, 0, 0]
k.stretch = 2000.0
k.min_limit = 0.95
k.max_limit = 1.05
k.use_limit = true
'''

ball_sacle = 0.4

ball_r = 0.2
bowl_r = 4 * ball_sacle
ball_pos = np.arange(-bowl_r + 2 * ball_r, bowl_r, ball_r * 2 + 0.1)

start_height = 2.3
height = np.arange(start_height, start_height + 3.5, ball_r * 2 + 0.1)

color = 1
print(pre.format(ball_sacle))
for h in height:
    for i in ball_pos:
        for j in ball_pos:
            print(ball_temp.format(color, ball_r, i, h, j))
            color += 1
print(post)