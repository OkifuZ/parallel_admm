from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy import linalg as LA


dt = 0.03

def wander_A_B(pos_A:np.ndarray, pos_B:np.ndarray, speed:float):
    round = 4
    dist = LA.norm(pos_A - pos_B)
    step_num = dist / speed / dt 
    pts = np.linspace(pos_A, pos_B, int(step_num))
    pts = np.concatenate((pts, pts[::-1]))
    tup = ()
    for _ in range(round):
        tup = tup + (pts,)
    pos = np.vstack(tup)

    quat = np.tile(R.identity().as_quat(), (pos.shape[0], 1))    

    return pos, quat


def rotate_y(total_degree:float, speed_degree:float):
    rotaxis = np.array([0, 1, 0])
    step_num = total_degree / speed_degree 
    rotvecs = np.linspace(0 * rotaxis, total_degree / 180 * np.pi * rotaxis, int(step_num))
    quats = np.zeros((rotvecs.shape[0], 4))
    for i in range(rotvecs.shape[0]):
        quats[i,:] = R.from_rotvec(rotvecs[i]).as_quat()
    pos = np.zeros((quats.shape[0], 3))
    return pos, quats

def still(frame_num:int):
    quat = np.tile(R.identity().as_quat(), (frame_num, 1))
    pos = np.zeros((frame_num, 3))
    return pos, quat
    

if __name__ == '__main__':
    # pos, quat = wander_A_B(np.asarray([0, 0, -6]), np.asarray([0, 0, 6]), 2.0)
    # np.savez('./resource/track/wander_AB.npz', 
    #          pos=pos, quat=quat)

    pos, quat = still(33)
    pos_, quat_ = rotate_y(3600, 5.4)
    pos = np.concatenate((pos, pos_), axis = 0)
    quat = np.concatenate((quat, quat_), axis = 0)
    print(quat)
    print(pos.shape, quat.shape)
    np.savez('./resource/track/rotate_y.npz', 
             pos=pos, quat=quat)
        
