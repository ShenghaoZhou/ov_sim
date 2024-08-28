from ov_sim.ov_sim import BsplineSE3
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def dump_res(res, save_name):
    for k, v in res.items():
        res[k] = np.array(v)
    # alternatively, dump it to txt. For the sake of easy loading, we can use npz
    np.savez(f"{save_name}.npz", **res)


def gen_motion_from_traj(traj, save_name, imu_freq=200):
    bspline = BsplineSE3()
    bspline.feed_trajectory(traj)
    res = defaultdict(list)
    time_start = bspline.get_start_time()
    is_time_end = False
    t = time_start
    while not is_time_end:
        t += 1.0 / imu_freq
        R, p, w, v, _, a = bspline.get_motion(t)
        if np.allclose(p, 0) and np.allclose(v, 0) and np.allclose(a, 0):
            is_time_end = True
            print(f"End at {t}")
            break
        res['t'].append(t)
        res['R'].append(R)
        res['p'].append(p)
        res['w'].append(w)
        res['v'].append(v)
        res['a'].append(a)
    dump_res(res, save_name)


def gen_timestamp(leng, freq):
    return np.arange(0, leng / freq, 1 / freq)


def gen_for_tartan_air(path, save_dir):
    path = Path(path)
    scene_name = path.parent.parent.parent.name
    diffuclty = path.parent.parent.name
    seq_name = path.parent.name
    save_name = f"{scene_name}_{diffuclty}_{seq_name}"
    save_path = save_dir / save_name
    poses = np.loadtxt(path)
    timestamp = gen_timestamp(leng=len(poses), freq=8)
    traj = np.hstack([timestamp[:, None], poses])
    gen_motion_from_traj(traj, save_path)


if __name__ == "__main__":
    root = Path("/media/shzhou/RPNG FLASH 1/tartan_air/tartanair_full_pose")
    file_lst = []
    for scene in root.iterdir():
        if scene.is_dir():
            for hard_level in scene.iterdir():
                if hard_level.is_dir():
                    for seq in hard_level.iterdir():
                        if seq.is_dir():
                            file_lst.append(seq / "pose_left.txt")
    # for file in tqdm(file_lst):
    #     gen_for_tartan_air(file)
    gen_fn = partial(gen_for_tartan_air, save_dir=Path(
        "/home/shzhou/project/inertia_only/diffusion/tartan_air_traj_dataset"))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(gen_fn, file_lst), total=len(file_lst)):
            pass
