import numpy as np
import torch

import trajnetplusplustools


def pre_process_test(sc_, obs_len=8):
    obs_frames = [primary_row.frame for primary_row in sc_[0]][:obs_len]
    last_frame = obs_frames[-1]
    sc_ = [[row for row in ped] for ped in sc_ if ped[0].frame <= last_frame]
    return sc_


def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask]


def get_limits_of_missing_intervals(finite_frame_inds, obs_len):
    """
    Given a SORTED array of indices of finite frames per pedestrian, get the 
    indices which represent limits of NaN (missing) intervals in the array.
    Example (for one pedestrian):
        array = [3, 4, 5, 8, 9, 10, 13, 14, 15, 18]
        obs_len = 18

        ==>> result = [0, 3, 5, 8, 10, 13, 15, 18]
    The resulting array is an array with an even number of elements,
    because it represents pairs of start-end indices (i.e. limits) for 
    intervals that should be padded. 
        ==>> intervals to be padded later: [0, 3], [5, 8], [10, 13], [15, 18]
    """
    # Adding start and end indices
    if 0 not in finite_frame_inds:
        finite_frame_inds = np.insert(finite_frame_inds, 0, -1) 
    if obs_len not in finite_frame_inds:
        finite_frame_inds = \
            np.insert(finite_frame_inds, len(finite_frame_inds), obs_len)

    # Keeping only starts and ends of continuous intervals
    limits, interval_len = [], 1
    for i in range(1, len(finite_frame_inds)):
        # If this element isn't the immediate successor of the previous
        if finite_frame_inds[i] > finite_frame_inds[i - 1] + 1:
            if interval_len:
                # Add the end of the previous interval
                if finite_frame_inds[i - 1] == -1:
                    limits.append(0)
                else:
                    limits.append(finite_frame_inds[i - 1])
                # Add the start of the new interval
                limits.append(finite_frame_inds[i])
                # If this is a lone finite element, add the next interval
                if interval_len == 1 and i != len(finite_frame_inds) - 1 \
                    and finite_frame_inds[i + 1] > finite_frame_inds[i] + 1:
                    limits.append(finite_frame_inds[i])
                    limits.append(finite_frame_inds[i + 1])
            interval_len = 0
        else:
            interval_len += 1
            
    return limits


def fill_missing_observations(pos_scene_raw, obs_len, test):
    """
    Performs the following:
        - discards pedestrians that are completely absent in 0 -> obs_len
        - discards pedestrians that have any NaNs after obs_len
        - In 0 -> obs_len:
            - finds FIRST non-NaN and fill the entries to its LEFT with it
            - finds LAST non-NaN and fill the entries to its RIGHT with it
    """

    # Discarding pedestrians that are completely absent in 0 -> obs_len
    peds_are_present_in_obs = \
        np.isfinite(pos_scene_raw).all(axis=2)[:obs_len, :].any(axis=0)
    pos_scene = pos_scene_raw[:, peds_are_present_in_obs, :]

    if not test:
        # Discarding pedestrians that have NaNs after obs_len
        peds_are_absent_after_obs = \
            np.isfinite(pos_scene).all(axis=2)[obs_len:, :].all(axis=0)
        pos_scene = pos_scene[:, peds_are_absent_after_obs, :]

    # Finding indices of finite frames per pedestrian
    finite_frame_inds, finite_ped_inds = \
        np.where(np.isfinite(pos_scene[:obs_len]).all(axis=2))
    finite_frame_inds, finite_ped_inds = \
        finite_frame_inds[np.argsort(finite_ped_inds)], np.sort(finite_ped_inds)

    finite_frame_inds_per_ped = np.split(
        finite_frame_inds, np.unique(finite_ped_inds, return_index=True)[1]
        )[1:]
    finite_frame_inds_per_ped = \
        [np.sort(frames) for frames in finite_frame_inds_per_ped]

    # Filling missing frames
    for ped_ind in range(len(finite_frame_inds_per_ped)):
        curr_finite_frame_inds = finite_frame_inds_per_ped[ped_ind]

        # limits_of_cont_ints: [start_1, end_1, start_2, end_2, ... ]
        limits_of_missing_ints = \
            get_limits_of_missing_intervals(curr_finite_frame_inds, obs_len)
        assert len(limits_of_missing_ints) % 2 == 0
            
        i = 0
        while i < len(limits_of_missing_ints):
            start_ind, end_ind = \
                limits_of_missing_ints[i], limits_of_missing_ints[i + 1]
            # If it's the beginning (i.e. first element is NaN):
            #   - pad with the right limit, else use left
            #   - include start_ind, else exclude it
            if start_ind == 0 and not np.isfinite(pos_scene[0, ped_ind]).all():
                padding_ind = end_ind 
                start_ind = start_ind 
            else:
                padding_ind = start_ind
                start_ind = start_ind + 1

            pos_scene[start_ind:end_ind, ped_ind] = pos_scene[padding_ind, ped_ind]
            i += 2

    return pos_scene


def trajnet_loader(
    data_loader, 
    args, 
    drop_distant_ped=False, 
    test=False, 
    keep_single_ped_scenes=False,
    fill_missing_obs=False
    ):
    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel = [], [], [], []
    loss_mask, seq_start_end = [], []
    non_linear_ped = torch.Tensor([]) # dummy
    num_batches = 0
    for batch_idx, (filename, scene_id, paths) in enumerate(data_loader):
        if test:
            paths = pre_process_test(paths, args.obs_len)
        
        ## Get new scene
        pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)
        if drop_distant_ped:
            pos_scene = drop_distant(pos_scene)

        if fill_missing_obs:
            pos_scene = fill_missing_observations(pos_scene, args.obs_len, test)
            full_traj = np.isfinite(pos_scene).all(axis=2).all(axis=0)
        else:
            # Removing Partial Tracks. Model cannot account for it !! NaNs in Loss
            full_traj = np.isfinite(pos_scene).all(axis=2).all(axis=0)
            pos_scene = pos_scene[:, full_traj]
        
        # Make Rel Scene
        vel_scene = np.zeros_like(pos_scene)
        vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]

        # STGAT Model needs atleast 2 pedestrians per scene.
        if sum(full_traj) > 1 or keep_single_ped_scenes:
            # Get Obs, Preds attributes
            obs_traj.append(torch.Tensor(pos_scene[:args.obs_len]))
            pred_traj_gt.append(torch.Tensor(pos_scene[-args.pred_len:]))
            obs_traj_rel.append(torch.Tensor(vel_scene[:args.obs_len]))
            pred_traj_gt_rel.append(torch.Tensor(vel_scene[-args.pred_len:]))

            # Get Seq Delimiter and Dummy Loss Mask
            seq_start_end.append(pos_scene.shape[1])
            curr_mask = torch.ones((pos_scene.shape[0], pos_scene.shape[1]))
            loss_mask.append(curr_mask)
            num_batches += 1

        if num_batches % args.batch_size != 0 and (batch_idx + 1) != len(data_loader):
            continue
        
        if len(obs_traj):
            obs_traj = torch.cat(obs_traj, dim=1).cuda()
            pred_traj_gt = torch.cat(pred_traj_gt, dim=1).cuda()
            obs_traj_rel = torch.cat(obs_traj_rel, dim=1).cuda()
            pred_traj_gt_rel = torch.cat(pred_traj_gt_rel, dim=1).cuda()
            loss_mask = torch.cat(loss_mask, dim=1).cuda().permute(1, 0)
            seq_start_end = [0] + seq_start_end
            seq_start_end = torch.LongTensor(np.array(seq_start_end).cumsum())
            seq_start_end = torch.stack((seq_start_end[:-1], seq_start_end[1:]), dim=1)
            yield (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                non_linear_ped, loss_mask, seq_start_end)
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel = [], [], [], []
            loss_mask, seq_start_end = [], []
