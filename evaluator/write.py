import shutil
import os
import pickle

import torch
import numpy as np

import trajnetplusplustools

## Parallel Compute
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

import scipy.io
import quantized_TF
from transformer.batch import subsequent_mask


def pre_process_test(sc_, obs_len=8):
    obs_frames = [primary_row.frame for primary_row in sc_[0]][:obs_len]
    last_frame = obs_frames[-1]
    sc_ = [[row for row in ped if row.frame in obs_frames] for ped in sc_ if ped[0].frame <= last_frame]
    return sc_


def process_scene(model, clusters, paths, args):
    ## For each scene, get predictions
    ## Taken snippet from test_trajnetpp_quantizedTF.py
    batch = {'src': []}
    device = 'cuda:0'
    pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)  # ALL ped
    vel_scene = np.zeros_like(pos_scene)
    vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]
    attr_scene = np.concatenate((pos_scene, vel_scene), axis=2)
    batch['src'] = torch.Tensor(attr_scene[:args.obs_length]).permute(1, 0, 2)
############################################################################################################

    scale = np.random.uniform(0.5, 2)
    n_in_batch = batch['src'].shape[0]
    speeds_inp = batch['src'][:, 1:, 2:4]
    inp = torch.tensor(
        scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                    -1)).to(
        device)
    src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
    start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)
    dec_inp = start_of_seq

    for i in range(args.pred_length):
        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
        out = model(inp, dec_inp, src_att, trg_att)
        dec_inp=torch.cat((dec_inp,out[:,-1:].argmax(dim=2)),1)

    preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()

    multimodal_outputs = {}
    multimodal_outputs[0] = [preds_tr_b[0], preds_tr_b[1:].transpose(1, 0, 2)]
    return multimodal_outputs


def process_multimodal_scene(model, clusters, paths, args):
    ## For each scene, get predictions
    ## Taken snippet from test_trajnetpp_quantizedTF.py
    batch = {'src': []}
    device = 'cuda:0'
    pos_scene = trajnetplusplustools.Reader.paths_to_xy(paths)  # ALL ped
    vel_scene = np.zeros_like(pos_scene)
    vel_scene[1:] = pos_scene[1:] - pos_scene[:-1]
    attr_scene = np.concatenate((pos_scene, vel_scene), axis=2)
    batch['src'] = torch.Tensor(attr_scene[:args.obs_length]).permute(1, 0, 2)
############################################################################################################
    scale = np.random.uniform(0.5, 2)
    n_in_batch = batch['src'].shape[0]
    speeds_inp = batch['src'][:, 1:, 2:4]
    inp = torch.tensor(
        scipy.spatial.distance.cdist(speeds_inp.reshape(-1, 2), clusters).argmin(axis=1).reshape(n_in_batch,
                                                                                                    -1)).to(
        device)
    src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
    start_of_seq = torch.tensor([clusters.shape[0]]).repeat(n_in_batch).unsqueeze(1).to(device)

    multimodal_outputs = {}
    for sam in range(args.modes):
        dec_inp = start_of_seq

        for i in range(args.pred_length):
            trg_att = subsequent_mask(dec_inp.shape[1]).repeat(n_in_batch, 1, 1).to(device)
            out = model.predict(inp, dec_inp, src_att, trg_att)
            h=out[:,-1]
            dec_inp=torch.cat((dec_inp,torch.multinomial(h,1)),1)


        preds_tr_b=clusters[dec_inp[:,1:].cpu().numpy()].cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
        if sam == 0:
            multimodal_outputs[0] = [preds_tr_b[0], preds_tr_b[1:].transpose(1, 0, 2)]
        else:
            multimodal_outputs[sam] = [preds_tr_b[0], []]
    return multimodal_outputs


def load_model(args):
    mat = scipy.io.loadmat(os.path.join(args.dataset_folder, args.dataset_name, "clusters.mat"))
    clusters=mat['centroids']
    device = 'cuda:0'
    model=quantized_TF.QuantizedTF(clusters.shape[0], clusters.shape[0]+1, clusters.shape[0], N=args.layers,
                                   d_model=args.emb_size, d_ff=1024, h=args.heads).to(device)
    model.load_state_dict(torch.load(f'models/QuantizedTF/{args.name}/{args.epoch}.pth'))
    model.to(device)
    model.eval()
    return model, clusters


def main(args=None):
    ## List of .json file inside the args.path (waiting to be predicted by the testing model)
    datasets = sorted([f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) if not f.startswith('.') and f.endswith('.ndjson')])
    seq_length = args.obs_length + args.pred_length

    ## Extract Model names from arguments and create its own folder in 'test_pred' for storing predictions
    ## WARNING: If Model predictions already exist from previous run, this process SKIPS WRITING
    model_name = 'traj_transformer'
    model_name = model_name + '_modes' + str(args.modes)

    ## Check if model predictions already exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)
    if not os.path.exists(args.path + model_name):
        os.makedirs(args.path + model_name)
    else:
        print('Predictions corresponding to {} already exist.'.format(model_name))
        print('Loading the saved predictions')
        return

    ## Start writing predictions in dataset/test_pred
    for dataset in datasets:
        # Model's name
        name = dataset.replace(args.path.replace('_pred', '') + 'test/', '') + '.ndjson'
        print('NAME: ', name)

        # Loading the model
        model, clusters = load_model(args)

        # Read Scenes from 'test' folder
        reader = trajnetplusplustools.Reader(args.path.replace('_pred', '') + dataset + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename (for goals)
        scenes = [(dataset, s_id, pre_process_test(s, args.obs_length)) for s_id, s in reader.scenes()]

        # Get the model prediction and write them in corresponding test_pred file
        # VERY IMPORTANT: Prediction Format
        # The predictor function should output a dictionary. The keys of the dictionary should correspond to the prediction modes.
        # ie. predictions[0] corresponds to the first mode. predictions[m] corresponds to the m^th mode.... Multimodal predictions!
        # Each modal prediction comprises of primary prediction and neighbour (surrrounding) predictions i.e. predictions[m] = [primary_prediction, neigh_predictions]
        # Note: Return [primary_prediction, []] if model does not provide neighbour predictions
        # Shape of primary_prediction: Tensor of Shape (Prediction length, 2)
        # Shape of Neighbour_prediction: Tensor of Shape (Prediction length, n_tracks - 1, 2).
        # (See LSTMPredictor.py for more details)
        scenes = tqdm(scenes)
        with open(args.path + '{}/{}'.format(model_name, name), "a") as myfile:
            ## Get all predictions in parallel. Faster!
            if args.modes == 1:
                pred_list = Parallel(n_jobs=1)(delayed(process_scene)(model, clusters, paths, args)
                                                for (_, _, paths) in scenes)
            else:
                pred_list = Parallel(n_jobs=1)(delayed(process_multimodal_scene)(model, clusters, paths, args)
                                                for (_, _, paths) in scenes)
            ## Write All Predictions
            for (predictions, (_, scene_id, paths)) in zip(pred_list, scenes):
                ## Extract 1) first_frame, 2) frame_diff 3) ped_ids for writing predictions
                observed_path = paths[0]
                frame_diff = observed_path[1].frame - observed_path[0].frame
                first_frame = observed_path[args.obs_length-1].frame + frame_diff
                ped_id = observed_path[0].pedestrian
                ped_id_ = []
                for j, _ in enumerate(paths[1:]): ## Only need neighbour ids
                    ped_id_.append(paths[j+1][0].pedestrian)

                ## Write SceneRow
                scenerow = trajnetplusplustools.SceneRow(scene_id, ped_id, observed_path[0].frame, 
                                                            observed_path[0].frame + (seq_length - 1) * frame_diff, 2.5, 0)
                # scenerow = trajnetplusplustools.SceneRow(scenerow.scene, scenerow.pedestrian, scenerow.start, scenerow.end, 2.5, 0)
                myfile.write(trajnetplusplustools.writers.trajnet(scenerow))
                myfile.write('\n')

                for m in range(len(predictions)):
                    prediction, neigh_predictions = predictions[m]
                    ## Write Primary
                    for i in range(len(prediction)):
                        track = trajnetplusplustools.TrackRow(first_frame + i * frame_diff, ped_id,
                                                                prediction[i, 0].item(), prediction[i, 1].item(), m, scene_id)
                        myfile.write(trajnetplusplustools.writers.trajnet(track))
                        myfile.write('\n')

                    ## Write Neighbours (if non-empty)
                    if len(neigh_predictions):
                        for n in range(neigh_predictions.shape[1]):
                            neigh = neigh_predictions[:, n]
                            for j in range(len(neigh)):
                                track = trajnetplusplustools.TrackRow(first_frame + j * frame_diff, ped_id_[n],
                                                                        neigh[j, 0].item(), neigh[j, 1].item(), m, scene_id)
                                myfile.write(trajnetplusplustools.writers.trajnet(track))
                                myfile.write('\n')
    print('')

if __name__ == '__main__':
    main()