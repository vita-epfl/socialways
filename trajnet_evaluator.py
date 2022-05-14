import os
import argparse
import pickle

from joblib import Parallel, delayed
import scipy
import torch
from tqdm import tqdm
from itertools import chain
import trajnetplusplustools
import numpy as np

from evaluator.trajnet_evaluator import trajnet_evaluate
from evaluator.write_utils import \
    load_test_datasets, preprocess_test, write_predictions

from trajnet_loader import trajnet_loader

# Socialways
from trajnet_utils import \
    EncoderLstm, EmbedSocialFeatures, AttentionPooling, DecoderFC, Discriminator
from trajnet_utils import predict_trajnet


# ====== Hard-coded hyperparameters ======
n_latent_codes = 2
num_social_features = 3
n_lstm_layers = 1
use_social = False
# ========================================


def predict_scene(models, batch, args):
    attention, feature_embedder, encoder, decoder, D = models 

    batch = [tensor.cuda() for tensor in batch]
    # Obs traj is of the shape [pred_len, num_peds, 2]
    # Should be [num_peds, pred_len, 2]
    obs_traj = batch[0].permute(1, 0, 2)

    # Needed to generate noise for the predict function
    noise_len = args.hidden_size // 2

    # Get the predictions and save them
    multimodal_outputs = {}    
    for num_p in range(args.modes):
        # Generate noise and obtain the predictions
        noise = torch.FloatTensor(torch.rand(obs_traj.shape[0], noise_len)).cuda()
        pred_traj_fake_4d = predict_trajnet(
            obs_traj, noise, args.pred_len,
            models, n_lstm_layers, use_social
            )
        pred_traj_fake = pred_traj_fake_4d[:, :, :2]

        output_primary = pred_traj_fake[:, 0]
        output_neighs = pred_traj_fake[:, 1:]
        multimodal_outputs[num_p] = [output_primary, output_neighs]

    ################
    # FIXME / TODO:
    #   - CHECK THE PRED_TRAJ_FAKE SHAPE AND COMPARE TO STGAT
    #
    #   - Normalization of the data!!! 
    #       - they normalize the data in the training script
    #       => What should we do??? Maybe save the scaler and use it here???
    ################

    return multimodal_outputs



def load_predictor(args):

    # ====== Hyper-parameters ======
    hidden_size = args.hidden_size
    social_feature_size = args.hidden_size
    noise_len = args.hidden_size // 2
    n_next = args.pred_len
    
    # ====== Loading checkpoints ======
    checkpoint_path = os.path.join(
        '..', 'trained_models', f'{args.model}-{args.dataset_name}.pt'
        )
    checkpoint = torch.load(checkpoint_path)

    # ======= Initializing models =======
    # LSTM-based path encoder
    encoder = EncoderLstm(hidden_size, n_lstm_layers).cuda()
    feature_embedder = \
        EmbedSocialFeatures(num_social_features, social_feature_size).cuda()
    attention = AttentionPooling(hidden_size, social_feature_size).cuda()

    # Decoder
    decoder = DecoderFC(hidden_size + social_feature_size + noise_len).cuda()

    # The Discriminator parameters and their optimizer
    D = Discriminator(n_next, hidden_size, n_latent_codes).cuda()

    # ======= Loading model params =======
    attention.load_state_dict(checkpoint['attentioner_dict'])
    feature_embedder.load_state_dict(checkpoint['feature_embedder_dict'])
    encoder.load_state_dict(checkpoint['encoder_dict'])
    decoder.load_state_dict(checkpoint['decoder_dict'])
    D.load_state_dict(checkpoint['D_dict'])

    # Put all of them in evaluation mode
    attention.eval()
    feature_embedder.eval()
    encoder.eval() 
    decoder.eval() 
    D.eval()

    return attention, feature_embedder, encoder, decoder, D


def get_predictions(args):
    """
    Get model predictions for each test scene and write the predictions 
    in appropriate folders.
    """
    # List of .json file inside the args.path 
    # (waiting to be predicted by the testing model)
    datasets = sorted([
        f.split('.')[-2] for f in os.listdir(args.path.replace('_pred', '')) \
        if not f.startswith('.') and f.endswith('.ndjson')
        ])

    # Extract Model names from arguments and create its own folder 
    # in 'test_pred' for storing predictions
    # WARNING: If Model predictions already exist from previous run, 
    # this process SKIPS WRITING
    for model in args.output:
        model_name = model.split('/')[-1].replace('.pkl', '')
        model_name = model_name + '_modes' + str(args.modes)

        ## Check if model predictions already exist
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        if not os.path.exists(args.path + model_name):
            os.makedirs(args.path + model_name)
        else:
            print(f'Predictions corresponding to {model_name} already exist.')
            print('Loading the saved predictions')
            continue

        print("Model Name: ", model_name)
        # Load: attention, feature_embedder, encoder, decoder, D 
        models = load_predictor(args)
        goal_flag = False

        # Iterate over test datasets
        for dataset in datasets:
            # Load dataset
            dataset_name, scenes, scene_goals = \
                load_test_datasets(dataset, goal_flag, args)

            # Convert it to a trajnet loader
            scenes_loader = trajnet_loader(
                scenes, 
                args, 
                drop_distant_ped=False, 
                test=True,
                keep_single_ped_scenes=args.keep_single_ped_scenes,
                fill_missing_obs=args.fill_missing_obs
                ) 

            # Can be removed; it was useful for debugging
            scenes_loader = list(scenes_loader)

            # Get all predictions in parallel. Faster!
            scenes_loader = tqdm(scenes_loader)
            pred_list = Parallel(n_jobs=args.n_jobs)(
                delayed(predict_scene)(models, batch, args)
                for batch in scenes_loader
                )
            
            # Write all predictions
            write_predictions(pred_list, scenes, model_name, dataset_name, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default="eth_data", type=str
        )
    parser.add_argument(
        '--obs_len', default=9, type=int, help='observation length'
        )
    parser.add_argument(
        '--pred_len', default=12, type=int, help='prediction length'
        )
    parser.add_argument(
        '--write_only', action='store_true', help='disable writing new files'
        )
    parser.add_argument(
        '--disable-collision', action='store_true', help='disable collision metrics'
        )
    parser.add_argument(
        '--labels', required=False, nargs='+', help='labels of models'
        )
    parser.add_argument(
        '--normalize_scene', action='store_true', help='augment scenes'
        )
    parser.add_argument(
        '--modes', default=1, type=int, help='number of modes to predict'
        )
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--fill_missing_obs", default=1, type=int)
    parser.add_argument("--keep_single_ped_scenes", default=1, type=int)
    parser.add_argument("--n_jobs", default=8, type=int)

    # Socialways
    parser.add_argument('--model', '--m', default='socialWays', choices=['socialWays'])
    parser.add_argument(
        '--latent-dim', '--ld', type=int, default=10, metavar='N',
        help='dimension of latent space (default: 10)'
        )
    parser.add_argument(
        '--d-learning-rate', '--d-lr', type=float, default=1E-3, metavar='N',
        help='learning rate of discriminator (default: 1E-3)'
        )
    parser.add_argument(
        '--g-learning-rate', '--g-lr', type=float, default=1E-4, metavar='N',
        help='learning rate of generator (default: 1E-4)'
        )
    parser.add_argument(
        '--unrolling-steps', '--unroll', type=int, default=1, metavar='N',
        help='number of steps to unroll gan (default: 1)'
        )
    parser.add_argument(
        '--hidden-size', '--h-size', type=int, default=64, metavar='N',
        help='size of network intermediate layer (default: 64)'
        )

    args = parser.parse_args()

    scipy.seterr('ignore')

    args.checkpoint = os.path.join(
        '..', 'trained_models', f'{args.model}-{args.dataset_name}.pt'
        )
    args.path = os.path.join('datasets', args.dataset_name, 'test_pred/')
    args.output = [args.checkpoint]

    # Adding arguments with names that fit the evaluator module
    # in order to keep it unchanged
    args.obs_length = args.obs_len
    args.pred_length = args.pred_len

    # Writes to Test_pred
    # Does NOT overwrite existing predictions if they already exist ###
    get_predictions(args)
    if args.write_only: # For submission to AICrowd.
        print("Predictions written in test_pred folder")
        exit()

    ## Evaluate using TrajNet++ evaluator
    trajnet_evaluate(args)


if __name__ == '__main__':
    main()


