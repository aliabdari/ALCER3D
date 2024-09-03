'''
This module is related to the checking of the permutations of margins for the two datasets, including 3dfront and museum
'''

import pickle
from DNNs import GRUNet, OneDimensionalCNN
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import StepLR
import train_utility
from Data_utils import DescriptionScene, DescriptionSceneMuseum
from tqdm import tqdm
import argparse
from datetime import date


def collate_fn(data):
    # desc
    tmp_description_povs = [x[0] for x in data]
    tmp = pad_sequence(tmp_description_povs, batch_first=True)
    descs_pov = pack_padded_sequence(tmp,
                                     torch.tensor([len(x) for x in tmp_description_povs]),
                                     batch_first=True,
                                     enforce_sorted=False)

    tmp_pov = [x[1] for x in data]
    padded_pov = pad_sequence(tmp_pov, batch_first=True)
    padded_pov = torch.transpose(padded_pov, 1, 2)

    indexes = [x[2] for x in data]
    return descs_pov, padded_pov, indexes


def get_similarity_relations(simmodel, dataset):
    return pickle.load(open(f'../scenes_relevances/relevance_{simmodel}_{dataset}_normalized.pkl', 'rb'))


def get_similarity_function(dataset, *args_):
    if len(args_) == 1:
        print('One Model ', args_[0], ' has been chosen')
        return pickle.load(open(f'../scenes_relevances/relevance_{args_[0]}_{dataset}_normalized.pkl', 'rb'))
    elif len(args_) == 2:
        print('Two Models including ', args_[0], ' and ', args_[1], ' have been chosen')
        s1 = pickle.load(open(f'../scenes_relevances/relevance_{args_[0]}_{dataset}_normalized.pkl', 'rb'))
        s2 = pickle.load(open(f'../scenes_relevances/relevance_{args_[1]}_{dataset}_normalized.pkl', 'rb'))
        s_final = {}
        for p in s1:
            s_final[p] = (s1[p] + s2[p]) / 2
        return s_final
    elif len(args_) == 3:
        print('Three Models including ', args_[0], ' and ', args_[1], ' and ', args_[2], ' have been chosen')
        s1 = pickle.load(open(f'../scenes_relevances/relevance_{args_[0]}_{dataset}_normalized.pkl', 'rb'))
        s2 = pickle.load(open(f'../scenes_relevances/relevance_{args_[1]}_{dataset}_normalized.pkl', 'rb'))
        s3 = pickle.load(open(f'../scenes_relevances/relevance_{args_[2]}_{dataset}_normalized.pkl', 'rb'))
        s_final = {}
        for p in s1:
            s_final[p] = (s1[p] + s2[p] + s3[p]) / 3
        return s_final
    else:
        assert ('ERROR')


def start_train(args):
    date_result = date.today()
    if args.simmodel2 is None and args.simmodel3 is None:
        approach_name = f'{date_result}_{args.simmodel}_{args.dataset}_status{args.status}'
        similarity_metric = get_similarity_function(args.dataset, args.simmodel)
    elif args.simmodel2 is not None and args.simmodel3 is None:
        approach_name = f'{date_result}_{args.simmodel}_{args.simmodel2}_{args.dataset}_status{args.status}'
        similarity_metric = get_similarity_function(args.dataset, args.simmodel, args.simmodel2)
    else:
        approach_name = f'{date_result}_{args.simmodel}_{args.simmodel2}_{args.simmodel3}_{args.dataset}_status{args.status}'
        similarity_metric = get_similarity_function(args.dataset, args.simmodel, args.simmodel2, args.simmodel3)

    output_feature_size = 256

    relevance_info = similarity_metric
    is_customized_margin = True if args.custom_margin else False

    if is_customized_margin:
        margins = args.margins
        thresholds = args.thresholds
        margins_ = dict()
        if args.status == 0:
            margins_['margin_low'] = margins[0]
            margins_['margin_high'] = margins[1]
        elif args.status == 1:
            margins_['margin_low'] = margins[0]
            margins_['margin_mid'] = margins[1]
            margins_['margin_high'] = margins[2]
        else:
            margins_['margin_low'] = margins[0]
            margins_['margin_mids'] = margins[1]
            margins_['margin_midl'] = margins[2]
            margins_['margin_high'] = margins[3]

        print(margins_)
        print(thresholds)
        do_training(approach_name=approach_name, is_customized_margin=is_customized_margin,
                    margins=margins_, output_feature_size=output_feature_size, relevance_info=relevance_info,
                    status=args.status, thres=thresholds, selected_dataset=args.dataset)
    else:
        do_training(approach_name=approach_name, is_customized_margin=is_customized_margin,
                    output_feature_size=output_feature_size, relevance_info=relevance_info,
                    status=args.status, selected_dataset=args.dataset)


def do_training(approach_name, is_customized_margin, output_feature_size, relevance_info, status, selected_dataset, thres=None, margins=None):
    is_bidirectional = True
    model_desc_pov = GRUNet(hidden_size=output_feature_size, num_features=512, is_bidirectional=is_bidirectional)
    model_pov = OneDimensionalCNN(in_channels=512, out_channels=512, kernel_size=5, feature_size=output_feature_size)
    cont_loss = train_utility.LossContrastive(name=approach_name, patience=25, delta=0.0001)
    num_epochs = 50
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device = ', device)
    model_desc_pov.to(device=device)
    model_pov.to(device=device)

    #     data section
    train_indices, val_indices, test_indices = train_utility.retrieve_indices(dataset=args.dataset)
    descriptions_path, pov_path = train_utility.get_entire_data(dataset=args.dataset)
    if selected_dataset == '3dfront':
        dataset = DescriptionScene(data_description_path=descriptions_path, mem=True, data_scene_path=pov_path)
    elif selected_dataset == 'museums':
        dataset = DescriptionSceneMuseum(data_description_path=descriptions_path, mem=True, data_scene_path=pov_path)

    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    test_subset = Subset(dataset, test_indices.tolist())
    train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4)
    '''Train Procedure'''
    params = list(model_desc_pov.parameters()) + list(model_pov.parameters())
    optimizer = torch.optim.Adam(params, lr=0.008)
    step_size = 27
    gamma = 0.75
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_r10 = 0
    print('Train procedure ...')
    for _ in tqdm(range(num_epochs)):

        if not cont_loss.is_val_improving():
            print('Early Stopping !!!')
            break

        total_loss_train = 0
        total_loss_val = 0
        num_batches_train = 0
        num_batches_val = 0

        output_description_val = torch.empty(len(val_indices), output_feature_size)
        output_pov_val = torch.empty(len(val_indices), output_feature_size)

        for i, (data_desc_pov, data_pov, indexes) in enumerate(train_loader):
            data_desc_pov = data_desc_pov.to(device)
            data_pov = data_pov.to(device)

            optimizer.zero_grad()

            output_desc_pov = model_desc_pov(data_desc_pov)
            output_pov = model_pov(data_pov)

            multiplication_dp = train_utility.cosine_sim(output_desc_pov, output_pov)
            if args.custom_margin:
                margin_tensor = train_utility.get_margin_tensor(indexes=indexes, relevance_info=relevance_info, margins=margins, status=status,
                                                                thresholds=thres)
                loss_contrastive = cont_loss.calculate_loss(multiplication_dp, margin_tensor=margin_tensor)
            else:
                loss_contrastive = cont_loss.calculate_loss(multiplication_dp)

            loss_contrastive.backward()

            optimizer.step()

            total_loss_train += loss_contrastive.item()
            num_batches_train += 1

        scheduler.step()
        print(scheduler.get_last_lr())
        epoch_loss_train = total_loss_train / num_batches_train

        model_desc_pov.eval()
        model_pov.eval()
        # Validation Procedure
        with torch.no_grad():
            for j, (data_desc_pov, data_pov, indexes) in enumerate(val_loader):

                data_desc_pov = data_desc_pov.to(device)
                data_pov = data_pov.to(device)

                output_desc_pov = model_desc_pov(data_desc_pov)
                output_pov = model_pov(data_pov)

                initial_index = j * batch_size
                final_index = (j + 1) * batch_size
                if final_index > len(val_indices):
                    final_index = len(val_indices)

                output_description_val[initial_index:final_index, :] = output_desc_pov
                output_pov_val[initial_index:final_index, :] = output_pov

                multiplication_dp = train_utility.cosine_sim(output_desc_pov, output_pov)
                if is_customized_margin:
                    margin_tensor = train_utility.get_margin_tensor(indexes=indexes, relevance_info=relevance_info,
                                                                    margins=margins, status=status, thresholds=thres)
                    loss_contrastive = cont_loss.calculate_loss(multiplication_dp, margin_tensor=margin_tensor)
                else:
                    loss_contrastive = cont_loss.calculate_loss(multiplication_dp)

                total_loss_val += loss_contrastive.item()
                num_batches_val += 1

            epoch_loss_val = total_loss_val / num_batches_val

            print('Loss Train', epoch_loss_train)
            cont_loss.on_epoch_end(epoch_loss_train, train=True)
            print('Loss Val', epoch_loss_val)
            cont_loss.on_epoch_end(epoch_loss_val, train=False)

        r1, r5, r10, _, _, _, _, _, _, _ = train_utility.evaluate(output_description=output_description_val,
                                                                  output_scene=output_pov_val, section='val')

        model_desc_pov.train()
        model_pov.train()

        if r10 > best_r10:
            best_r10 = r10
            train_utility.save_best_model(approach_name, model_pov.state_dict(), model_desc_pov.state_dict())
    bm_pov, bm_desc_pov = train_utility.load_best_model(approach_name)
    model_pov.load_state_dict(bm_pov)
    model_desc_pov.load_state_dict(bm_desc_pov)
    model_pov.eval()
    model_desc_pov.eval()
    output_description_test = torch.empty(len(test_indices), output_feature_size)
    output_pov_test = torch.empty(len(test_indices), output_feature_size)
    # Evaluate test set
    with torch.no_grad():
        for j, (data_desc_pov, data_pov, indexes) in enumerate(test_loader):

            data_desc_pov = data_desc_pov.to(device)
            data_pov = data_pov.to(device)

            output_desc_pov = model_desc_pov(data_desc_pov)
            output_pov = model_pov(data_pov)

            initial_index = j * batch_size
            final_index = (j + 1) * batch_size
            if final_index > len(test_indices):
                final_index = len(test_indices)
            output_description_test[initial_index:final_index, :] = output_desc_pov
            output_pov_test[initial_index:final_index, :] = output_pov
    ds1, ds5, ds10, sd1, sd5, sd10, ndgc_10, ndcg, ds_medr, sd_medr = train_utility.evaluate(
        output_description=output_description_test,
        output_scene=output_pov_test,
        section="test")
    # train_utility.save_results(margins, [ds1, ds5, ds10, sd1, sd5, sd10, ndgc_10, ndcg, ds_medr, sd_medr], thres, f'{approach_name}.txt')


def parse_float_tuple(arg):
    try:
        return tuple(float(x) for x in arg.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError("Thresholds and margins must be a comma-separated list of floats.")


def check_args(args):
    if args.custom_margin:
        assert args.status in [0, 1, 2], 'status should be declared and can be either 0,1, or 2 when custom_margin is supposed to being used!!!'
        assert args.margins is not None, 'margins should be declared based on the status when custom_margin is supposed to being used!!!'
        assert args.thresholds is not None, 'thresholds should be declared based on the status when custom_margin is supposed to being used!!!'

        if args.status == 0:
            assert len(args.thresholds) == 1, 'A tuple containing one floating point should be assigned for the thresholds based on the chosen status'
            assert len(args.margins) == 2, 'A tuple containing two floating points should be assigned for the margins based on the chosen status'
        elif args.status == 1:
            assert len(args.thresholds) == 2, 'A tuple containing two floating points should be assigned for the thresholds based on the chosen status'
            assert len(args.margins) == 3, 'A tuple containing three floating points should be assigned for the margins based on the chosen status'
        elif args.status == 2:
            assert len(args.thresholds) == 3, 'A tuple containing three floating points should be assigned for the thresholds based on the chosen status'
            assert len(args.margins) == 4, 'A tuple containing four floating points should be assigned for the margins based on the chosen status'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='which dataset and which model')
    parser.add_argument('-dataset', type=str, default='3dfront', required=True, help='3dfront and museums')
    parser.add_argument('-custom_margin', action='store_true', help='True or False for using the customized margins')
    parser.add_argument('-status', type=int, help='0(1 Threshold, 2 Margins), 1(2 Thresholds, 3 Margins), 2(3 Thresholds, 4 Margins')
    parser.add_argument('-simmodel', type=str, default='distilroberta', help='distilroberta, MiniLM, gte-large')
    parser.add_argument('-simmodel2', type=str, help='distilroberta, MiniLM, gte-large')
    parser.add_argument('-simmodel3', type=str, help='distilroberta, MiniLM, gte-large')
    parser.add_argument('-thresholds', type=parse_float_tuple, help='the thresholds should be a comma-separated list of floats, this should match the status')
    parser.add_argument('-margins', type=parse_float_tuple, help='the margins should be a comma-separated list of floats, this should match the status')
    args = parser.parse_args()

    check_args(args)
    
    start_train(args)
