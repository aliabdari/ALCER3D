import argparse
import os
import pickle
from tqdm import tqdm


def prepare(args):
    pov_images = []
    if args.dataset == '3dfront':
        path = '3dfront'
        #     images
        for i in tqdm(range(2)):
            pov_images.extend(pickle.load(open(path + os.sep + f'pov_images_3dfront_part{i}.pkl', 'rb')))

        pickle_file = open(f'{path}/pov_images_3dfront.pkl', 'wb')
        pickle.dump(pov_images, pickle_file)
    elif args.dataset == 'museums':
        descs = []
        path = 'museums'
        for i in tqdm(range(3)):
            pov_images.extend(pickle.load(open(path + os.sep + f'pov_images_museums_part{i}.pkl', 'rb')))
        pickle_file = open(f'{path}/pov_images_museum.pkl', 'wb')
        pickle.dump(pov_images, pickle_file)
        for i in tqdm(range(6)):
            descs.extend(pickle.load(open(path + os.sep + f'descs_museums_part{i}.pkl', 'rb')))
        pickle_file = open(f'{path}/descs_museum.pkl', 'wb')
        pickle.dump(descs, pickle_file)
    else:
        print('dataset is not defined')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='which dataset')
    parser.add_argument('-dataset', type=str, required=True,
                        help='3dfront and museums')
    prepare(args=parser.parse_args())
