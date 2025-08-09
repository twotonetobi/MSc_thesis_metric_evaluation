import Sources.utils.logger as logger
from Sources.utils.functional import load_data
from Sources.utils.util import dotdict
import argparse
import numpy as np
from Sources.Evaluation.evaluator import LightEvaluator
import Sources.utils.util as util
import os
# to evaluate a dataset by itself to get insights in evaluation scores

def main():
    """ Main function """
    np.random.seed(1234)

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which GPU to use, CPU has value -1')

    parser.add_argument('--dataset', type=str, required=True,
                        help='name of dataset')

    parser.add_argument('--dataset_dir', type=str, default='../../Data/Input/LearningDataSets',
                        help='directory of datasets')

    parser.add_argument('--base_gen', type=str, default='',
                        help='what generator to use as base for all values (zero,random, onset, novelty)')

    parser.add_argument('--dim_gen', type=str,
                        default='',
                        help='what dimmer generator to use (random, onbeat, rms, onset)')

    parser.add_argument('--hue_gen', type=str,
                        default='',
                        help='what hue generator to use (random, onset)')

    parser.add_argument('--sat_gen', type=str,
                        default='',
                        help='what sat generator to use (random, onset)')

    parser.add_argument('--pan_gen', type=str,
                        default='',
                        help='what pan generator to use (random, onset)')

    parser.add_argument('--tilt_gen', type=str,
                        default='',
                        help='what tilt generator to use (random, onset)')

    base_args = parser.parse_args()

    base_args.dataset_path = os.path.join(base_args.dataset_dir, base_args.dataset)

    train_data, test_data, val_data, dataset_config = load_data(base_args)

    data = np.concatenate((train_data, test_data, val_data))

    util.add_all_audio_features(data, dataset_config)

    d = {}
    config = dotdict(d)
    config.prepro_config = dataset_config

    logger.init(None)

    evaluator = LightEvaluator(None, None, config, base_args, is_dataset_eval=True)

    evaluator.set_test_data(data)

    evaluator.plot_distribution_and_metric_plots()

    logger.log('Starting Evaluation')



if __name__ == '__main__':
    main()