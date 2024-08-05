# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import os
import math
import argparse
import numpy as np
from copy import deepcopy

from lib.utils.utils import prYellow
from lib.env.quantize_env import QuantizeEnv
from lib.rl.ddpg import DDPG
from tensorboardX import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import models as customized_models

import warnings
import pretrained_models
warnings.filterwarnings('ignore')
# Models

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__") and not name.startswith("VGG") and not name.startswith("vgg")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]
"""
pretrained_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))
for name in pretrainedmodels.__dict__:
    if name.islower() and not name.startswith("__") and callable(pretrainedmodels.__dict__[name]):
        models.__dict__[name] = pretrainedmodels.__dict__[name]
"""
model_names = default_model_names + customized_models_names
#print('support models: ', model_names)

def export_model_action(env, args):
    """
    exprot model to saved path, use this when actor/critic is trained
    final strategy ratio need to provide
    :param env: pruning environment
    :param args: user given input
    """
    assert args.ratios is not None or args.channels is not None, 'Please provide a valid ratio list or pruned channels'

    env.set_export_path()

    print('=> Original model channels: {}'.format(env.org_channels))
    if args.ratios:
        with open(args.ratios, 'r') as f:
            data = f.read()
        model_actions = data.split('\n')
        final_plan = []
        for layer_actions in model_actions:
            ratios = layer_actions.split(',')

            def convert_scfun(str_num):
                if 'e' not in str_num:
                    return float(str_num)
                before_e = float(str_num.split('e')[0])
                sign = str_num.split('e')[1][:1]
                after_e = int(str_num.split('e')[1][1:])

                if sign == '+':
                    float_num = before_e * math.pow(10, after_e)
                elif sign == '-':
                    float_num = before_e * math.pow(10, -after_e)
                else:
                    float_num = None
                    print('error: unknown sign')
                return float_num

            layer_ratios = [convert_scfun(action) for action in ratios]
            final_plan.append(layer_ratios)
    else:
        NotImplemented
    print('=> Pruning with ratios: {}'.format(ratios))

    for r in ratios:
        env.step_cm_no_adjust(r)

    return

def export_model_actions(env, args):
    """
    exprot model to saved path, use this when actor/critic is trained
    final strategy ratio need to provide
    :param env: pruning environment
    :param args: user given input
    """
    assert args.ratios is not None, 'Please provide a valid ratio list or pruned channels'

    env.set_export_path()

    print('=> Original model channels: {}'.format(env.org_channels))
    if args.ratios:
        with open(args.ratios, 'r') as f:
            data = f.read()
        model_actions = data.split('\n')
        export_actions = []
        for layer_actions in model_actions:
            if len(layer_actions) < 9:
                continue
            else:
                export_actions.append([float(action) for action in layer_actions.split(',')])

        assert len(export_actions) == len(env.prunable_idx)

    for r in export_actions:
        env.step_cm_no_adjust(r)

    return

def train(num_episode, agent, env, output, debug=False):
    # best record
    best_reward = -math.inf
    best_policy = []

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    # get layer importance and store it in json file
    # importance_json = env._extract_filter_importance()
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        # action range in [0., 1.]
        if episode <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step_cm_no_adjust(action, None)#step_cm(action, None)
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # [optional] save intermideate model
        if int(num_episode / 10) > 0 and episode % int(num_episode / 10) == 0:
            # save actor and critic
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        observation = deepcopy(observation2)

        if done:  # end of episode
            episode_reward = reward
            if debug:
                print('#{}: episode_reward:{:.4f} acc: {:.4f}, weight: {:.4f} , flops:{:.4f}'.format(episode, episode_reward,
                                                                                         info['accuracy'],
                                                                                         info['w_ratio'],
                                                                                         info['FLOPS_ratio']))
            text_writer.write(
                '#{}: episode_reward:{:.4f} acc: {:.4f}, weight: {:.4f} MB\n'.format(episode, episode_reward,
                                                                                     info['accuracy'],
                                                                                     info['w_ratio'],
                                                                                     info['FLOPS_ratio']))
            final_reward = T[-1][0]
            # agent observe and update policy
            for i, (r_t, s_t, s_t1, a_t, done) in enumerate(T):
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    # todo: AMC don't have this condition
                    for i in range(args.n_update):
                        agent.update_policy()
                        # print('{}/{}:value_loss:{:.4f}, policy_loss:{:.4f}\n'.format(i, episode, agent.value_loss, agent.policy_loss))

            agent.memory.append(
                observation,
                agent.select_action(observation, episode=episode),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            if final_reward > best_reward:
                best_reward = final_reward
                best_policy = env.strategy

            value_loss = agent.get_value_loss()
            policy_loss = agent.get_policy_loss()
            # delta = agent.get_delta()
            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', best_reward, episode)
            tfwriter.add_scalar('info/accuracy', info['accuracy'], episode)
            tfwriter.add_scalar('info/w_ratio', info['w_ratio'], episode)
            tfwriter.add_text('info/best_policy', str(best_policy), episode)
            tfwriter.add_text('info/current_policy', str(env.strategy), episode)
            tfwriter.add_scalar('value_loss', value_loss, episode)
            tfwriter.add_scalar('policy_loss', policy_loss, episode)
            # tfwriter.add_scalar('delta', delta, episode)


            text_writer.write('best reward: {}\n'.format(best_reward))
            text_writer.write('best policy: {}\n'.format(best_policy))
    text_writer.close()
    return best_policy, best_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning')
    parser.add_argument('--job', default='train', type=str, help='support option: train/export')
    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    # check point for loading customized model
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')

    #quantify options
    parser.add_argument('--quantify_opt', default='fix', type=str, help='support option: fix/mix')
    parser.add_argument('--iou_thresh', default=0.3, type=float, help='iou_thresh for object detection')
    #detection

    # env
    #todo: add pan/ ir to dataset
    #dataset is the dataset name,e.g. imagenet
    #dataset_root is the folder to store dataset (only 1) If train/val given, change here
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use)')
    parser.add_argument('--dataset_root', default='../dataset/imagenet/', type=str, help='path to dataset)')
    parser.add_argument('--preserve_ratio', default=0.2, type=float, help='preserve ratio of the model size')
    # add those 2 bound to ensure channel pruning works

    parser.add_argument('--quantize_bit', default=8, type=float, help='minimum bit to use')
    parser.add_argument('--float_bit', default=32, type=int, help='the bit of full precision float')
    parser.add_argument('--alpha', default=0.1, type=float, help='maximum KL divergence caused by pruning and quantization')

    parser.add_argument('--is_pruned', dest='is_pruned', action='store_true')
    # only for channel pruning todo: 0.7 to 0.9 for debug
    parser.add_argument('--lbound', default=0.01, type=float, help='minimum pruning ratio')
    parser.add_argument('--rbound', default=0.4, type=float, help='maximum pruning ratio')
    parser.add_argument('--classifier_preserve_ratio', default=0.1, type=float, help='perserve ratio for classifier') #todo: 0.7 to 0.8
    parser.add_argument('--n_calibration_batches', default=32, type=int,
                        help='n_calibration_batches')# 60 -> 1

    parser.add_argument('--n_points_per_layer', default=10, type=int,
                        help='method to prune (fg/cp for fine-grained and channel pruning)')
    parser.add_argument('--channel_round', default=2, type=int, help='Round channel to multiple of channel_round')
    # ddpg
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=25, type=int,
                        help='time without training but only filling the replay memory') # 100 -> 1
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')# sample size for DDPG 64 -> 32, next time make it larger
    parser.add_argument('--rmsize', default=100, type=int, help='memory size for each layer') # change from 128 to 100
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.95, type=float,
                        help='delta decay during exploration')
    parser.add_argument('--n_update', default=1, type=int, help='number of rl to update each time')
    # training
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='../../save', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=100, type=int, help='train iters each timestep') # change from 600 to 500[450]
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=234, type=int, help='') # from 234 to none
    parser.add_argument('--n_worker', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=60, type=int, help='number of data per batch size')# 60 -> 16
    parser.add_argument('--data_sample_size', default=16, type=int, help='number of data batch size')  # 60 -> 16
    parser.add_argument('--finetune_epoch', default=1, type=int, help='')
    parser.add_argument('--finetune_gamma', default=0.8, type=float, help='finetune gamma')
    parser.add_argument('--finetune_lr', default=0.001, type=float, help='finetune gamma')
    parser.add_argument('--finetune_flag', default=5, type=bool, help='whether to finetune')
    parser.add_argument('--matrix', default='AP', type=str, help='matrix in reward, e.g.: top-5 or AP')
    parser.add_argument('--mode', default='PQA', type=str, help='AMC_mode in PQA, AMC, HAQ')
    parser.add_argument('--train_size', default=2000, type=int, help='number of train data size')
    parser.add_argument('--val_size', default=1000, type=int, help='number of val data size')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='mobilenet_v2', choices=model_names,
                    help='model architecture:' + ' | '.join(model_names) + ' (default: mobilenet_v2)')
    # export
    parser.add_argument('--ratios', default=None, type=str, help='ratios for pruning')
    parser.add_argument('--channels', default=None, type=str, help='channels after pruning')
    #parser.add_argument('--export_path', default=None, type=str, help='path for exporting models')
    parser.add_argument('--use_new_input', dest='use_new_input', action='store_true', help='use new input feature')


    parser.add_argument('--learn_epochs', default=15, type=int, help='')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_type', default='cos', type=str,
                        help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--warmup_epoch', default=0, type=int, metavar='N',
                        help='manual warmup epoch number (useful on restarts)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--finetune_path', default='./finetunes/mobilenet_0.2model_size_export.pth.tar', type=str, help='path for exporting models')
    parser.add_argument('--schedule', type=int, nargs='+', default=[31, 61, 91],
                        help='Decrease learning rate at these epochs.')
    # device options
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    base_folder_name = '{}_{}'.format(args.arch, args.dataset)
    if args.suffix is not None:
        base_folder_name = base_folder_name + '_' + args.suffix
    # log all operation
    args.output = os.path.join(args.output, base_folder_name)
    tfwriter = SummaryWriter(logdir=args.output)
    text_writer = open(os.path.join(args.output, 'log.txt'), 'w')


    print('==> Output path: {}...'.format(args.output))

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    assert torch.cuda.is_available(), 'CUDA is needed for CNN'

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # will download weight from pytorch website
    model = models.__dict__[args.arch](pretrained=True)
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        # load pretrained model from pytorch
        # enable data parallel in module layer
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    # model.state_dict() returns a dictionary containing a whole state of the module
    pretrained_model = model.state_dict()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    cudnn.benchmark = True
    env = QuantizeEnv(model, pretrained_model, args.dataset, args.dataset_root,
                      compress_ratio=args.preserve_ratio, n_data_worker=args.n_worker,
                      batch_size=args.data_bsize, args=args, float_bit=args.float_bit, quantize_bit = args.quantize_bit,
                      is_model_pruned=args.is_pruned,
                      export_model=args.job == 'export')
    if args.job == 'train':
        # get number of states for an observation
        nb_states = env.layer_embedding.shape[1]
        # number of action
        nb_actions = 9  # actions for weight and activation quantization
        # repaly memory size
        args.rmsize = args.rmsize * len(env.quantifiable_idx)  # for each layer
        print('** Actual replay buffer size: {}'.format(args.rmsize))
        """
        from xlwt import Workbook

        wb = Workbook()
        sheet1 = wb.add_sheet('MIX')
        sheet1.write(0, 0, 'reward')
        len_prun = len(env.prunable_idx)
        for i in range(len_prun):
            sheet1.write(0,i+1,'idx'+str(i))
        for i in range(len_prun):
            sheet1.write(0, len_prun+i, 'sp'+str(i))
        wb.save(args.arch+'temp.xls')
        """
        # init actor and critic network
        agent = DDPG(nb_states, nb_actions, args)
        best_policy, best_reward = train(args.train_episode, agent, env, args.output, debug=args.debug)
        print('best_reward: ', best_reward)
        print('best_policy: ', best_policy)
    elif args.job == 'export':
        export_model_actions(env, args)
    else:
        raise RuntimeError('Undefined job {}'.format(args.job))



