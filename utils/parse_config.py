import json
import numpy as np
import os


def parse_config(args):
    cwd = os.getcwd()

    # config_file = 'config_vgg16_bn_cifar10.json'
    config_file = f'config_{args.arch}_{args.dataset.lower()}.json'  # args.configs

    # Load setting from args to config file
    session_config = json.load(open(os.path.join(cwd, 'configs', config_file)))
    session_config['Meta']['batchsize'] = args.batchsize
    session_config['Gpu_config']['cuda_visible_devices'] = args.gpu_id
    session_config['Prune_config']['target_sparsity'] = args.target_sparsity
    session_config['Prune_config']['prune_reg'] = args.prune_reg
    session_config['Prune_config']['prune_1st'] = args.prune_1st
    session_config['DDPG']['actor_bounds'] = [args.actor_lbound, args.actor_ubound]
    session_config['Meta']['output_folder'] = '/'.join(['snapshots',
                                                        args.arch,
                                                        session_config['Dataset']['name'].lower(),
                                                        args.prune_reg,
                                                        str(args.target_sparsity) + args.output_name])

    assert session_config['Prune_config']['prune_reg'] != '', 'prune_reg is not defined, please check configuration!'

    out_dir = os.path.join(cwd, session_config['Meta']['output_folder'])
    print(f">> Exporting results in {out_dir}")

    if not os.path.exists(out_dir):
        new_dir = out_dir.split('/')
        out_dir = '/'
        for f in new_dir:
            out_dir = os.path.join(out_dir, f)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

    out_file = os.path.join(out_dir, config_file)
    if os.path.exists(out_file):
        # Cover original json file
        out = open(out_file, 'w')
        out.close()
    with open(out_file, 'a') as out:
        json_dumps = json.dumps(session_config, indent=2)
        print(json_dumps)
        out.write(json_dumps)

    return session_config
