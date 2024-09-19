import configargparse


def config_parser(cmd=None, exist_parser=None):
    if exist_parser is None:
        parser = configargparse.ArgumentParser()
    else:
        parser = exist_parser
    parser.add_argument('--config', is_config_file=True, help='config file path')

    parser.add_argument("--color_density_steps", type=str, default='[(500, 500), (200, 200), (100, 100)]',
                        help='alternating training the color and density according to the given steps')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--edit_name", type=str, default='')
    parser.add_argument("--freeze_density", action="store_true")
    parser.add_argument("--loss_rate_ori_rgb", type=float, default=0.)
    parser.add_argument("--lr_weight_entropy", type=float, default=5e-3)
    parser.add_argument("--lr_clip", type=float, default=0)
    parser.add_argument("--lr_finetune_density", type=float, default=0.01)
    parser.add_argument("--no_dir", action="store_true")
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')

    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=8.0)
    parser.add_argument('--downsample_test', type=float, default=8.0)

    parser.add_argument('--model_name', type=str, default='TensorVMSplit',
                        choices=['TensorVMSplit', 'TensorCP'])

    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=20000)

    parser.add_argument('--dataset_name', type=str, default='llff',
                        choices=['blender', 'llff', 'nsvf', 'dtu','tankstemple', 'own_data', 'fang'])


    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02,
                        help='learning rate')    
    parser.add_argument("--lr_basis", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')

    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=1.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=1.0,
                        help='loss weight')
    
    # model
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append", default=[16, 4, 4])
    parser.add_argument("--n_lamb_sh", type=int, action="append", default=[48, 12, 12])
    parser.add_argument("--data_dim_color", type=int, default=27)

    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
                        
    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_Fea",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=0,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=0,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')

    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=1)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=1)
    parser.add_argument("--export_mesh", type=int, default=0)

    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='relu')
    parser.add_argument('--ndc_ray', type=int, default=1)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    parser.add_argument('--N_voxel_init',
                        type=int,
                        default=128**3)
    parser.add_argument('--N_voxel_final',
                        type=int,
                        default=512**3)
    parser.add_argument("--upsamp_list", type=int, action="append", default=[2000, 3000, 4000, 5500])
    parser.add_argument("--update_AlphaMask_list", type=int, action="append", default=[2500])

    parser.add_argument('--idx_view',
                        type=int,
                        default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=4,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000,
                        help='frequency of visualize the image')
    if exist_parser is not None:
        return parser

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
