from torch.distributions import Categorical
import numpy as np
import torch
import os
from tqdm.auto import tqdm

from .opt import config_parser
from .dataLoader import dataset_dict
from .renderer import *
from .utils import *

from IPython import embed
import clip
import json, random
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import datetime
import sys
# os.environ['path'] = '/home/fangshuangkang/miniconda3/envs/pix2pix-zero-1/bin/ffmpeg'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class CalLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vgg_grad, clip_grad):
        result = torch.exp(x)
        ctx.save_for_backward([result, vgg_grad, clip_grad])
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, vgg_grad, clip_grad = ctx.saved_tensors
        return grad_output * result


def entropy_loss(p_tensor):
    return Categorical(probs=p_tensor).entropy()


class ClipLoss():
    def __init__(self, w=504, h=378):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device, jit=False)
        self.clip_model.eval()
        self.h, self.w = h, w
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.aug = T.Compose([
            T.RandomCrop((h - 16, w - 16)),
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.aug_eval = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def cal_loss(self, nerf_pred_rgb, ref_text):
        # clip_model.encode_text(clip.tokenize(['a cat', 'a dog']).cuda()) is [2,512]; 'a cat'==['a cat']==[1,512]
        pred_rgb = self.aug_eval(nerf_pred_rgb.view(1, self.h, self.w, 3).permute(0, 3, 1, 2))
        emb_ref_text = self.clip_model.encode_text(clip.tokenize([ref_text]).to(device))
        emb_ref_text = emb_ref_text / emb_ref_text.norm(dim=-1, keepdim=True)
        emb_img = self.clip_model.encode_image(pred_rgb)
        emb_img = emb_img / emb_img.norm(dim=-1, keepdim=True)  # normalize features
        loss_clip = 1 - (emb_img * emb_ref_text).sum(-1).mean()

        return loss_clip


class SimpleSampler:
    def __init__(self, total, batch, sample_one_img=False):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None
        self.sample_one_img = False

    def nextids(self):
        # strategy 1: 对单张图像内的ray进行shuffle, 但要存储下出的rgb值, 一张图像ray过完再计算vgg/clip等loss
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            if self.sample_one_img:
                self.ids = torch.LongTensor(np.arange(self.total))  # 这种完全不shuffle的效果会略差
            else:
                self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    dataset = dataset_dict[args.dataset_name]
    # is_stack 可以让all_rays不展开成[N, 3]的形式，而是保持[num_img, h, w, 3]的形式
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'args': args})
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False, edited_img_folder=args.edited_img_folder, args=args)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, edited_img_folder=args.edited_img_folder, args=args)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
        logfolder = f'{args.basedir}_{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
    if args.color_density_steps:
        train_steps_scheduler = []

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        kwargs.update({'args': args})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
        #  assert not (args.freeze_density and args.color_density_steps)
        if args.freeze_density:
            for n, p in tensorf.named_parameters():
                if 'density' in n:
                    p.requires_grad = False
                print(n, p.shape, p.requires_grad)
        args.upsamp_list = [1e10]
        args.update_AlphaMask_list = [1e10]
        args.N_voxel_init = args.N_voxel_final
        upsamp_list = args.upsamp_list
        update_AlphaMask_list = args.update_AlphaMask_list
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct, args=args)

    grad_vars = tensorf.get_optparam_groups(args.lr_finetune_density, args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    optimizer_density = torch.optim.Adam(grad_vars[:2], betas=(0.9,0.99))
    optimizer_color = torch.optim.Adam(grad_vars[2:4], betas=(0.9,0.99))
    optimizer_others = torch.optim.Adam(grad_vars[4:], betas=(0.9,0.99))
    # optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]
    if args.lr_clip <= 0:
        clip_loss_utils = None
    else:
        clip_loss_utils = ClipLoss()

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    try:
        allrays, allrgbs, oriallrgbs = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.ori_all_rgbs
    except:
        allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
        oriallrgbs = allrgbs.clone()  # FIXME if use no edit dataset(ie ori), it should be changed
    if not args.ndc_ray:
        allrays, allrgbs, oriallrgbs = tensorf.filtering_rays(allrays, allrgbs, oriallrgbs, bbox_only=True)  # FIXME if add other rgb loss, changing this
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size, args.lr_clip > 0)  # batch_size4096

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        if iteration % 100 == 0:
            print(f"\n***********density_plane:{tensorf.state_dict()['density_plane.0'].mean().cpu().item():.4f}, app_plane:{tensorf.state_dict()['app_plane.0'].mean().cpu().item():.4f}*********\n")
        ray_idx = trainingSampler.nextids()
        # rays_train.shape [4096, 6], nSamples=462 for 'flower' scene
        rays_train, rgb_train, ori_rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device), oriallrgbs[ray_idx].to(device)

        #rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)
        loss_depth = torch.mean(depth_map**2)
        loss_ori_rgb = torch.mean((rgb_map - ori_rgb_train) ** 2)
        if iteration > args.n_iters / 2 and args.lr_weight_entropy > 0:
            try:
                loss_weight_entropy = entropy_loss(weights).mean()
            # loss_weight_entropy = entropy_loss(weights / weights.sum(dim=-1).unsqueeze(-1)).mean()  # normlization weights before Entropy
            except:
                if iteration % 20 == 0: print('\n************ loss_weight_entropy field  **************\n')
                loss_weight_entropy = 0
        else:
            loss_weight_entropy = 0
        # 还需要添加 depth平滑TV(loss) + KL(edit_depth, ori_depth); 这里需要注意的是此时的depth_map必须是相邻的ray才行(先修改采ray策略) 
        if args.lr_clip > 0:
            loss_clip = clip_loss_utils.cal_loss(rgb_map, 'a photography of a T.rex dinosaur in the exhibition room')
        else:
            loss_clip = 0
        loss = loss + args.lr_weight_entropy * loss_weight_entropy + args.lr_clip * loss_clip

        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer_color.zero_grad()
        optimizer_density.zero_grad()
        optimizer_others.zero_grad()
        total_loss.backward()
        optimizer_others.step()
        optimizer_color.step()
        optimizer_density.step()
        '''
        # Alternate Training
        if iteration < 300:
            optimizer_color.step()
        elif iteration < 600:
            optimizer_density.step()
        elif iteration < 800:
            optimizer_color.step()
        elif iteration < 1000:
            optimizer_density.step()
        elif iteration < 1100:
            optimizer_color.step()
        elif iteration < 1200:
            optimizer_density.step()
        else:
            optimizer_color.step()
        '''
        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()

        loss = loss.detach().item()

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)

        # for param_group in optimizer.param_groups:
        #    param_group['lr'] = param_group['lr'] * lr_factor
        for param_group in optimizer_color.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor
        for param_group in optimizer_density.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor
        for param_group in optimizer_others.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor


        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            # summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
            '''
            c2ws = test_dataset.render_path
            print('========>',c2ws.shape)
            os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
            evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all_{iteration}/',
                                    N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
            '''

        if iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:  # update volume resolution
                reso_mask = reso_cur
            else:
                reso_mask = [256, 256, 256]
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays, allrgbs, oriallrgbs = tensorf.filtering_rays(allrays, allrgbs, oriallrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size, args.lr_clip > 0)

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            # optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
            optimizer_density = torch.optim.Adam(grad_vars[:2], betas=(0.9, 0.99))
            optimizer_color = torch.optim.Adam(grad_vars[2:4], betas=(0.9, 0.99))
            optimizer_others = torch.optim.Adam(grad_vars[4:], betas=(0.9, 0.99))

    tensorf.save(f'{logfolder}/{args.expname}.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        trans_t = lambda t : torch.Tensor([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,t],
            [0,0,0,1]]).float()

        rot_phi = lambda phi : torch.Tensor([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1]]).float()

        rot_theta = lambda th : torch.Tensor([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1]]).float()

        def pose_rotate_x(theta, angle, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(angle/180.*np.pi) @ c2w  # 将phi参数改为angle
            c2w = rot_theta(theta/180.*np.pi) @ c2w
            c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
            return c2w

        def pose_rotate_z(theta, phi, radius, angle):
            c2w = trans_t(radius)
            c2w = rot_phi(phi/180.*np.pi) @ c2w
            c2w = rot_theta(theta/180.*np.pi) @ c2w
            c2w = torch.Tensor([
                [np.cos(angle/180.*np.pi), -np.sin(angle/180.*np.pi), 0, 0],
                [np.sin(angle/180.*np.pi), np.cos(angle/180.*np.pi), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]) @ c2w
            return c2w

        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi/180.*np.pi) @ c2w
            c2w = rot_theta(theta/180.*np.pi) @ c2w
            blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            blender2opencv = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            blender2opencv = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            blender2opencv = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            blender2opencv = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            blender2opencv = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            c2w = torch.Tensor(c2w @ torch.Tensor(blender2opencv))
            # c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
            return c2w

        c2ws = test_dataset.render_path
        # c2ws = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        # c2ws = torch.stack([pose_rotate_x(30.0, angle, 4.0) for angle in range(0, 360, 10)], 0)
        # render_poses = torch.stack([pose_rotate_z(30.0, -30.0, 4.0, angle) for angle in range(0, 360, 10)], 0)
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)
    
    '''For CE3D editing results'''
    args.basedir = '/data/llff/flower/work_space/editing_space'
    args.datadir  = '/data/llff/flower'
    args.edited_img_folder = '/data/llff/flower/work_space/editing_space/frames'
    args.expname = 'test_TRF'
    args.no_dir = True

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

