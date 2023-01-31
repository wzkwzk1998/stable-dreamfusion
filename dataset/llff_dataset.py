import torch
import numpy as np
import os, imageio
from torch.utils.data.dataset import Dataset 
from torch.utils.data import DataLoader
import sys
import os
print(os.getcwd)
sys.path.append(os.getcwd())
from dataset.base_dataset import BaseDataset
########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')] 
    # the last
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    


def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    

    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    # bounding_box = get_bbox3d_for_llff(poses[:,:3,:4], poses[0,:3,-1], near=0.0, far=1.0)

    return images, poses, bds, render_poses, i_test 



class LlffDataset(BaseDataset):
    
    def __init__(self, basedir, device, split='train', factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, N_rand=-1, 
                H=360, W=504, ndc:bool=True, llffhold=8):
        self.images, self.poses, self.bds, self.render_poses, self.i_test = \
            load_llff_data(basedir, factor=factor, recenter=recenter, bd_factor=bd_factor, spherify=spherify, path_zflat=path_zflat)

        self.bounding_box = self.get_bbox3d_for_llff(self.poses[:,:3,:4], self.poses[0,:3,-1], near=0.0, far=1.0) 
        self.split = split
        self.N_rand = N_rand
        self.images = torch.Tensor(self.images)
        self.device = device

        if llffhold > 0:
            # log('[INFO] Auto LLFF holdout,', llffhold)
            self.i_test = np.arange(self.images.shape[0])[::llffhold]
        self.i_val = self.i_test
        self.i_train = np.array([i for i in np.arange(int(self.images.shape[0])) if
                        (i not in self.i_test and i not in self.i_val)])

        # get H and W
        H_origin, W_origin, focal_origin = self.poses[0, :3, -1]
        assert W_origin / W == H_origin / H, "scale in W and H must be the same"
        scale = W_origin / W
        self.H = int(H)
        self.W = int(W)
        self.focal = focal_origin / scale

        # which data to save
        if self.split == 'train':
            self.poses = self.poses[self.i_train, :3, :4]
            self.images = self.images[self.i_train]
        elif self.split == 'test':
            self.poses = self.poses[self.i_test, :3, :4]
            self.images = self.images[self.i_test]
        self.video_poses = self.render_poses[:, :3, :4]
        self.video_length = len(self.video_poses)

        self.K = np.array([
                    [self.focal, 0, 0.5*self.W],
                    [0, self.focal, 0.5*self.H],
                    [0, 0, 1]
                ])
        
        if not ndc:
            self.near = np.ndarray.min(self.bds) * .9
            self.far = np.ndarray.max(self.bds) * 1.
        else:
            self.near = 0.
            self.far = 1.

    
    def __len__(self):
        return len(self.poses)
        

    def __getitem__(self, index):
        image = self.images[index]
        pose = self.poses[index] 
        H = int(self.H)
        W = int(self.W)
        focal = self.focal
        rays_o, rays_d = self.get_rays(H, W, focal, torch.Tensor(pose))
        if self.split == 'train':
            if self.N_rand is not None and self.N_rand > 0:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                image = image[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)   

        rays_o, rays_d = self.ndc_rays(H, W, focal, 1., rays_o, rays_d)
        rays_o = torch.reshape(rays_o, (-1, 3))
        rays_d = torch.reshape(rays_d, (-1, 3))
        near = self.near * torch.ones(list(rays_d.shape[:-1]) + [1])
        far = self.far * torch.ones(list(rays_d.shape[:-1]) + [1])
        
        data_dict = {
            'rgb_gt' : image.to(self.device),
            'rays_o' : rays_o.to(self.device),
            'rays_d' : rays_d.to(self.device),
            'H' : H,
            'W' : W,
            'nears' : near.to(self.device),
            'fars' : far.to(self.device), 
        }
        
        return data_dict

        
    def get_random_data(self):
        idx = np.random.randint(0, len(self))
        image = self.images[idx]
        pose = self.poses[idx] 
        rays_o, rays_d = self.get_rays(H, W, focal, torch.Tensor(pose))
        if self.split == 'train':
            if self.N_rand is not None and self.N_rand > 0:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                image = image[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)   
        
        near = self.near * torch.ones(list(rays_d.shape[:-1]) + [1])
        far = self.far * torch.ones(list(rays_d.shape[:-1]) + [1])
        rays_o, rays_d = self.ndc_rays(H, W, focal, 1., rays_o, rays_d)

        data_dict = {
            'rgb_gt' : torch.Tensor(image),
            'rays_o' : torch.Tensor(rays_o),
            'rays_d' : torch.Tensor(rays_d),
            'near' : near,
            'far' : far,
        }
        
        return data_dict

    def get_video_data(self, idx):
        pose = self.video_poses[idx]  
        H = int(self.H // self.training_scale)
        W = int(self.W // self.training_scale)
        focal = self.focal / self.training_scale
        rays_o, rays_d = self.get_rays(H, W, focal, torch.Tensor(pose))
        
        near = self.near * torch.ones(list(rays_d.shape[:-1]) + [1])
        far = self.far * torch.ones(list(rays_d.shape[:-1]) + [1])
        rays_o, rays_d = self.ndc_rays(H, W, focal, 1., rays_o, rays_d)

        data_dict = {
            'rays_o' : rays_o,
            'rays_d' : rays_d,
            'near' : near,
            'far' : far,
        }
        
        return data_dict
    

    def get_video_length(self):
        return self.video_length


    def get_bbox3d_for_llff(self, poses, hwf, near=0.0, far=1.0):
        H, W, focal = hwf
        H, W = int(H), int(W)
        
        # ray directions in camera coordinates
        min_bound = [100, 100, 100]
        max_bound = [-100, -100, -100]

        points = []
        poses = torch.FloatTensor(poses)
        print(f'poses.shape {poses.shape}')
        for pose in poses:
            rays_o, rays_d = self.get_rays(H, W, focal, torch.Tensor(pose))
            rays_o, rays_d = self.ndc_rays(H, W, focal, 1.0, rays_o, rays_d)
            rays_o = torch.reshape(rays_o, (-1, 3))
            rays_d = torch.reshape(rays_d, (-1, 3))

            def find_min_max(pt):
                for i in range(3):
                    if(min_bound[i] > pt[i]):
                        min_bound[i] = pt[i]
                    if(max_bound[i] < pt[i]):
                        max_bound[i] = pt[i]
                return

            for i in [0, W-1, H*W-W, H*W-1]:
                min_point = rays_o[i] + near*rays_d[i]
                max_point = rays_o[i] + far*rays_d[i]
                points += [min_point, max_point]
                find_min_max(min_point)
                find_min_max(max_point)

        return (torch.tensor(min_bound)-torch.tensor([0.1,0.1,0.0001]), torch.tensor(max_bound)+torch.tensor([0.1,0.1,0.0001]))

    def dataloader(self):
        loader = DataLoader(self, batch_size=1, shuffle=(self.split=='train'), num_workers=0)
        return loader


if __name__ == '__main__':
    seed_everything(0)
    data_dir = '/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/flower'
    dataset = LlffDataset(data_dir, 'cpu', N_rand=1024)
    print(dataset.__getitem__(0))