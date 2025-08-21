import math
import torch
from torch import Tensor
from utils import ops_3d

try:
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    GaussianRasterizationSettings, GaussianRasterizer = None, None


def depths_to_points(Tw2v, Tw2c, size, depthmap):
    c2w = Tw2v.T.inverse()
    W, H = size
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ Tw2c
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(),
                                    indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(Tw2v, Tw2c, size, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(Tw2v, Tw2c, size, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def render_2d_gs_offical(points: Tensor, opacity: Tensor, scales: Tensor = None, rotations: Tensor = None,
                         sh_features: Tensor = None, sh_features_rest: Tensor = None, colors=None,
                         covariance: Tensor = None, depth_ratio=1.,
                         raster_settings: GaussianRasterizationSettings = None, **kwargs):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    if raster_settings is None:
        size = kwargs['size']
        if kwargs.get('focals', None) is not None:
            tan_FoV = 0.5 * torch.as_tensor(size, device=points.device) / kwargs['focals']
        else:
            tan_FoV = torch.tan(kwargs['FoV'] * 0.5)
        Tw2v = kwargs['Tw2v']
        if 'Tv2c' in kwargs:
            Tw2c = kwargs['Tv2c'] @ Tw2v
        else:
            Tw2c = kwargs['Tw2c']
        raster_settings = GaussianRasterizationSettings(
            image_width=size[0],
            image_height=size[1],
            tanfovx=tan_FoV[0].item(),
            tanfovy=tan_FoV[1].item(),
            scale_modifier=kwargs.get('scale_modifier', 1.),
            viewmatrix=Tw2v.transpose(-1, -2).contiguous(),
            projmatrix=Tw2c.transpose(-1, -2).contiguous(),
            sh_degree=kwargs['sh_degree'],
            campos=kwargs['campos'] if 'campos' in kwargs else ops_3d.rigid.inverse(Tw2v)[:3, 3],
            prefiltered=False,
            debug=False,
            bg=kwargs.get('bg', points.new_zeros(3))
        )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(points, requires_grad=True) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = points
    means2D = screenspace_points
    if rotations is not None:
        rotations = rotations[..., (3, 0, 1, 2)]  # (x, y, z w) -> (w, x, y, z)
    if sh_features_rest is not None:
        sh_features = torch.cat([sh_features, sh_features_rest], dim=1)
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=sh_features,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=covariance)
    outputs = {
        "images": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    # additional regularization
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1, 2, 0) @ (raster_settings.viewmatrix[:3, :3].T)).permute(2, 0, 1)

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]

    # pseudo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk Anti-Aliasing.
    surf_depth = render_depth_expected * (1 - depth_ratio) + depth_ratio * render_depth_median

    # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
    surf_normal = depth_to_normal(
        raster_settings.viewmatrix, raster_settings.projmatrix,
        (raster_settings.image_width, raster_settings.image_height),
        surf_depth
    )
    surf_normal = surf_normal.permute(2, 0, 1)
    # remember to multiply with accum_alpha since render_normal is normalized.
    surf_normal = surf_normal * render_alpha.detach()

    outputs.update({
        'opacity': render_alpha.squeeze(0),  # shape: [H, W]
        'normals': render_normal,
        'distortion': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
    })

    return outputs
