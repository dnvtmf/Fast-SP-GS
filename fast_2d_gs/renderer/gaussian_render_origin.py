"""
paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023
code: https://github.com/graphdeco-inria/gaussian-splatting
"""
import math
from typing import Tuple, Union, List, Dict
import torch
from torch import Tensor
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from utils import ops_3d


def render_gs_official(
    points: Tensor,
    opacity: Tensor,
    raster_settings: GaussianRasterizationSettings = None,
    scales: Tensor = None,
    rotations: Tensor = None,
    covariance: Tensor = None,
    sh_features: Tensor = None,
    sh_features_rest: Tensor = None,
    colors=None,
    extras=None,
    **kwargs
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    if raster_settings is None:
        size = torch.as_tensor(kwargs['size'])
        if kwargs.get('focals', None) is not None:
            tan_FoV = 0.5 * size.to(kwargs['focals'].device) / kwargs['focals']
        else:
            tan_FoV = torch.tan(kwargs['FoV'] * 0.5)
        Tw2v = kwargs['Tw2v']
        if 'Tv2c' in kwargs:
            Tw2c = kwargs['Tv2c'] @ Tw2v
        else:
            Tw2c = kwargs['Tw2c']
        raster_settings = GaussianRasterizationSettings(
            image_width=size[0].item(),
            image_height=size[1].item(),
            tanfovx=tan_FoV[0].item(),
            tanfovy=tan_FoV[1].item(),
            scale_modifier=kwargs.get('scale_modifier', 1.),
            viewmatrix=Tw2v.transpose(-1, -2),
            projmatrix=Tw2c.transpose(-1, -2),
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
    # assert extras is None and len(kwargs) == 0, f"Not supported"
    if rotations is not None:
        rotations = rotations[..., (3, 0, 1, 2)]  # (x, y, z w) -> (w, x, y, z)
    if sh_features_rest is not None:
        sh_features = torch.cat([sh_features, sh_features_rest], dim=1)
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    outputs = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=sh_features,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=covariance)
    if len(outputs) == 4:
        rendered_image, radii, depth, alpha = outputs
    elif len(outputs) == 3:
        (rendered_image, radii, depth), alpha = outputs, None
    else:
        (rendered_image, radii), depth, alpha = outputs, None, None
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "images": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depths": depth,
        "alpha": alpha,
    }


def render_gs_official_batch(
    size: Union[Tuple[int, int], List[Tuple[int, int]], Tensor],
    Tw2v: Tensor,
    Tv2c: Tensor,
    FoV: Tensor,
    points: Tensor,
    opacity: Tensor,
    campos: Tensor = None,
    scales: Tensor = None,
    rotations: Tensor = None,
    covariance: Tensor = None,
    sh_features: Tensor = None,
    colors=None,
    extras=None,
    bg: Tensor = None,
    sh_degree: int = 0,
    **kwargs
):
    """
    Render the scene, and get multi images.
    """
    assert extras is None
    shape = torch.broadcast_shapes(Tw2v.shape[:-2], Tv2c.shape[:-2], FoV.shape[:-1])
    if len(shape) == 0:
        return render_gs_official(points, opacity, None, scales, rotations, covariance, sh_features, colors, None,
                                  size=size, FoV=FoV, Tw2v=Tw2v, Tv2c=Tv2c, campos=campos, sh_degree=sh_degree, **kwargs)
    Tw2v = Tw2v.expand(*shape, 4, 4).view(-1, 4, 4)
    Tw2c = Tv2c.expand(*shape, 4, 4).view(-1, 4, 4) @ Tw2v
    FoV = FoV.expand(*shape, 2).view(-1, 2)
    campos = ops_3d.rigid.inverse(Tw2v)[:, :3, 3] if campos is None else campos.expand(*shape, 3)
    if isinstance(size, Tensor):
        size = size.expand(*shape, 2)
    bg = points.new_zeros(*shape, 3) if bg is None else bg.expand(*shape, 3)
    num_batch = shape.numel()

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = points.new_zeros(num_batch, *points.shape, requires_grad=True) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    if rotations is not None:
        rotations = rotations[..., (3, 0, 1, 2)]  # (x, y, z w) -> (w, x, y, z)

    images, radiis, depths, alphas = [], [], [], []
    for b in range(num_batch):
        if isinstance(size, Tensor):
            size_b = size[b].tolist()
        elif isinstance(size[0], int):
            size_b = size
        else:
            size_b = size[b]
        raster_settings = GaussianRasterizationSettings(
            image_width=size_b[0],
            image_height=size_b[1],
            tanfovx=math.tan(0.5 * FoV[b, 0].item()),
            tanfovy=math.tan(0.5 * FoV[b, 1].item()),
            scale_modifier=kwargs.get('scale_modifier', 1.),
            viewmatrix=Tw2v[b].transpose(-1, -2),
            projmatrix=Tw2c[b].transpose(-1, -2),
            sh_degree=sh_degree,
            campos=campos[b],
            prefiltered=False,
            debug=False,
            bg=bg[b]
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = points
        means2D = screenspace_points[b]
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        outputs_b = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=sh_features,
            colors_precomp=colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=covariance
        )
        images.append(outputs_b[0])
        radiis.append(outputs_b[1])
        if len(outputs_b) > 2:
            depths.append(outputs_b[2])
        if len(outputs_b) > 3:
            depths.append(outputs_b[3])

    radiis = torch.stack(radiis).reshape(*shape, -1, 1)
    screenspace_points.view(*shape, *points.shape)
    outputs: Dict[str, Union[Tensor, List[Tensor]]] = {
        "viewspace_points": screenspace_points,
        "visibility_filter": radiis > 0,
        "radii": radiis,
    }
    if all(img.shape == images[0].shape for img in images):
        C, H, W = images[0].shape
        outputs['images'] = torch.stack(images).reshape(*shape, C, H, W)
        if len(depths) > 0:
            outputs['depths'] = torch.stack(depths).reshape(*shape, H, W)
        if len(alphas) > 0:
            outputs['alpha'] = torch.stack(alphas).reshape(*shape, H, W)
    else:
        outputs['images'] = images
        if len(depths) > 0:
            outputs['depths'] = depths
        if len(alphas) > 0:
            outputs['alpha'] = alphas
    return outputs
