import argparse
import math
import time
import os

import torch
import torch.nn.functional as F
import pyexr
from PIL import Image

from modules import Siren

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def gradient(p, sdf, method='tetrahedron'):
  if method == 'autodiff':
    # autodiff is slower
    y, x = sdf(p, True)
    return torch.autograd.grad(y, x, torch.ones_like(y))[0]
  elif method == 'tetrahedron':
    e = torch.tensor([1., -1.]) * 1e-3
    offsets = torch.tensor([
        [e[0], e[1], e[1]],
        [e[1], e[1], e[0]],
        [e[1], e[0], e[1]],
        [e[0], e[0], e[0]]
    ])
    p = p.unsqueeze(1)
    d = sdf(p + offsets)
    n = (offsets * d.unsqueeze(-1)).sum(dim=1)
    return F.normalize(n)
  else:
    assert False

def cast_shadow_ray(ro, rd, k, sdf):
  B = ro.shape[0]
  sha = torch.ones(B)
  t = torch.full((B,), 0.01)
  for _ in range(100):
    p = ro + rd * t[:, None]
    d = sdf(p)
    sha = torch.min(sha, k * F.relu(d) / t)
    # if ((torch.abs(d) < 1e-3) | (p[:,1] > 3.)).all():
    #   break
    t += torch.clamp(d, 0.005, 0.05)
    # t += d
  return sha

def cast_occlusion_ray(ro, rd, sdf):
  B = ro.shape[0]
  occ = torch.zeros(B)
  falloff = 1.
  for i in range(5):
    t = 0.01 + 0.1 * i / 4
    d = sdf(ro + rd * t)
    occ += (t - d) * falloff
    falloff *= 0.95
  return torch.clamp(1 - 2 * occ, 0, 1)

def shade_image(sample_coord, pixel_coord, res, sdf,
                max_iter=50, max_dis=10, cam_angle=0.0, cam_height=2.2, cam_target=[0, 0, 0],
                key_light=[1, 1.5, 1.4], ground=-1):
  B = pixel_coord.shape[0]

  uv = sample_coord / res
  uv = uv.flip(1)
  uv -= 0.5
  uv[:,0] *= res[1] / res[0]

  # out = torch.zeros((res[0], res[1], 3))
  # out[res[0] - 1 - frag_coord[:, 0], frag_coord[:, 1]] = torch.cat(
  #     (uv, torch.zeros((B, 1))), dim=1
  # )
  # return out

  r = 3.
  an = 2. * math.pi * cam_angle
  ro = torch.tensor([r * math.cos(an), cam_height, r * math.sin(an)]) # camera pos
  ta = torch.tensor(cam_target) # target
  # camera matrix: mat3(uu, vv, ww)
  up = torch.tensor([0., 1., 0.])
  ww = F.normalize(ta - ro, dim=-1)
  uu = torch.linalg.cross(ww, up)
  vv = torch.linalg.cross(uu, ww)
  M = torch.stack([uu, vv, ww]) # transposed camera matrix
  rd = torch.cat((uv, torch.ones((B, 1))), dim=-1)
  rd = F.normalize(torch.matmul(rd, M))

  # sphere tracing
  t = torch.zeros(B)
  for _ in range(max_iter):
    p = ro + rd * t[:, None]
    d = sdf(p)
    t += d
    hit = d * d < 1e-6
    if (hit | (t > max_dis)).all():
      break
  
  pos = p
  col = torch.zeros((B, 3))
  # ray trace y=ground to find ground for rays that miss
  t = (ground - ro[1]) / rd[~hit][:,1]
  pos[~hit] = ro + rd[~hit] * t[:, None]

  nor = torch.zeros((B, 3))
  nor[~hit] = torch.tensor([0., 1., 0.])
  nor[hit] = gradient(pos[hit], sdf)

  # albedo
  alb = torch.zeros((B, 3))
  alb[~hit] = torch.full((3,), 1.2)
  alb[hit] = torch.full((3,), 0.5)
  # key light dir
  key = F.normalize(torch.tensor(key_light), dim=0)
  # diffuse/lambertian lighting
  dif = F.relu((nor * key).sum(dim=-1, keepdim=True))
  # soft shadows (doesnt work well)
  sha = torch.ones((B, 1))
  mask = dif[:,0] > 0.01
  # sha[mask] = cast_shadow_ray(pos[mask] + nor[mask] * 0.01, key, 5, sdf).unsqueeze(-1)
  sha[mask] = cast_shadow_ray(pos[mask] + nor[mask] * 0.01, key, 150, sdf).unsqueeze(-1)
  # AO
  occ = cast_occlusion_ray(pos, nor, sdf).unsqueeze(-1)
  # ambient lighting
  amb = 0.8 + 0.2 * nor[:,1]
  amb[~hit] = 0.3
  amb = amb.unsqueeze(-1)
  # back lighting
  bac = F.relu(0.3 + 0.7 * (nor * -key).sum(dim=-1, keepdim=True))
  # bounce lighting
  bou = torch.clamp(0.2 - 0.8 * nor[:,1], 0, 1)
  bou *= torch.clamp(1 - pos[:,1], 0, 1)
  bou = bou.unsqueeze(-1)
  # fresnel
  fre = (nor * rd).sum(dim=-1, keepdim=True)
  fre = torch.pow( torch.clamp(1 + fre, 0, 1), 5 )
  # specular (blinn-phong version)
  hal = F.normalize(key - rd)
  spe = torch.pow( F.relu((nor * hal).sum(dim=-1)), 60 )
  spe *= 0.04 + 0.96 * torch.pow(torch.clamp(1 + (hal * rd).sum(dim=-1), 0, 1), 5)
  spe = spe.unsqueeze(-1)

  col += 0.3 * torch.tensor([1., 1., 1.]) * amb * occ
  col += 0.7 * torch.tensor([1., 1., 1.]) * dif * sha
  col += 0.4 * torch.tensor([1., 1., 1.]) * bac * occ
  col += 0.1 * torch.tensor([1., 1., 1.]) * bou * occ
  col += 0.5 * torch.tensor([1., 1., 1.]) * fre * occ * (0.5 + 0.5 * dif * sha)
  col += 1.0 * torch.tensor([1., 1., 1.]) * spe * occ * dif * sha

  # only shadows on ground plane, no other lighting
  col[~hit] = 0.1 + 0.9 * torch.tensor([1., 1., 1.]) * sha[~hit]
  col[~hit] += 0.1 * torch.tensor([1., 1., 1.]) * amb[~hit] * occ[~hit]

  # col += 0.3 * torch.tensor([1., 1., 1.]) * amb * occ
  # col += 0.7 * torch.tensor([1., 1., 1.]) * dif
  # col += 0.4 * torch.tensor([1., 1., 1.]) * bac * occ
  # col += 0.1 * torch.tensor([1., 1., 1.]) * bou * occ
  # col += 0.5 * torch.tensor([1., 1., 1.]) * fre * occ * (0.4 + 0.6 * dif)
  # col += 1.4 * torch.tensor([1., 1., 1.]) * spe * occ * dif

  col *= alb

  # normal map
  # col = 0.5 * (nor + 1)
  
  # col = torch.tensor([1., 1., 1.]) * sha

  out = torch.ones((res[0], res[1], 3))
  out[res[0] - 1 - pixel_coord[:, 0], pixel_coord[:, 1]] = col

  return out


# https://iquilezles.org/articles/distfunctions/
def sd_box(p, b):
  q = torch.abs(p) - b
  return torch.linalg.norm(F.relu(q), dim=-1) + torch.clamp_max(torch.max(q, dim=-1).values, 0) 


def gaussian_filter(x, y, a):
  if not isinstance(a, torch.Tensor):
    a = torch.tensor(a)
  exp_x = exp_y = torch.exp(-a * 0.5 * 0.5)
  gauss_x = F.relu(torch.exp(-a * x * x) - exp_x)
  gauss_y = F.relu(torch.exp(-a * y * y) - exp_y)
  return gauss_x * gauss_y


# ACES tone mapping curve fit to go from HDR to LDR
# https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
def aces_film(x):
  a = 2.51
  b = 0.03
  c = 2.43
  d = 0.59
  e = 0.14
  return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def linear_to_sRGB(col):
  mask = col < 0.0031308
  col[~mask] = 1.055 * torch.pow(col[~mask], 1.0 / 2.4) - 0.055
  col[mask] = col[mask] * 12.92
  return col


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--width', type=int, default=512)
  parser.add_argument('--height', type=int, default=512)
  parser.add_argument('--samples', type=int, default=10)
  parser.add_argument('--output_path', type=str, default='out/out')
  parser.add_argument('--model_path', type=str, default='net/model.pth')

  parser.add_argument('--camera_angle', type=float, default=0.25)
  parser.add_argument('--max_iter', type=int, default=50)
  parser.add_argument('--camera_height', type=float, default=2.2)
  parser.add_argument('--camera_target', type=float, nargs=3, default=[0, 0, 0])
  parser.add_argument('--key_light', type=float, nargs=3, default=[1, 1.5, 1.4])
  parser.add_argument('--ground', type=float, default=-1)

  args = parser.parse_args()

  net = Siren(in_features=3, out_features=1, hidden_features=256, num_hidden_layers=3)
  net.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))

  def sdf(p, requires_grad=False):
    net.eval()
    with torch.set_grad_enabled(requires_grad):
      if requires_grad:
        y, x = net(p)
        return y.squeeze(-1), x
      else:
        d = sd_box(p, torch.tensor([0.8, 0.8, 0.8]))
        # sdf is undefined outside of [-1,1]^3
        mask = sd_box(p, torch.ones(3)) <= 0
        d[mask] = net(p[mask])[0].squeeze(-1)
        return d

  y_coords, x_coords = torch.meshgrid(torch.arange(args.height), torch.arange(args.width), indexing='ij')
  coords = torch.stack([y_coords, x_coords], dim=-1).reshape(-1, 2)

  print(f"Rendering on {device}")

  start_time = time.time()
  resolution = torch.tensor([args.height, args.width])
  if args.samples > 1:
    # anti-aliasing
    jitter = torch.rand(args.samples, coords.shape[0], 2)
    color = torch.stack(
      [
        shade_image(
          coords + j,
          coords,
          resolution,
          sdf,
          cam_angle=args.camera_angle,
          max_iter=args.max_iter,
          cam_height=args.camera_height,
          cam_target=args.camera_target,
          key_light=args.key_light,
          ground=args.ground
        )
        for j in jitter
      ]
    )
    color = color.mean(dim=0)

    # # gaussian weights
    # jitter -= 0.5
    # wei = gaussian_filter(jitter[...,0], jitter[...,1], 0.5)
    # wei = wei.view(args.samples, args.height, args.width, 1)
    # # weighted avg
    # color *= wei
    # color = color.sum(dim=0) / wei.sum(dim=0)
  else:
    color = shade_image(coords + 0.5, coords, resolution, sdf,
                        cam_angle=args.camera_angle, max_iter=args.max_iter,
                        cam_height=args.camera_height, cam_target=args.camera_target,
                        key_light=args.key_light, ground=args.ground)
  print(f"Render time {time.time() - start_time}")

  output_dir = os.path.dirname(args.output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  print(f"Writing image to {args.output_path + '.exr'}")
  pyexr.write(args.output_path + '.exr', color.cpu().numpy())

  print(f"Writing image to {args.output_path + '.png'}")
  color = aces_film(0.7 * color)
  color = linear_to_sRGB(color)
  # hacky
  d = torch.pow(color - color[0][0], 2)
  color = torch.clamp_max(color + torch.pow(d + 1, -222), 1)

  color = (255 * color).to(torch.uint8)
  im = Image.fromarray(color.cpu().numpy())
  im.save(args.output_path + '.png')