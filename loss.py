import torch
import torch.nn.functional as F

# https://github.com/vsitzmann/siren/blob/master/loss_functions.py
def sdf(model_output, ground_truth):
  gt_sdf = ground_truth['sdf']
  gt_normals = ground_truth['normals']

  pred_sdf, coords = model_output
  gradient = torch.autograd.grad(pred_sdf, coords, torch.ones_like(pred_sdf), create_graph=True)[0]

  # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
  sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
  inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
  normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                  torch.zeros_like(gradient[..., :1]))
  grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
  # Exp      # Lapl
  # -----------------
  return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
          'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
          'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
          'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1