import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from modules import Siren
from dataset import PointCloud
from loss import sdf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, loss_fn, epochs, lr=1e-4, steps_til_summary=300, logs_dir='logs',
          steps_til_checkpoint=300, checkpoints_dir='checkpoints'):
  print(f"Training on {device}")

  optim = torch.optim.Adam(lr=lr, params=model.parameters())
  writer = SummaryWriter(logs_dir)
  os.makedirs(checkpoints_dir, exist_ok=True)

  steps = 0
  with tqdm(total=len(dataloader) * epochs) as pbar:
    for epoch in range(epochs):
      for model_input, ground_truth in dataloader:
        start = time.time()

        coords = model_input['coords'].to(device)
        ground_truth = {k: v.to(device) for k, v in ground_truth.items()}

        model_output = model(coords)
        losses = loss_fn(model_output, ground_truth)
        loss = sum(loss for loss in losses.values())

        writer.add_scalar('loss', loss, steps)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if steps % steps_til_checkpoint == 0:
          torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"model_step_{steps}.pth"))

        if steps % steps_til_summary == 0:
          tqdm.write(f"{epoch=}, {steps=}, loss={loss.item()}, iteration time={time.time() - start}")
          torch.save(model.state_dict(), os.path.join(checkpoints_dir, "model_current.pth"))

        pbar.update(1)
        steps += 1


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--point_cloud_path', type=str, required=True)
  parser.add_argument('--batch_size', type=int, default=1400)
  parser.add_argument('--final_model_path', type=str, default='net/model.pth')

  parser.add_argument('--epochs', type=int, default=10_000)
  parser.add_argument('--lr', type=float, default=1e-4)
  parser.add_argument('--steps_til_summary', type=int, default=300)
  parser.add_argument('--logs_dir', type=str, default='logs')
  parser.add_argument('--steps_til_checkpoint', type=int, default=300)
  parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
  args = parser.parse_args()

  dataset = PointCloud(args.point_cloud_path, args.batch_size)
  dataloader = DataLoader(dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

  net = Siren(in_features=3, out_features=1, hidden_features=256, num_hidden_layers=3).to(device)

  train(net, dataloader, sdf,
        epochs=args.epochs, lr=args.lr,
        steps_til_summary=args.steps_til_summary, logs_dir=args.logs_dir,
        steps_til_checkpoint=args.steps_til_checkpoint, checkpoints_dir=args.checkpoints_dir)

  print(f"Saving model to {args.final_model_path}")
  output_dir = os.path.dirname(args.final_model_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  torch.save(net.state_dict(), args.final_model_path)