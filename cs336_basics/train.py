import os
import argparse
import numpy as np
import torch
import wandb

from cs336_basics.data_loading import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.transformer import Transformer
from cs336_basics.embedding import Embedding
from cs336_basics.crossentropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.adamw import AdamW
from cs336_basics.lr_schedule import learning_rate_schedule


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str,
    num_samples: int = 100
) -> float:
    model.eval()
    total_loss = 0

    for _ in range(num_samples):
        xb, yb = get_batch(val_data, batch_size, context_length, device)
        if torch.max(xb) > 9999:
            print("valid!")
            print(torch.max(xb))
            print(val_data)
            print(xb)
        logits = model(xb)
        loss = cross_entropy(logits, yb)
        total_loss += loss

    return total_loss / num_samples


def main(args: argparse.Namespace) -> None:
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    dtype = torch.float32

    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")

    wandb.init(project=args.wandb_project, config=vars(args))
    emb = Embedding(args.vocab_size, args.d_model, device, dtype)
    model = Transformer(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        device,
        dtype
    )

    optimizer = AdamW(
        model.parameters(),
        args.max_lr,
        (args.beta0, args.beta1),
        1e-8,
        args.decay
    )

    iteration = 0
    if args.resume and os.path.exists(args.resume):
        iteration = load_checkpoint(args.resume, model, optimizer)

    while iteration < args.max_iters:
        lr = learning_rate_schedule(
            iteration,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_iters
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        xb, yb = get_batch(train_data, args.batch_size, args.context_length, device)
        if torch.max(xb) > 9999:
            print("Initial")
            print(torch.max(xb))
        logits = model(xb)
        loss = cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if iteration % args.log_interval == 0:
            model.eval()
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, device)
            model.train()

            wandb.log({
                "train_loss": loss.item(),
                "val_loss": val_loss.item(),
                "lr": lr,
                "iter": iteration
            })

            print(f"Iter {iteration}: Train loss {loss.item():.4f}, Val loss {val_loss.item():.4f}")

        if iteration % args.ckpt_interval == 0 and args.ckpt_path:
            save_checkpoint(model, optimizer, iteration, args.ckpt_path)

        iteration += 1

    if args.ckpt_path:
        save_checkpoint(model, optimizer, iteration, args.ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="checkpoint.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="transformer-training")

    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--beta0", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.999)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_iters", type=int, default=9000)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--ckpt_interval", type=int, default=1000)

    args = parser.parse_args()
    main(args)
