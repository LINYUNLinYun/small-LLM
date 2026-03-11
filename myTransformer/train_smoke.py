import torch

from modelArgs import ModelArgs
from myTransformer import Transformer


def build_toy_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    # 固定小数据集，便于快速过拟合验证前后向是否正常
    idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = idx.clone()
    return idx, targets


def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = ModelArgs(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_embd=64,
        max_seq_len=64,
        dropout=0.1,
        vocab_size=100,
        block_size=32,
    )

    model = Transformer(args).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    idx, targets = build_toy_batch(batch_size=16, seq_len=32, vocab_size=args.vocab_size, device=device)

    steps = 1000
    print(f"device={device}, steps={steps}")

    first_loss = None
    last_loss = None

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        if first_loss is None:
            first_loss = last_loss

        if step % 10 == 0 or step == 1:
            print(f"step={step:03d} loss={last_loss:.4f}")

    print(f"initial_loss={first_loss:.4f}")
    print(f"final_loss={last_loss:.4f}")

    if last_loss < first_loss:
        print("smoke test passed: loss decreased")
    else:
        print("smoke test warning: loss did not decrease")


if __name__ == "__main__":
    main()
