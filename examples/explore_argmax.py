import torch


def main():
    x = torch.Tensor([5, 4, 1, 2])
    print(torch.argmax(x))


if __name__ == '__main__':
    main()