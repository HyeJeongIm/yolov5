import torch

def eval(net, c, image, device):
    net.eval()
    with torch.no_grad():
        image = image.float().to(device)
        z = net(image)
        dist = torch.sum((z - c) ** 2, dim=1)
        score = dist.cpu().numpy().tolist()
    return score