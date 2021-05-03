import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import beta


def utility_score(context, actions, device='cuda', mode='metrics'):

    # context columns: 'date', 'weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp', 'ts_id', 'regime'
    dates, weights, resps = context[:,0], context[:,1], context[:,6]

    if mode == 'loss':
        # generalization
        resps = context[np.arange(context.shape[0]), 2+np.random.choice(5, context.shape[0])]
        resps = torch.normal(mean=resps, std=torch.abs(resps)/2)

    dates_involved = torch.unique(dates)
    daily_profit = []

    for d in dates_involved:
        pnl = torch.mul(torch.mul(weights,d==dates),resps).unsqueeze(dim=0)
        daily_profit.append(torch.matmul(pnl, actions))
        
    p = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
    vol = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        
    for dp in daily_profit:
        p = p + dp
        vol = vol + dp**2
    
    t = p / vol**.5 * (250/len(dates_involved))**.5

    ceiling = torch.tensor(6, dtype=torch.float32, requires_grad=False, device=device)
    floor = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
    t = torch.min(torch.max(t, floor), ceiling)

    # if profit is negative the utility score is not clipped to 0 in loss mode (for learning purposes)
    if mode == 'loss' and p < 0.0:
        u = p
    else:
        u = torch.mul(p, t)

    if mode == 'loss':
        return -u
    else:
        return t.cpu().item(), p.cpu().item(), u.cpu().item()


def blend(x, context, weights, respect_side=True, ab=0.8):

    if len(x) < 2:
        return x, context, weights

    # the leftover data point if exists is dismissed (it will come up in future epochs, batches are shuffled)
    if x.shape[0] % 2 > 0:
        x = x[:-1]
        context = context[:-1]
        weights = weights[:-1]

    if respect_side:
        side_idx = x[:,0].sort()[1]
        x = x[side_idx]
        context = context[side_idx]
        weights = weights[side_idx]

    b = torch.tensor(beta.rvs(ab, ab, size=x.shape[0]//2), device='cuda', dtype=torch.float32).reshape(-1,1)

    # blending pairs
    blended_x = b * x[::2] + (1-b) * x[1::2]
    blended_c = b * context[::2] + (1-b) * context[1::2]
    blended_w = b * weights[::2] + (1-b) * weights[1::2]

    # the side of the blended data points is collapsed to the closest value
    blended_c[:,0] = torch.where(b > 0.5, context[::2,0].reshape(-1,1), context[1::2,0].reshape(-1,1)).squeeze()

    return blended_x, blended_c, blended_w


def melt(x, context, chunk_size, sortby=6):

    days = context[:,0]
    sides = x[:,0]

    melted_x = []
    melted_c = []

    for d in torch.unique(days):
        
        for s in np.array([-1,+1]):

            grouped_x = x[(days==d.item()) & (sides==s)]
            grouped_c = context[(days==d.item()) & (sides==s)]

            if len(grouped_x) > 1:

                sorted_idx = grouped_c[:,sortby].sort()[1]

                chunks_x = torch.split(grouped_x[sorted_idx], chunk_size, dim=0)
                chunks_c = torch.split(grouped_c[sorted_idx], chunk_size, dim=0)

                for chidx in np.arange(len(chunks_x)):
                    melted_x.append(torch.mean(chunks_x[chidx], dim=0))
                    melted_c.append(torch.mean(chunks_c[chidx], dim=0))

            elif len(grouped_x) == 1:
                melted_x.append(grouped_x)
                melted_c.append(grouped_c)

    return torch.vstack(melted_x), torch.vstack(melted_c)


def smooth_bce(logits, targets, smoothing=0.05, weight=None):

    with torch.no_grad():
        smooth_targets = targets * (1.0 - smoothing) + 0.5 * smoothing

    return F.binary_cross_entropy_with_logits(logits.squeeze(), smooth_targets.squeeze(), weight=weight)

