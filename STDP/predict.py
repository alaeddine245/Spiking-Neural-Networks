import torch
import numpy as np
import os


def pass_through_network(model, loader, X_path=None, y_path=None, device='cuda'):
    if X_path is not None and os.path.isfile(X_path):
        print('Entered the if statement')
        features = torch.load(X_path)
        targets = torch.load(y_path)
    else:
        features = []
        targets = []
        # one single pass
        for data, target in loader:
            features.extend(pass_batch_through_network(model, data, device))
            print('Got the features')
            targets.extend(target.tolist())
        print('Finished preprocessing')
        print('Ready to save')
        if X_path is None:
            torch.save(features, 'tmp/test_x.pt')
            torch.save(targets, 'tmp/test_y.pt')
        else:
            torch.save(features, X_path)
            torch.save(targets, y_path)
    return features, targets


def pass_batch_through_network(model, batch, device='cuda'):
    with torch.no_grad():
        ans = []
        for data in batch:
            data_in = data.to(device)
            output = model(data_in)
            ans.append(output.reshape(-1).cpu().tolist())
        print("Finished passing through the network")
        return ans
    

def eval(X, y, predictions):
    non_silence_mask = np.count_nonzero(X, axis=1) > 0
    correct_mask = predictions == y
    correct_non_silence = np.logical_and(correct_mask, non_silence_mask)
    correct = np.count_nonzero(correct_non_silence)
    silence = np.count_nonzero(~non_silence_mask)
    return (sum(correct_mask) / len(X), (len(X) - (correct + silence)) / len(X), silence / len(X))
