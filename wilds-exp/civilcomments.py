from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from wilds.common.metrics.all_metrics import Accuracy


class MyCivilCommentsDataset(CivilCommentsDataset):
  def eval(self, y_pred, y_true, metadata):
    if type(metadata) is tuple:
      algorithm = metadata[1]
      metadata = metadata[0]
      results, results_str = super().eval(y_pred, y_true, metadata)

      # Class Accuracy
      metric = Accuracy()
      acc_vec = metric.compute_element_wise(y_pred, y_true, False)
      acc0 = acc_vec[y_true < 0.5].mean().item()
      acc1 = acc_vec[y_true > 0.5].mean().item()
      results[f'acc0'] = acc0
      results[f'acc1'] = acc1
      results[f'worst-class'] = min(acc0, acc1)
      results_str += f'acc0: {acc0}\nacc1: {acc1}\n'

      # Validation Loss
      d = dict()
      d['y_pred'] = y_pred
      d['y_true'] = y_true
      val_loss = algorithm.objective(d).item()
      results[f'val_loss'] = val_loss
      results_str += f'val loss: {val_loss}\n'
    else:
      results, results_str = super().eval(y_pred, y_true, metadata)

    return results, results_str