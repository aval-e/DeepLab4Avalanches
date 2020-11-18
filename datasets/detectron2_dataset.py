from datasets.avalanche_inst_dataset import AvalancheInstDataset
import detectron2.structures as structures


def detectron_targets_to_torchvision(sample):
    targets = {'boxes': sample.gt_boxes.tensor,
               'labels': sample.gt_classes,
               'masks': sample.gt_masks.tensor,
               }
    return targets


def detectron_preds_to_torchvision(sample):
    preds = {'boxes': sample.pred_boxes.tensor,
             'labels': sample.pred_classes,
             'masks': sample.pred_masks.unsqueeze(dim=1),
             }
    return preds


class Detectron2Dataset(AvalancheInstDataset):

    def __getitem__(self, idx):
        samples = super(Detectron2Dataset, self).__getitem__(idx)
        if isinstance(samples, tuple):
            return self.to_detectron_sample(samples)

        # batch augmentation case
        return [self.to_detectron_sample(sample) for sample in samples]

    def to_detectron_sample(self, sample):
        image, targets = sample
        instances = structures.Instances((self.tile_size, self.tile_size),
                              gt_boxes=structures.Boxes(targets['boxes']),
                              gt_classes=targets['labels'],
                              gt_masks=structures.BitMasks(targets['masks']))
        sample = {'image': image,
                  'instances': instances,
                  }
        return sample