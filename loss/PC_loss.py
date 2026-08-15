import torch
import torch.nn.functional as F

from abc import ABC
from torch import nn


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self,
                 temperature=0.07,
                 base_temperature=0.07,
                 max_samples=1024,
                 max_views=100,
                 ignore_index=-1,
                 device='cuda:0',
                 memory=True,
                 memory_size=100,
                 pixel_update_freq=10,
                 pixel_classes=3,
                 percentile=0.8,
                 dim=256,
                 use_prototype=True):

        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_index

        self.max_samples = max_samples
        self.max_views = max_views

        self.device = device

        self.memory = memory
        self.memory_size = memory_size
        self.pixel_update_freq = pixel_update_freq
        self.percentile = percentile
        self.use_prototype = use_prototype
        if use_prototype:
            self.prototype = nn.Parameter(
                torch.randn(pixel_classes, dim),
                requires_grad=True
            )
            self.proto_momentum = 0.9
        if self.memory:
            self.segment_queue = torch.randn(pixel_classes, self.memory_size, dim)
            self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
            self.segment_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)
            self.pixel_queue = torch.zeros(pixel_classes, self.memory_size, dim)
            self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
            self.pixel_queue_ptr = torch.zeros(pixel_classes, dtype=torch.long)

    def _update_prototype(self, features, labels):
        with torch.no_grad():
            for cls_id in torch.unique(labels):
                if cls_id == self.ignore_label:
                    continue
                mask = (labels == cls_id)
                curr_proto = features[mask].mean(dim=0)
                self.prototype[cls_id] = (
                        self.proto_momentum * self.prototype[cls_id] +
                        (1 - self.proto_momentum) * curr_proto
                )

    def _prototype_contrast(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        prototypes = F.normalize(self.prototype, p=2, dim=1)
        sim = torch.mm(features, prototypes.T) / self.temperature
        mask = torch.zeros_like(sim)
        mask.scatter_(1, labels.unsqueeze(1), 1)
        exp_sim = torch.exp(sim)
        pos = (exp_sim * mask).sum(dim=1)
        neg = (exp_sim * (1 - mask)).sum(dim=1)
        loss = -torch.log(pos / (pos + neg)).mean()
        return loss
    def _dequeue_and_enqueue_signal(self, keys, labels):

        n_view = keys.shape[1]
        feat_dim = keys.shape[2]

        keys = keys.permute(2, 0, 1)
        labels = labels.unsqueeze(-1).repeat(1, n_view).unsqueeze(0)


        this_feat = keys.contiguous().view(feat_dim, -1)
        this_label = labels.contiguous().view(-1)
        this_label_ids = torch.unique(this_label)
        this_label_ids = [x for x in this_label_ids if x > 0]
        for lb in this_label_ids:
            idxs = (this_label == lb).nonzero()
            lb = int(lb.item())

            feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
            ptr = int(self.segment_queue_ptr[lb])
            self.segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
            self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.memory_size


            num_pixel = idxs.shape[0]
            perm = torch.randperm(num_pixel)
            K = min(num_pixel, self.pixel_update_freq)
            feat = this_feat[:, perm[:K]]
            feat = torch.transpose(feat, 0, 1)
            ptr = int(self.pixel_queue_ptr[lb])

            if ptr + K >= self.memory_size:
                self.pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                self.pixel_queue_ptr[lb] = 0
            else:
                self.pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + 1) % self.memory_size
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        labels = torch.nn.functional.interpolate(labels, (keys.shape[2], keys.shape[3]), mode='nearest')

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x > 0]
            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()
                lb = int(lb.item())

                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(self.segment_queue_ptr[lb])
                self.segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.memory_size


                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, self.pixel_update_freq)
                feat = this_feat[:, perm[:K]]
                feat = torch.transpose(feat, 0, 1)
                ptr = int(self.pixel_queue_ptr[lb])

                if ptr + K >= self.memory_size:
                    self.pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = 0
                else:
                    self.pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + 1) % self.memory_size
    def _anchor_sampling(self, X, y_hat, y, plabel=False, en_map=None):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]


            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]

            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)
        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)


        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).to(self.device)
        y_ = torch.zeros(total_classes, dtype=torch.float).to(self.device)

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:

                if plabel and cls_id!=0 and ii < batch_size//2:
                    condition = (this_y_hat == cls_id) & (this_y != cls_id)

                    selected_indices = torch.nonzero(condition).squeeze(1)


                    sorted_indices = selected_indices[torch.argsort(en_map[ii][cls_id - 1][selected_indices], descending=False)]



                    length = sorted_indices.size(0)
                    start = int(length * self.percentile) - (n_view // 2)
                    start = max(0, min(start, length - n_view))

                    middle_indices = sorted_indices[start:start + n_view]
                    hard_indices = middle_indices.unsqueeze(-1)

                else:
                    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()

                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()
                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]


                if num_hard >= n_view and num_easy >= n_view:
                    num_hard_keep = n_view
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view:
                    num_hard_keep = n_view
                    num_easy_keep = n_view - num_hard_keep
                elif num_easy >= n_view:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                elif num_easy < n_view and num_hard < n_view:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1
        return X_, y_

    def _sample_negative(self, Q):
        class_num, memory_size, feat_size = Q.shape

        x_ = torch.zeros((class_num * memory_size, feat_size)).float().to(self.device)
        y_ = torch.zeros((class_num * memory_size, 1)).float().to(self.device)

        sample_ptr = 0
        for c in range(class_num):
            if c == 0:
                continue
            this_q = Q[c, :memory_size, :]
            x_[sample_ptr:sample_ptr + memory_size, ...] = this_q
            y_[sample_ptr:sample_ptr + memory_size, ...] = c
            sample_ptr += memory_size
        return x_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)


        logits = F.cosine_similarity(anchor_feature.unsqueeze(1), contrast_feature.unsqueeze(0), dim=2) / self.temperature

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).\
            scatter_(1, torch.arange(anchor_num * anchor_count).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-5)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss

    def forward(self, feats, labels=None, predict=None, plabel=False, en_map=None):

        queue_feats = feats.detach()
        queue_labels = labels.detach()

        queue = self.segment_queue if self.memory else None


        labels = labels.float().clone()
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2], feats.shape[3]), mode='nearest')
        predict = torch.nn.functional.interpolate(predict, (feats.shape[2], feats.shape[3]), mode='nearest')

        labels = labels.long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        if plabel:
            en_map = torch.nn.functional.interpolate(en_map, (feats.shape[2], feats.shape[3]))
            en_map = en_map.contiguous().view(batch_size//2, en_map.shape[1], -1)

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])




        feats_, labels_ = self._anchor_sampling(feats, labels, predict, plabel, en_map)
        pixel_loss = self._contrastive(feats_, labels_, queue=queue)
        if self.use_prototype:

            self._update_prototype(feats, labels)

            proto_loss = self._prototype_contrast(feats, labels)

        if self.memory:

            self._dequeue_and_enqueue(queue_feats, queue_labels)
        loss =  pixel_loss + proto_loss
        return loss