import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import contextlib
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from model.model_utils import *
from model.losses import SinkhornKnopp, KD
from model.get_model import get_backbone
from utils.parse import get_args
from data.augmentations import get_transform
from data.get_datasets import get_datasets
from utils.eval_utils import split_cluster_acc_v2, cluster_eval
import wandb
import os
import numpy as np
from tqdm import tqdm
import copy
from torch.optim import AdamW, SGD
import pickle as pkl

class Net(nn.Module):

    def __init__(self,
                 backbone,
                 num_labeled,
                 num_unlabeled,
                 feat_dim=512,
                 hidden_dim=2048,
                 proj_dim=256,
                 num_heads=5,
                 num_hidden_layers=1):
        super().__init__()

        self.encoder = backbone
        self.feat_dim = feat_dim

        self.head_lab = Prototypes(self.feat_dim, num_labeled)
        if num_heads is not None:
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        if getattr(self, "head_unlab", False):
            self.head_unlab.normalize_prototypes()

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(F.normalize(feats))}
        if hasattr(self, "head_unlab"):
            logits_unlab, proj_feats_unlab = self.head_unlab(feats)
            out.update({
                "logits_unlab": logits_unlab,
                "proj_feats_unlab": proj_feats_unlab
            })
        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder(views)
            out = self.forward_heads(feats)
            out["feats"] = feats
            return out


def cross_entropy_loss(preds, targets, temperature):
    preds = F.log_softmax(preds / temperature, dim=-1)
    return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)


def swapped_prediction(args, logits, targets):
    loss = 0
    for view in range(args.num_large_crops):
        for other_view in np.delete(range(args.num_crops), view):
            loss += cross_entropy_loss(logits[other_view],
                                       targets[view],
                                       temperature=args.temperature)
    return loss / (args.num_large_crops * (args.num_crops - 1))


def train_pretrain(model, train_loader, test_loader, args):
    model = model.cuda()
    model_statistics(model)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum_opt,
        weight_decay=args.weight_decay_opt,
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.pretrain_epochs,
        warmup_start_lr=args.min_lr,
        eta_min=args.min_lr,
    )

    # train
    for epoch in range(args.pretrain_epochs):
        model.train()
        bar = tqdm(train_loader)
        nlc = args.num_labeled_classes

        for batch in bar:
            optimizer.zero_grad()
            images, labels, _ = batch

            labels = labels.cuda(non_blocking=True)
            images = [image.cuda(non_blocking=True) for image in images] 
            # normalize prototypes
            model.normalize_prototypes()
            # forward
            outputs = model(images)

            # supervised los
            loss = torch.stack([
                F.cross_entropy(o / args.temperature, labels)
                for o in outputs["logits_lab"]
            ]).mean()

            loss.backward()
            optimizer.step()

            bar.set_postfix(
                {"loss": "{:.2f}".format(loss.detach().cpu().numpy())})

        with torch.no_grad():
            model.eval()
            bar = tqdm(test_loader)
            preds = None
            labels = None
            for batch in bar:
                images, label, _ = batch
                label, images = label.cuda(non_blocking=True), images.cuda(
                    non_blocking=True)
                outputs = model(images)
                if preds is None:
                    preds = outputs["logits_lab"]
                    labels = label
                else:
                    preds = torch.cat([preds, outputs["logits_lab"]], dim=0)
                    labels = torch.cat([labels, label], dim=0)
            acc = torch.mean((torch.argmax(preds, dim=1) == labels).float())
            print("Epoch: {}, Lr: {}, Pretrain acc: {:.2f}".format(epoch, optimizer.param_groups[0]["lr"], acc))

        scheduler.step()
    # save model
    if args.save_model:
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(
            model_to_save.state_dict(),
            os.path.join(
                args.model_save_dir, "pretrained_{}_{}_{}.pth".format(
                    args.dataset, args.num_labeled_classes,
                    args.num_unlabeled_classes)))


def train_discover(model, old_model, train_loader, train_val_loader, test_loader, args):
    if args.pretrained_path is not None:
        print("Load supervised pretrain from {}".format(args.pretrained_path))
        state_dict = torch.load(args.pretrained_path)
        updated_state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        model.load_state_dict(updated_state_dict, strict=False)
        old_model.load_state_dict(updated_state_dict, strict=False)

    model = model.cuda()
    old_model = old_model.cuda()
    old_model.eval()
    for _, p in old_model.named_parameters():
        p.requires_grad = False

    model_statistics(model)
    if "cifar" in args.dataset:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum_opt, weight_decay=args.weight_decay_opt)
    else:
        print("adam")
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_opt)

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        warmup_start_lr=args.min_lr,
        eta_min=args.min_lr,
    )

    # wandb
    state = {k: v for k, v in args._get_kwargs()}
    wandb.init(project=args.project,
               entity=args.entity,
               config=state,
               name=args.comment,
               dir=args.log_dir)

    sk = SinkhornKnopp()
    loss_per_head = torch.zeros(args.num_heads).cuda()

    start_epoch = 0
    # train
    best_scores = {"epoch": 0, "acc": 0}
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        bar = tqdm(train_loader)
        nlc = args.num_labeled_classes
        scaler = GradScaler()
        amp_cm = autocast() if args.amp else contextlib.nullcontext()

        for batch in bar:
            optimizer.zero_grad()
            images, labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            labels, mask_lab = labels.cuda(non_blocking=True), mask_lab.cuda(
                non_blocking=True).bool()
            images = [image.cuda() for image in images]

            # normalize prototypes
            model.normalize_prototypes()
            with amp_cm:
                # forward
                outputs = model(images)
                with torch.no_grad():
                    old_outputs = old_model(images)
                    old_outputs["logits_lab"] = (old_outputs["logits_lab"].unsqueeze(1).expand(-1, args.num_heads, -1, -1))
                    old_logits = old_outputs["logits_lab"].detach()
                                # gather outputs

                # gather outputs
                outputs["logits_lab"] = (
                    outputs["logits_lab"].unsqueeze(1).expand(
                        -1, args.num_heads, -1, -1))
                logits = torch.cat(
                    [outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)

                # create targets
                targets_lab = F.one_hot(
                    labels[mask_lab],
                    num_classes=args.num_labeled_classes).float()

                targets = torch.zeros_like(logits)

                # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
                for v in range(args.num_large_crops):
                    for h in range(args.num_heads):
                        targets[v, h,
                                mask_lab, :nlc] = targets_lab.type_as(targets)

                        targets[v, h, ~mask_lab,
                                nlc:] = sk(outputs["logits_unlab"][
                                    v, h, ~mask_lab]).type_as(targets)

                # compute swapped prediction loss
                loss_cluster = swapped_prediction(args, logits, targets)

                kd_loss = KD(args, old_logits[:args.num_large_crops], logits[:args.num_large_crops], mask_lab, T=args.kd_temperature)

                kd_loss = args.alpha * kd_loss.mean()

                # update best head tracker
                loss_per_head += loss_cluster.clone().detach()

                loss_cluster = loss_cluster.mean()

                loss = loss_cluster + kd_loss

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bar.set_postfix(
                {"loss": "{:.2f}".format(loss.detach().cpu().numpy())})
            results = {
                "loss": loss.clone(),
                "loss_cluster": loss_cluster.clone(),
                "kd_loss": kd_loss.clone(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            wandb.log(results)
        scheduler.step()
        best_head = torch.argmin(loss_per_head)
        
        test_results = test(args, model, test_loader, best_head, prefix="test")
        train_results = test(args,
                            model,
                            train_val_loader,
                            best_head,
                            prefix="train")


        wandb.log(train_results)

        # save model
        if args.save_model:
            model_to_save = model.module if hasattr(model, "module") else model
            state_save = {
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(
                state_save,
                os.path.join(args.model_save_dir, "latest_checkpoint.pth"))
            if train_results["train/novel/avg"] > best_scores["acc"]:
                best_scores["acc"] = train_results["train/novel/avg"]
                best_scores.update(train_results)
                torch.save(
                    state_save,
                    os.path.join(args.model_save_dir, "best_checkpoint.pth"))

        # log
        lr = np.around(optimizer.param_groups[0]["lr"], 4)
        print("--Comment-{}--Epoch-[{}/{}]--LR-[{}]--Train-Novel-[{:.2f}]--"
            "Test-All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]".format(
                args.comment, epoch, args.max_epochs, lr,
                train_results["train/novel/avg"] * 100,
                test_results["test/all/avg"] * 100,
                test_results["test/novel/avg"] * 100,
                test_results["test/seen/avg"] * 100))


@torch.no_grad()
def test(args, model, val_dataloader, best_head, prefix):
    model.eval()
    all_labels = None
    all_preds = None
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            images, labels, _ = batch
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)

            if prefix == "train":
                preds_inc = outputs["logits_unlab"]
            else:
                preds_inc = torch.cat(
                    [
                        outputs["logits_lab"].unsqueeze(0).expand(
                            args.num_heads, -1, -1),
                        outputs["logits_unlab"],
                    ],
                    dim=-1,
                )

            preds_inc = preds_inc.max(dim=-1)[1]

            if all_labels is None:
                all_labels = labels
                all_preds = preds_inc
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)
                all_preds = torch.cat([all_preds, preds_inc], dim=1)

    all_labels = all_labels.detach().cpu().numpy()
    all_preds = all_preds.detach().cpu().numpy()

    results = {}
    for head in range(args.num_heads):
        if prefix == "train":
            _res = cluster_eval(all_labels, all_preds[head])
        else:
            _res = split_cluster_acc_v2(all_labels,
                                        all_preds[head],
                                        num_seen=args.num_labeled_classes)

        for key, value in _res.items():
            if key in results.keys():
                results[key].append(value)
            else:
                results[key] = [value]

    log = {}
    for key, value in results.items():
        log[prefix + "/" + key + "/" + "avg"] = round(
            sum(value) / len(value), 4)
        log[prefix + "/" + key + "/" + "best"] = round(value[best_head], 4)

    return log


if __name__ == "__main__":
    args = get_args()
    # model
    backbone = get_backbone(args)
    model = Net(backbone,
                num_labeled=args.num_labeled_classes,
                num_unlabeled=args.num_unlabeled_classes,
                num_heads=args.num_heads,
                feat_dim=args.feat_dim)

    # dataset
    train_transform, test_transform = get_transform(args=args)

    train_dataset, test_dataset, val_dataset, test_seen_dataset = get_datasets(
        args.dataset, train_transform, test_transform, args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_seen_loader = torch.utils.data.DataLoader(
        test_seen_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    train_val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if args.eval:
        with torch.no_grad():
            print(f'==> Resuming from checkpoint {args.eval_model_path} for evaluation.')
            checkpoint = torch.load(args.eval_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
            best_head = 0
            train_results = test(args, model, train_val_loader, best_head, prefix="train")
            test_results = test(args, model, test_loader, best_head, prefix="test")
            print(f"test results: {test_results}, train results: {train_results}")
    else:
        if args.pretrain:
            train_pretrain(model, train_loader, test_seen_loader, args)
        else:
            old_model = Net(copy.deepcopy(backbone),
                num_labeled=args.num_labeled_classes,
                num_unlabeled=args.num_unlabeled_classes,
                num_heads=None,
                feat_dim=args.feat_dim)
            train_discover(model, old_model, train_loader, train_val_loader, test_loader, args)