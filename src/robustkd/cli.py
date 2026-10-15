import argparse
import io
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR100
from torchvision.transforms import functional as TF
from PIL import Image, ImageEnhance
from .model import Teacher, Student, cost
from .common import load, save, table, plot


SEEN = {"gaussian_noise", "motion_blur", "brightness", "jpeg_compression"}
CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def corrupt(image, rng):
    severity = float(rng.uniform(0.01, 1.0))
    kind = rng.choice(sorted(SEEN))
    if kind == "gaussian_noise":
        return np.uint8(
            np.clip(
                np.asarray(image, dtype=float)
                + rng.normal(0, 40 * severity, (32, 32, 3)),
                0,
                255,
            )
        ), severity
    if kind == "brightness":
        return np.array(ImageEnhance.Brightness(image).enhance(1 + severity)), severity
    if kind == "jpeg_compression":
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=max(5, int(95 - 85 * severity)))
        stream.seek(0)
        return np.array(Image.open(stream).convert("RGB")), severity
    x = np.asarray(image, dtype=float)
    radius = int(1 + 3 * severity)
    padded = np.pad(x, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    return np.uint8(
        sum(padded[:, j : j + 32] for j in range(2 * radius + 1)) / (2 * radius + 1)
    ), severity


def tensor(x, device):
    x = (
        torch.as_tensor(np.asarray(x).copy(), device=device).permute(0, 3, 1, 2).float()
        / 255
    )
    return (
        x - x.new_tensor([0.5071, 0.4867, 0.4408])[None, :, None, None]
    ) / x.new_tensor([0.2675, 0.2565, 0.2761])[None, :, None, None]


def train(a, c):
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    dataset = CIFAR100(a.data, train=True, download=True)
    model = Teacher() if a.method == "teacher" else Student()
    model.to(c["device"])
    teacher = None
    projection = None
    if a.method not in ("teacher", "student"):
        if not a.teacher:
            raise ValueError("--teacher is required for distillation")
        teacher = Teacher().to(c["device"])
        teacher.load_state_dict(
            torch.load(a.teacher, map_location=c["device"], weights_only=True)["model"]
        )
        teacher.eval().requires_grad_(False)
        projection = nn.Linear(model.features, teacher.features, bias=False).to(
            c["device"]
        )
    params = list(model.parameters()) + (
        list(projection.parameters()) if projection else []
    )
    opt = torch.optim.SGD(
        params, lr=c["lr"], momentum=c["momentum"], weight_decay=c["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, c["milestones"], gamma=0.1)
    history = []
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    save(output / "config.json", dict(c, method=a.method))
    for epoch in range(c["epochs"]):
        model.train()
        total = 0.0
        indices = rng.permutation(len(dataset))
        for start in range(0, len(indices), c["batch_size"]):
            ids = indices[start : start + c["batch_size"]]
            clean = []
            altered = []
            levels = []
            for i in ids:
                image = Image.fromarray(dataset.data[i])
                image = TF.pad(image, 4, padding_mode="reflect")
                top, left = rng.integers(0, 9, size=2)
                image = TF.crop(image, int(top), int(left), 32, 32)
                if rng.random() < 0.5:
                    image = TF.hflip(image)
                other, level = corrupt(image, rng)
                clean.append(np.asarray(image))
                altered.append(other)
                levels.append(level)
            x, xc = (
                tensor(np.stack(clean), c["device"]),
                tensor(np.stack(altered), c["device"]),
            )
            y = torch.as_tensor(np.asarray(dataset.targets)[ids], device=c["device"])
            z, f = model(x)
            zc, fc = model(xc)
            loss = nn.functional.cross_entropy(z, y)
            if teacher:
                with torch.no_grad():
                    zt, ft = teacher(x)
                    ztc, ftc = teacher(xc)
                temperature = c["temperature"]
                kd = (
                    nn.functional.kl_div(
                        (z / temperature).log_softmax(-1),
                        (zt / temperature).softmax(-1),
                        reduction="batchmean",
                    )
                    * temperature**2
                )
                loss = loss + c["kd"] * kd
                if a.method in ("feature", "response", "robustkd"):
                    loss = loss + c["feature"] * (projection(f) - ft).square().mean()
                if a.method in ("response", "robustkd"):
                    dz = 1 - nn.functional.cosine_similarity(
                        zc - z, ztc - zt, dim=1, eps=1e-8
                    )
                    df = (
                        (
                            nn.functional.normalize(projection(fc - f), dim=1)
                            - nn.functional.normalize(ftc - ft, dim=1)
                        )
                        .square()
                        .sum(1)
                    )
                    rel = (
                        (
                            ztc.softmax(-1).max(-1).values
                            * torch.exp(
                                -c["eta"] * torch.tensor(levels, device=c["device"])
                            )
                        )
                        if a.method == "robustkd"
                        else 1.0
                    )
                    loss = loss + c["response"] * (rel * (dz + c["mu"] * df)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(ids)
        scheduler.step()
        history.append(
            dict(
                epoch=epoch + 1, loss=total / len(dataset), lr=opt.param_groups[0]["lr"]
            )
        )
        table(output / "history.csv", history)
    torch.save(
        dict(
            model=model.cpu().state_dict(),
            architecture="teacher" if a.method == "teacher" else "student",
            config=c,
        ),
        output / "model.pt",
    )


def evaluate(a, c):
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    model = (Teacher() if state["architecture"] == "teacher" else Student()).to(
        c["device"]
    )
    model.load_state_dict(state["model"])
    model.eval()

    def accuracy(images, labels):
        correct = 0
        with torch.no_grad():
            for start in range(0, len(labels), c["batch_size"]):
                x = tensor(images[start : start + c["batch_size"]], c["device"])
                pred = model(x)[0].argmax(1).cpu().numpy()
                correct += np.sum(pred == labels[start : start + c["batch_size"]])
        return float(correct / len(labels))

    test = CIFAR100(a.data, train=False, download=True)
    rows = []
    labels = np.load(Path(a.corruptions) / "labels.npy")
    for kind in CORRUPTIONS:
        images = np.load(Path(a.corruptions) / f"{kind}.npy", mmap_mode="r")
        if len(images) != 5 * len(test):
            raise ValueError("Expected five ordered severity blocks of 10000 images")
        for severity in range(1, 6):
            yy = (
                labels[(severity - 1) * len(test) : severity * len(test)]
                if len(labels) == len(images)
                else labels
            )
            if len(yy) != len(test):
                raise ValueError("Unexpected corruption label count")
            rows.append(
                dict(
                    corruption=kind,
                    severity=severity,
                    accuracy=accuracy(
                        images[(severity - 1) * len(test) : severity * len(test)], yy
                    ),
                    seen=kind in SEEN,
                )
            )
    params, flops = cost(model)
    summary = dict(
        method=a.method,
        clean_accuracy=accuracy(test.data, np.array(test.targets)),
        seen_accuracy=np.mean([r["accuracy"] for r in rows if r["seen"]]),
        unseen_accuracy=np.mean([r["accuracy"] for r in rows if not r["seen"]]),
        overall_accuracy=np.mean([r["accuracy"] for r in rows]),
        parameters=params,
        flops=flops,
    )
    table(Path(a.output) / "corruptions.csv", rows)
    save(Path(a.output) / "metrics.json", summary)
    table(
        Path(a.output) / "severity.csv",
        [
            dict(
                severity=s,
                accuracy=np.mean([r["accuracy"] for r in rows if r["severity"] == s]),
            )
            for s in range(1, 6)
        ],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["prepare", "train", "evaluate", "compare", "plot"]
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--data", default="data/cifar100")
    p.add_argument("--corruptions", default="data/CIFAR-100-C")
    p.add_argument("--output", default="runs/robustkd")
    p.add_argument(
        "--method",
        choices=["teacher", "student", "kd", "feature", "response", "robustkd"],
        default="robustkd",
    )
    p.add_argument("--teacher")
    p.add_argument("--checkpoint")
    p.add_argument("--inputs", nargs="*")
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        CIFAR100(a.data, train=True, download=True)
        CIFAR100(a.data, train=False, download=True)
    elif a.command == "train":
        train(a, c)
    elif a.command == "evaluate":
        evaluate(a, c)
    elif a.command == "compare":
        rows = [load(Path(d) / "metrics.json") for d in a.inputs]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
