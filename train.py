import os
from pathlib import Path
from dataclasses import dataclass
import warnings
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.cuda.amp import GradScaler
from torch.profiler import profile, record_function, ProfilerActivity

from utils import prepare_val_folder, load_dataset, count_parameters, save_model, func_timer
from convnext import ConvNext
from fastswish import FastSwish

# enable use of gpu tensor cores (only relevant in double precision)
torch.set_float32_matmul_precision('high')


@dataclass
class Params():
    epochs = 30
    lr = 0.0003
    train_batch_size = 128
    val_batch_size = 32
    weight_decay = 0.01

    num_workers = 16
    prefetch_factor = 4

    torch_compile = False
    inf_batch_size = 1
    mixed_precision = True

    artifacts_path = "./artifacts"
    dataset_path = "./tiny-imagenet-200"


def count_correct_preds(y_pred, y_true):
    pred_indices = y_pred.max(1, keepdim=True)[1]    
    count_correct = pred_indices.eq(y_true.view_as(pred_indices)).sum().double()
    return count_correct


@func_timer
def evaluate(model, val_data, params):
    model.cuda()
    model.eval()
    data_loader = data.DataLoader(val_data, batch_size=params.val_batch_size,
                                  shuffle=False, num_workers=params.num_workers,
                                  pin_memory=True)
    cnt_correct = 0
    for batch in data_loader:
        x, y = batch[0].cuda(), batch[1].cuda()
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=params.mixed_precision):
            y_pred = model(x)

        cnt_correct += count_correct_preds(y_pred, y)
    return cnt_correct/len(val_data)


def train_model(model, train_data, params):
    model.cuda()
    model.train()
    scaler = GradScaler(enabled=params.mixed_precision)
    epochs = params.epochs
    optimizer = optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    cross_entropy_loss = nn.CrossEntropyLoss()
    data_loader = data.DataLoader(train_data, batch_size=params.train_batch_size, shuffle=True,
                                  num_workers=params.num_workers, prefetch_factor=params.prefetch_factor,
                                  pin_memory=True)
    for epoch in range(epochs):
        running_loss = 0
        for i, batch in enumerate(data_loader):
            x, y = batch[0].cuda(), batch[1].cuda()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=params.mixed_precision):
                y_pred = model(x)
                loss = cross_entropy_loss(y_pred, y)

            # training step
            running_loss += loss.item()

            # reset gradient, let memory allocator handle it
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f"epoch {epoch+1}, batch {i+1}/{len(data_loader)}, loss {running_loss/(i+1)}")


def export_onnx_graph(model, params):
    x = torch.rand((1, 3, 224, 224), dtype=torch.float32)

    # regular export
    torch.onnx.export(model, x, f=str(params.artifacts_path / "model.onnx"),
                      training=torch.onnx.TrainingMode.TRAINING,
                      do_constant_folding=False,
                      export_modules_as_functions={FastSwish})
    
    # dynamo export, doesn't work with previous `export_modules_as_functions`
    # with warnings.catch_warnings(): # ignore opset 18 warning
    #    warnings.simplefilter("ignore")
    #    export_output = torch.onnx.dynamo_export(model, x)
    
    # export_output.save(str(params.artifacts_path / "model_dynamo.onnx"))


def measure_training(model, params):
    train, val = load_dataset(params.dataset_path)

    start_time = time()
    train_model(model, train, params)
    duration = time() - start_time
    print(f"Time per train epoch: {duration/params.epochs:.4f}")

    accuracy = evaluate(model, val, params)
    print(f"Classification accuracy {accuracy}")

    save_model(model, params.artifacts_path / 'model.pt')


def evaluate_inference(model, params):
    model.cuda()
    model.eval()
    input_1 = torch.rand(size=(params.inf_batch_size, 3, 224, 224), device="cuda")
    model(input_1)

    input_2 = torch.rand(size=(params.inf_batch_size, 3, 224, 224), device="cuda")
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
        with record_function("model_inference"):
                model(input_2)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(str(params.artifacts_path / "trace.json"))


def main():
    params = Params()
    params.artifacts_path = Path(params.artifacts_path)

    # prepare data and folders
    if not os.path.exists(params.artifacts_path):
        os.makedirs(params.artifacts_path)
    
    prepare_val_folder(params.dataset_path)
    
    # parepare model
    model = ConvNext(img_size=224, num_classes=200, channels=(64, 96, 128, 256))
    print(f"Number of model parameters: {count_parameters(model)/10e5:.2f} mil")

    # NOTE: We export before compilation to avoid errors. My guess is that Trition is generating
    # custom symbols and JIT compiled kernels that are not compatiable with the ONNX opset.
    # Exporting with dynamo_export probably only optimizes the graph in a ONNX-compatible way.
    export_onnx_graph(model, params)

    # jit optimization for faster inference with torch (inductor) backend
    if params.torch_compile:
        # NOTE:  Using cudagraphs lets PyTorch execute multiple kernels via one CPU submit.
        model = torch.compile(model, dynamic=True)#, options={"triton.cudagraphs": True})
       
    # train model and measure time
    measure_training(model, params)

    # evaluate inference runtime
    evaluate_inference(model, params)


if __name__ == "__main__":
    main()