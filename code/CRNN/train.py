import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss
from dataset import ic15Dataset, synth90k_collate_fn
from model import CRNN
from config import train_config as config
from torch.utils.tensorboard import SummaryWriter
import math
from evaluate import evaluate
import os

# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    epochs = config['epochs']
    train_batch_size = config['train_batch_size']

    lr = config['lr']
    show_interval = config['show_interval']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_width = config['img_width']
    img_height = config['img_height']
    data_dir = config['data_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_dataset = ic15Dataset(root_dir=data_dir,image_dir=data_dir, mode='train',
                                    img_height=img_height, img_width=img_width)
    test_dataset = ic15Dataset(root_dir=data_dir,image_dir=data_dir, mode='test',
                                    img_height=img_height, img_width=img_width)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=synth90k_collate_fn)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=synth90k_collate_fn)
    

    num_class = len(ic15Dataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)
    
    
    writer = SummaryWriter(log_dir='./code/CRNN/runs/exp1')
    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)
    best_val_acc = 0
    assert save_interval % valid_interval == 0 or valid_interval % save_interval ==0
    i = 1
    for epoch in range(0, epochs ):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0

        crnn.train()  # Set the model to training mode
        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)

            # 检查是否出现 inf 或 NaN
            if math.isinf(loss) or math.isnan(loss):
                print(f"Warning: Loss is invalid (inf or NaN) at iteration {i}. Skipping this batch.")
                continue

            train_size = train_data[0].size(0)
            tot_train_loss += loss
            tot_train_count += train_size

            # 记录训练损失到 TensorBoard
            writer.add_scalar('Training Loss', loss / train_size, i)

            if i % show_interval == 0:
                print(f'train_batch_loss[{i}]: ', loss / train_size)

            if i % save_interval == 0:
                save_model_path = os.path.join(config["checkpoints_dir"], f"{epoch}-crnn.pt")
                torch.save(crnn.state_dict(), save_model_path)
                print(f'save model at {save_model_path}')

            i += 1

        if epoch % 5 == 0:
                # 在每个验证间隔后评估模型
                eval_result = evaluate(crnn, test_loader, criterion,
                                       max_iter=None, decode_method='beam_search', beam_size=10)

                print(f"Validation loss: {eval_result['loss']:.4f}, accuracy: {eval_result['acc']:.4f}")
                writer.add_scalar('Validation Loss', eval_result['loss'], i)
                writer.add_scalar('Validation Accuracy', eval_result['acc'], i)

                # Early stopping logic: if the validation accuracy does not improve, stop training
                if eval_result['acc'] > best_val_acc:
                    best_val_acc = eval_result['acc']
                    save_model_path = os.path.join(config["checkpoints_dir"], "best-crnn.pt")
                    torch.save(crnn.state_dict(), save_model_path)
                    print(f'save model at {save_model_path}')
                

        print(f'train_loss: {tot_train_loss / tot_train_count}')
    writer.close()

if __name__ == '__main__':
    main()
