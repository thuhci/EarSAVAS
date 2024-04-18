import os
import time
import torch
import joblib
import datetime
from pycm import *
import numpy as np
from utilities.util import *
from EarVAS_models import FocalLossMulti
from utilities.EarVAS_stat_calcu import *

def train(audio_model, train_loader, test_loader, cfg):
    device = cfg.Model.device
    if device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    global_step, epoch, best_f1_macro = 0, 0, 0.0

    Dataset_config = cfg.Dataset
    Model_config = cfg.Model

    exp_dir = Model_config.exp_dir

    if not isinstance(audio_model, nn.DataParallel) and device.type == 'cuda':
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1000000))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_trainables) / 1000000))
    trainables = audio_trainables

    optimizer = torch.optim.Adam(trainables, Model_config.lr, weight_decay=Model_config.weight_decay, betas=(0.95, 0.999))

    print('now use new scheduler')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(10, 60)), gamma=1.0)

    epoch += 1

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    audio_model.train()
    while epoch < Model_config.num_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print(datetime.datetime.now())

        for i, data in enumerate(train_loader):
            # measure data loading time
            audio_input = data[0]
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            if 'without' in Model_config.task:
                labels = data[-1].to(device, non_blocking=True) # labels are always second to last item
            else:
                labels = data[-2].to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            if 'SWITest' in Model_config.task:
                audio_output = audio_model(audio_input)
            else:
                imu_input = data[1].to(device, non_blocking=True)
                audio_output = audio_model(audio_input, imu_input)

            # full subject vocal activity
            loss_fn = FocalLossMulti(alpha = [0.35360856584308326, 0.07880689123104302, 0.13360323641582555, 
                                            0.0894736825693862, 0.1682411125236322, 0.02513590968890248, 
                                            0.10366172702597115, 0.04619501733699732, 0.0012738573651589443], gamma=2)
            loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))

            # original optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % Model_config.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (Model_config.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                'Train Loss {loss_meter.val:.4f}\t'.format(
                epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                    per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        stats, valid_loss, f1_macro, confusion_matrix = validate(audio_model, test_loader, cfg)
        print('validation finished')

        acc = np.mean([stat['acc'] for stat in stats])

        print("---------------------Epoch {:d} Results---------------------".format(epoch))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))
        print("confusion_matrix: ", confusion_matrix)

        # write the confusion matrix into txt files 
        with open(exp_dir + f'/confusion_matrix_{Model_config.task}_SAMoSA_{Model_config.samosa}.txt', 'a') as f:
            f.write(f'Validation Confusion Matrix:\n')
            f.write(str(confusion_matrix))
            f.write('\n')

        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            torch.save(audio_model.state_dict(), f"{exp_dir}/models/best_audio_model_{Model_config.task}_SAMoSA_{Model_config.samosa}.pth")

        scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

def validate(audio_model, val_loader, cfg, detail_analysis=False, label_list=None, label_dict=None):
    device = cfg.Model.device
    if device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel) and device.type == 'cuda':
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []

    if detail_analysis:
        exp_dir = cfg.Model.exp_dir
        log_dir = os.path.join(exp_dir, f'test_logs')
        os.makedirs(log_dir, exist_ok=True)

    with torch.no_grad():
        if detail_analysis:
            earsense_recognition_confusion_matrix = np.zeros((9, 9), dtype = np.int32)
            
            fp_label_dict = {label: 0 for label in label_list}
            fn_label_dict = {label: 0 for label in label_list}

            false_multi_dict = {label: {} for label in label_list if label != 'others' and 'non_subject' not in label}
            for label in false_multi_dict:
                false_multi_dict[label] = {label: 0 for label in label_list}

            valid_label_idx = [idx for idx, label in enumerate(label_list) if label != 'others' and 'non_subject' not in label]
            others_idx = label_list.index('others') 
        for i, data in enumerate(val_loader):
            audio_input = data[0].to(device)

            if detail_analysis:
                labels_raw = data[-1]

            if "SWITest" in cfg.Model.task:
                labels = data[1]
            else:
                imu_input = data[1].to(device)
                labels = data[2]

            if 'SWITest' in cfg.Model.task:
                audio_output = audio_model(audio_input)
            else:
                audio_output = audio_model(audio_input, imu_input)
            
            predictions = nn.Softmax(dim=1)(audio_output).to('cpu').detach()
            
            A_predictions.append(predictions)
            A_targets.append(labels)

            if detail_analysis:
                index_predictions = torch.argmax(predictions, dim = 1)
                raw_labels = torch.argmax(labels, dim = 1)
                fp_indexes = np.where((labels_raw > 7) & (index_predictions <= 7))[0]
                fn_indexes = np.where((labels_raw <= 7) & (index_predictions > 7))[0]
                
                raw_raw_labels = labels_raw.cpu().numpy()
                fp_raw_labels = raw_raw_labels[fp_indexes]
                fn_raw_labels = raw_raw_labels[fn_indexes]
                label_dict_reverse = {idx: label for label, idx in label_dict.items()}
                fp_raw_labels = [label_dict_reverse[label] for label in fp_raw_labels]
                fn_raw_labels = [label_dict_reverse[label] for label in fn_raw_labels]

                for label in label_list:
                    fp_label_dict[label] += fp_raw_labels.count(label)
                    fn_label_dict[label] += fn_raw_labels.count(label)

                for valid_idx in valid_label_idx:
                    valid_label = label_dict_reverse[valid_idx]
                    fp_indexes = np.where(((raw_labels == others_idx) & (index_predictions == valid_idx)))[0]
                    fp_raw_labels = raw_raw_labels[fp_indexes]
                    fp_raw_labels = [label_dict_reverse[label] for label in fp_raw_labels]
                    for label in label_list:
                        false_multi_dict[valid_label][label] += fp_raw_labels.count(label)

                np.add.at(earsense_recognition_confusion_matrix, (raw_labels, index_predictions), 1)

            labels = labels.to(device)
            loss_fn = FocalLossMulti(alpha=[0.35360856584308326, 0.07880689123104302, 0.13360323641582555, 0.0894736825693862,
                                            0.1682411125236322, 0.02513590968890248, 0.10366172702597115, 0.04619501733699732,
                                            0.0012738573651589443], gamma=2)
            loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1)).to('cpu').detach()
            A_loss.append(loss)

            batch_time.update(time.time() - end)
            end = time.time()

        if detail_analysis:
            with open(os.path.join(log_dir, f'{cfg.Model.task}_SAMoSA_{cfg.Model.samosa}_results.txt'), 'w') as f:
                f.write(f"fp_label_dict: {fp_label_dict}\n")
                f.write(f"fn_label_dict: {fn_label_dict}\n")

                f.write("False Multi Detection:\n")
                for label in false_multi_dict:
                    f.write(label)
                    f.write(str(false_multi_dict[label]))
                    f.write('\n')
            
                f.write("Testing Confusion Matrix:\n")
                f.write(str(earsense_recognition_confusion_matrix))
                f.write('\n')

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        audio_output_cm = torch.argmax(audio_output, dim=1)
        target_cm = torch.argmax(target, dim=1)
        cm = ConfusionMatrix(actual_vector=target_cm.cpu().numpy(), predict_vector=audio_output_cm.cpu().numpy())
        f1_macro = cm.F1_Macro
        loss = np.mean(A_loss)
        stats, confusion_matrix = calculate_stats(audio_output, target)

    return stats, loss, f1_macro, confusion_matrix