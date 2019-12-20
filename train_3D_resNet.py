if __name__ ==  '__main__':

    import torch
    import matplotlib.pyplot as plt
    from skimage import io
    import os
    import numpy as np
    from random import seed
    from random import randint
    import uuid

    from model_defs.ResNet_3DConv import ResNet_3DConv
    from utils.system_utils import mkdir_p
    from utils.plot_utils import get_fig_2d_array_of_images, plot_loss_lines
    from dataset_loader.dataset_reader_v1 import DatasetReaderV1
    from torch.utils.data import DataLoader

    DATASET_PATH = "E:/gkopanas/museum_front_new/gkopanas_dataset"
    BATCH_SIZE = 5
    RESIDUAL_BLOCKS = 8
    IN_CHANNELS = 32
    INTERNAL_DEPTH = 32
    KERNEL_SIZE = 3
    STRIDE = 1
    PADDING = 1
    BIAS = False
    BATCH_NORM = False
    LEARNING_RATE = 0.001
    TRAINING_EPOCHS = 5000
    CONVOLUTION_TYPE = "3D"

    def crop_center(img, cropx, cropy):
        y,x,c = img.shape
        startx = x//2 - cropx//2
        starty = y//2 - cropy//2
        return img[starty:starty+cropy, startx:startx+cropx, :]


    def getInputBatchFromFolder(path):
        batch_volume = None
        batch_gt = None
        batch_mask = None
        for folder in os.listdir(path):
            np_volume = None
            for filename in os.listdir(os.path.join(path, folder)):
                if filename.startswith("title_cam"):
                    image = io.imread(os.path.join(path, folder, filename))
                    image = crop_center(image, 256, 256)/255
                    image = np.transpose(image, (2, 0, 1))
                    image = np.expand_dims(image, 0)
                    image = np.expand_dims(image, 0)
                    if np_volume is None:
                        np_volume = np.array(image)
                    else:
                        np_volume = np.concatenate((np_volume, image), axis=1)
                if filename.startswith("gt_"):
                    image = io.imread(os.path.join(path, folder, filename))
                    image = crop_center(image, 256, 256)/255
                    image = np.transpose(image, (2, 0, 1))
                    image = np.expand_dims(image, 0)
                    image = np.expand_dims(image, 0)
                    if batch_gt is None:
                        batch_gt = np.array(image)
                    else:
                        batch_gt = np.concatenate((batch_gt, image), axis=0)
                if filename.startswith("title_mask"):
                    image = io.imread(os.path.join(path, folder, filename))
                    image = crop_center(image, 256, 256)/255
                    image = np.transpose(image, (2, 0, 1))
                    image = np.expand_dims(image, 0)
                    image = np.expand_dims(image, 0)
                    if batch_mask is None:
                        batch_mask = np.array(image)
                    else:
                        batch_mask = np.concatenate((batch_mask, image), axis=0)
            if batch_volume is None:
                batch_volume = np.array(np_volume)
            else:
                batch_volume = np.concatenate((batch_volume, np_volume), axis=0)
        return torch.from_numpy(batch_volume), torch.from_numpy(batch_gt), torch.from_numpy(batch_mask)

    device = torch.device('cuda')

    model = ResNet_3DConv(in_channels=IN_CHANNELS, internal_depth=INTERNAL_DEPTH,
                          blocks=RESIDUAL_BLOCKS, kernel_size=KERNEL_SIZE, stride=STRIDE,
                          padding=PADDING, bias=BIAS, batch_norm=BATCH_NORM).to(device)
    print(model)

    # Prepare Output Folders
    unique_str = str(uuid.uuid4())
    figures_unique_folder = "./output_figures/{}".format(unique_str)
    model_unique_folder = "./models/{}".format(unique_str)
    mkdir_p(figures_unique_folder)
    mkdir_p(model_unique_folder)
    print("Output Folders:")
    print(" * {}".format(figures_unique_folder))
    print(" * {}".format(model_unique_folder))

    seed(1)

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True, "num_workers": 12}
    training_dataset = DatasetReaderV1(os.path.join(DATASET_PATH, "train"))
    training_generator = DataLoader(training_dataset, **params)

    """
    train_batch_volume_cpu, train_batch_gt_cpu, train_batch_mask_cpu = getInputBatchFromFolder(os.path.join(DATASET_PATH, "train"))
    train_batch_volume = train_batch_volume_cpu.to(device, dtype=torch.float32)
    train_batch_gt = train_batch_gt_cpu.to(device, dtype=torch.float32)
    train_batch_mask = train_batch_mask_cpu.to(device, dtype=torch.float32)
    """
    test_batch_volume_cpu, test_batch_gt_cpu, test_batch_mask_cpu = getInputBatchFromFolder(os.path.join(DATASET_PATH, "test"))
    test_batch_volume = test_batch_volume_cpu.to(device, dtype=torch.float32)
    test_batch_gt = test_batch_gt_cpu.to(device, dtype=torch.float32)
    test_batch_mask = test_batch_mask_cpu.to(device, dtype=torch.float32)



    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_loss_history = []
    train_ulr_loss_history = []
    test_loss_history = []
    test_ulr_loss_history = []
    i=0
    for idx in range(1, TRAINING_EPOCHS):
        for batch_n, batch in enumerate(training_generator):
            i += 1
            train_batch_volume_cpu = batch["image_volume"]
            train_batch_volume = train_batch_volume_cpu.to(device, dtype=torch.float32)
            train_batch_gt_cpu = batch["gt_render"]
            train_batch_gt = train_batch_gt_cpu.to(device, dtype=torch.float32)
            train_batch_mask_cpu = batch["gt_mask"]
            train_batch_mask = train_batch_mask_cpu.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(train_batch_volume)
            train_loss = (torch.abs((outputs - train_batch_gt))*train_batch_mask).mean()
            train_loss_history.append((i, train_loss.item()))

            train_ulr_loss = (torch.abs((outputs - train_batch_volume[:, 0, :, :, :]))*train_batch_mask).mean()
            train_ulr_loss_history.append((i, train_ulr_loss.item()))

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            print('Train Step [{}/{}], Loss: {:.4f} ULR Loss:{:.4f}'
                  .format(i, TRAINING_EPOCHS, train_loss.item(), train_ulr_loss.item()))

            # Train Dataset Progress Visualisation
            if i % 10 == 0:
                mat_of_images = []
                outputs_cpu = outputs.detach().cpu().clone().numpy()
                for j in range(max(outputs_cpu.shape[0], 5)):
                    r = randint(0, outputs_cpu.shape[0] - 1)
                    a = outputs_cpu[r, 0, :, :, :].transpose((1, 2, 0))
                    ulr = train_batch_volume_cpu.numpy()[r, 0, :, :, :].transpose((1, 2, 0))
                    diff_with_ulr = (ulr - a) / (ulr - a).max()
                    gt = train_batch_gt_cpu.numpy()[r, 0, :, :, :].transpose((1, 2, 0))
                    mat_of_images.append([a, ulr, diff_with_ulr, gt])
                fig = plt.figure()
                fig = get_fig_2d_array_of_images(fig, mat_of_images)
                plt.savefig("./{}/iteration_{}.png".format(figures_unique_folder, i))
                plt.close(fig)

            # Test Dataset Progress Visualisation
            if i % 10 == 0:
                torch.save(model, "./{}/model_{}".format(model_unique_folder, i))
                outputs = model(test_batch_volume)

                test_loss = (torch.abs((outputs - test_batch_gt)) * test_batch_mask).mean()
                test_loss_history.append((i, test_loss.item()))

                test_ulr_loss = (torch.abs((outputs - test_batch_volume[:, 0, :, :, :])) * test_batch_mask).mean()
                test_ulr_loss_history.append((i, test_ulr_loss.item()))

                print('Testing Step [{}/{}], Loss: {:.4f} ULR Loss {:.4f}'
                      .format(i, TRAINING_EPOCHS, test_loss.item(), test_ulr_loss.item()))

                num_of_test_images = 3
                mat_of_images = []
                for j in range(num_of_test_images):
                    a = outputs.detach().cpu().clone().numpy()[j,0, :, :, :].transpose((1, 2, 0))
                    ulr = test_batch_volume.detach().cpu().clone().numpy()[j, 0, :, :, :].transpose((1, 2, 0))
                    diff_with_ulr = (ulr - a) / (ulr - a).max()
                    gt = test_batch_gt.detach().cpu().clone().numpy()[j, 0, :, :, :].transpose((1, 2, 0))
                    mat_of_images.append([a, ulr, diff_with_ulr, gt])
                fig = plt.figure()
                fig = get_fig_2d_array_of_images(fig, mat_of_images)
                plt.savefig("./{}/iteration_{}.png".format(model_unique_folder, i))
                plt.close(fig)

            # Loss Visualisation
            if i % 200 == 0:
                fig = plt.figure()
                plt.legend(['train_loss_history', 'train_ulr_loss_history',
                            'test_loss_history', 'test_ulr_loss_history'], loc='upper left')
                plot_loss_lines([train_loss_history, train_ulr_loss_history,
                                 test_loss_history, test_ulr_loss_history])
                plt.savefig("./{}/loss_{}.png".format(model_unique_folder, i))
                plt.close(fig)


    print('Finished Training')