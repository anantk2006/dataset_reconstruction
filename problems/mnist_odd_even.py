import torch
import torchvision.datasets
import torchvision.transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import random


def load_bound_dataset(dataset, batch_size, shuffle=False, start=None, end=None, **kwargs):
    def _bound_dataset(dataset, start, end):
        if start is None:
            start = 0
        if end is None:
            end = len(dataset)
        return torch.utils.data.Subset(dataset, range(start, end))

    dataset = _bound_dataset(dataset, start, end)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, **kwargs)


def fetch_mnist(root, train=False, transform=None, target_transform=None):
    transform = transform if transform is not None else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root, train=train, transform=transform, target_transform=target_transform, download=True)
    return dataset


def load_mnist(root, batch_size, train=False, transform=None, target_transform=None, **kwargs):
    dataset = fetch_mnist(root, train, transform, target_transform)
    return load_bound_dataset(dataset, batch_size, **kwargs)


def move_to_type_device(x, y, device):
    print(device)
    print('X:', x.shape)
    print('y:', y.shape)
    x = x.to(torch.get_default_dtype())
    y = y.to(torch.get_default_dtype())
    
    x, y = x.to(device), y.to(device)
    return x, y


def create_labels(y0):
    labels_dict = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1}
    y0 = torch.stack([torch.tensor(labels_dict[int(cur_y)]) for cur_y in y0])
    return y0


def get_balanced_data(args, data_loader, data_amount):
    print('BALANCING DATASET...')
    # get balanced data
    data_amount_per_class = data_amount // 2

    labels_counter = {1: 0, 0: 0}
    x0, y0 = [], []
    got_enough = False
    for bx, by in data_loader:
        by = create_labels(by)
        for i in range(len(bx)):
            if labels_counter[int(by[i])] < data_amount_per_class:
                labels_counter[int(by[i])] += 1
                x0.append(bx[i])
                y0.append(by[i])
            if (labels_counter[0] >= data_amount_per_class) and (labels_counter[1] >= data_amount_per_class):
                got_enough = True
                break
        if got_enough:
            break
    x0, y0 = torch.stack(x0), torch.stack(y0)
    return x0, y0


def load_mnist_data(args):
    # Get Train Set
    data_loader = load_mnist(root=args.datasets_dir, batch_size=100, train=True, shuffle=True, start=0, end=50000)
    x0, y0 = get_balanced_data(args, data_loader, args.data_amount)

    # Get Test Set
    print('LOADING TESTSET')
    assert not args.data_use_test or (args.data_use_test and args.data_test_amount >= 2), f"args.data_use_test={args.data_use_test} but args.data_test_amount={args.data_test_amount}"
    data_loader = load_mnist(root=args.datasets_dir, batch_size=100, train=False, shuffle=True, start=0, end=10000)
    x0_test, y0_test = get_balanced_data(args, data_loader, args.data_test_amount)

    # move to cuda and double
    x0, y0 = move_to_type_device(x0, y0, args.device)
    x0_test, y0_test = move_to_type_device(x0_test, y0_test, args.device)

    print(f'BALANCE: 0: {y0[y0 == 0].shape[0]}, 1: {y0[y0 == 1].shape[0]}')

    return [(x0, y0)], [(x0_test, y0_test)], None


def get_data(dataset_name, dataroot, batch_size, val_ratio, world_size, rank, args, heterogeneity=0, num_workers=1, small=False, b01 = False):
    """
    Args:
        dataset_name (str): the name of the dataset to use, currently only
            supports 'MNIST', 'FashionMNIST', 'CIFAR10' and 'CIFAR100'.
        dataroor (str): the location to save the dataset.
        batch_size (int): batch size used in training.
        val_ratio (float): the percentage of trainng data used as validation.
        world_size (int): how many processed will be used in training.
        rank (int): the rank of this process.
        heterogeneity (float): dissimilarity between data distribution across clients.
            Between 0 and 1.
        small (bool): Whether to use miniature dataset.
    Outputs:
        iterators over training, validation, and test data.
    """
    if ((val_ratio < 0) or (val_ratio > 1.0)):
        raise ValueError("[!] val_ratio should be in the range [0, 1].")
    if heterogeneity < 0:
        raise ValueError("Data heterogeneity must be positive.")
    if world_size == 1 and heterogeneity > 0:
        raise ValueError("Cannot create a heterogeneous dataset when world_size == 1.")

    # Mean and std are obtained for each channel from all training images.
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))
        num_labels = 10
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize((0.5071, 0.4866, 0.4409),
                                         (0.2673, 0.2564, 0.2762))
        num_labels = 100
    elif dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        num_labels = 2 if not args.multi_class else 10
    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST
        normalize = transforms.Normalize((0.2860,), (0.3530,))
        num_labels = 2
    else:
        raise NotImplementedError

    if dataset_name.startswith('CIFAR'):
        # Follows Lee et al. Deeply supervised nets. 2014.
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              normalize])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             normalize])
    elif dataset_name in ['MNIST', 'FashionMNIST']:
        if args.model_type=="res":
            transform_train =  transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])
            transform_test = transform_train
        transform_train =  transforms.Compose([transforms.ToTensor()])
        transform_test = transform_train

    # load and split the train dataset into train and validation and 
    # deployed to all GPUs.
    if not b01 and not args.multi_class: 
        ttransform = lambda x: x%2
        ttransform = transforms.Lambda(ttransform)
    else: ttransform=None 
    
    train_set = dataset(root=dataroot, train=True,
                        download=True, transform=transform_train, target_transform=ttransform)
    val_set = dataset(root=dataroot, train=True,
                      download=True, transform=transform_test, target_transform=ttransform)
    test_set = dataset(root=dataroot, train=False,
                       download=True, transform=transform_test, target_transform=ttransform)

    # partition the training data into multiple GPUs if needed. Data partitioning to
    # create heterogeneity is performed according to the specifications in
    # https://arxiv.org/abs/1910.06378.
    if world_size > 1:
        random.seed(1234)
    train_data_len = len(train_set)
    train_label_idxs = get_label_indices(dataset_name, train_set, num_labels)
    label_proportions = torch.tensor([float(len(train_label_idxs[i])) for i in range(num_labels)])
    label_proportions /= torch.sum(label_proportions)
    for l in range(num_labels):
        random.shuffle(train_label_idxs[l])
    worker_idxs = [[] for _ in range(world_size)]

    # Divide samples from each label into iid pool and non-iid pool. Note that samples
    # in iid pool are shuffled while samples in non-iid pool are sorted by label.
    iid_pool = []
    non_iid_pool = []
    for i in range(num_labels):
        iid_split = round((1.0 - heterogeneity) * len(train_label_idxs[i]))
        iid_pool += train_label_idxs[i][:iid_split]
        non_iid_pool += train_label_idxs[i][iid_split:]
    random.shuffle(iid_pool)

    # Allocate iid and non-iid samples to each worker.
    num_iid = len(iid_pool) // world_size
    num_non_iid = len(non_iid_pool) // world_size
    partition_size = num_iid + num_non_iid
    for j in range(world_size):
        worker_idxs[j] += iid_pool[num_iid * j: num_iid * (j+1)]
        worker_idxs[j] += non_iid_pool[num_non_iid * j: num_non_iid * (j+1)]
        random.shuffle(worker_idxs[j])

    # Split training set into training and validation for current worker.
    val_split = int(val_ratio * partition_size)
    local_train_idx = worker_idxs[rank][val_split:]
    local_valid_idx = worker_idxs[rank][:val_split]

    # Check that each worker dataset is disjoint. This is slow, so only comment this out
    # for testing.
    #"""
    # for i in range(world_size):
    #     current_idxs = worker_idxs[i]
    #     other_idxs = []
    #     for j in range(world_size):
    #         if j == i:
    #             continue
    #         other_idxs += worker_idxs[j]
    #     for idx in current_idxs:
    #         assert idx not in other_idxs
    #"""

    # Get indices of local test dataset.
    test_partition = len(test_set) // world_size
    test_idxs = list(range(test_partition * world_size))
    random.shuffle(test_idxs)
    local_test_idx = test_idxs[rank * test_partition: (rank+1) * test_partition]

    # Use miniature dataset, if necessary.
    if small:
        local_train_idx = local_train_idx[:round(len(local_train_idx) / 100)]
        local_valid_idx = local_valid_idx[:round(len(local_valid_idx) / 100)]
        local_test_idx = local_test_idx[:round(len(local_test_idx) / 100)]

    # Construct loaders for train, valid, extra, and test sets.
   
    if b01:
        index = 0
        for i in [*local_train_idx]:
            X, Y = train_set[i]
            Y = int(Y)
            if Y != 0 and Y != 1:
                
                local_train_idx.pop(index)
            else: index+=1
        index = 0 
        for i in [*local_valid_idx]:
            X, Y = val_set[i]
            Y = int(Y)
            if Y != 0 and Y != 1:
                local_valid_idx.pop(index)
            else: index +=1
        index = 0 
        for i in [*local_test_idx]:
            X, Y = test_set[i]
            Y = int(Y)
            if Y != 0 and Y != 1:
                local_test_idx.pop(index)
            else: index +=1

                
            
    

    train_sampler = SubsetRandomSampler(local_train_idx)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_sampler = SubsetRandomSampler(local_valid_idx)
    valid_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_sampler = SubsetRandomSampler(local_test_idx)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Reformat data to match the structure expected by reconstruction code.
    train_list_x = []
    train_list_y = []
    max_batch = args.data_amount//batch_size
    for i, (x,y) in enumerate(train_loader):
        
        if i==max_batch: break
        train_list_x.append(x)
        train_list_y.append(y)
    train_loader = [(torch.cat(train_list_x, dim = 0).to(args.device), torch.cat(train_list_y, dim = 0).to(args.device).to(torch.float32))]

    test_list_x = []
    test_list_y = []
    max_batch = args.data_test_amount//batch_size
    for i, (x,y) in enumerate(test_loader):
        if i==max_batch: break
        print(y)
        test_list_x.append(x)
        test_list_y.append(y)
    test_loader = [(torch.cat(test_list_x, dim = 0).to(args.device), torch.cat(test_list_y, dim = 0).to(args.device).to(torch.float32))]

    return train_loader, test_loader, None


def get_label_indices(dataset_name, dset, num_labels):
    """
    Returns a dictionary mapping each label to a list of the indices of elements in
    `dset` with the corresponding label.
    """

    label_indices = [[] for _ in range(num_labels)]
    if dataset_name in ["CIFAR10", "CIFAR100", "MNIST"]:
        for idx, label in enumerate(dset.targets):
            label_indices[label%num_labels].append(idx)
    else:
        raise NotImplementedError
    
    return label_indices


def get_dataloader(args):
    args.input_dim = 28 * 28
    args.num_classes = 2
    args.output_dim = 1
    args.dataset = 'mnist'

    if args.run_mode == 'reconstruct':
        args.extraction_data_amount = args.extraction_data_amount_per_class * args.num_classes

    # for legacy:
    args.data_amount = args.data_per_class_train * args.num_classes
    args.data_use_test = True
    args.data_test_amount = 1000

    #data_loader = load_mnist_data(args)
    data_loader = get_data("MNIST", "data", args.data_amount, 0, args.num_clients, args.rank, args, heterogeneity=args.heterogeneity, b01 = args.two_classes)
    y0 = data_loader[0][0][1]
    print(f'BALANCE: 0: {y0[y0 == 0].shape[0]}, 1: {y0[y0 == 1].shape[0]}')
    
    return data_loader
