import argparse

def main(opt):
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu")
    print(f"Using {device} device")

    
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        default = './AIGC-Detection-Dataset/AIGC-Detection-Dataset/train',
        type=str,
        help='Directory for train dataset'
    )
    parser.add_argument(
        '--test_dir',
        default = './AIGC-Detection-Dataset/AIGC-Detection-Dataset/val',
        type=str,
        help='Directory for test dataset'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Spcifies learing rate for optimizer. (default: 1e-4)')
    parser.add_argument(
        '--path_to_checkpoint',
        type=str,
        default='model_checkpoint.ckpt',
        help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs. (default: 10)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=12,
        help='Batch size for data loaders. (default: 12)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of workers for data loader. (default: 8)'
    )
    parser.add_argument(
        '--resume',
        action="store_true",
        help='Wether to resume the training from the stored checkpoint'
    )
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    main(args)