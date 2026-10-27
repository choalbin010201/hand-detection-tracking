"""PDF의 YOLO26n fine-tuning; Mac용 batch/worker 설정."""
import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset/data.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
    args = parser.parse_args()
    import os
    import yaml
    os.chdir(Path(__file__).resolve().parent)
    import torch
    import ultralytics
    from ultralytics import YOLO
    data = Path(args.data).resolve()
    if not data.is_file():
        raise FileNotFoundError('Run prepare_dataset.py first: ' + str(data))
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    config = yaml.safe_load(data.read_text(encoding='utf-8'))
    config['path'] = str(data.parent)
    runtime_data = data.parent / 'data_runtime.yaml'
    runtime_data.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    data = runtime_data
    model = YOLO('yolo26n.pt')
    model.train(data=str(data), epochs=args.epochs, batch=args.batch, imgsz=640,
                device=device, workers=0, seed=42, cache=False, amp=False,
                project=str(Path('runs/detect').resolve()), name='hand_train', exist_ok=False, plots=True)
    save_dir = Path(model.trainer.save_dir).resolve()
    info = dict(created=datetime.now().isoformat(), data=str(data), device=device,
                torch_version=torch.__version__, ultralytics_version=ultralytics.__version__,
                epochs_requested=args.epochs, batch=args.batch, imgsz=640,
                best_weights=str(save_dir / 'weights/best.pt'), output_dir=str(save_dir))
    Path('latest_training.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print('\nTraining output:', save_dir)
    print('Best weights:', info['best_weights'])
    print('If this was not the first run, pass this path with --weights when tracking.')


if __name__ == '__main__':
    main()
