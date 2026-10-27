"""최초에는 학습 후 추적, 이후에는 저장된 학습 모델로 추적."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(command, log_name):
    with open(log_name, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = proc.wait()
    if code:
        raise SystemExit(code)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='test.mp4')
    p.add_argument('--device', default='auto', choices=['auto', 'mps', 'cpu'])
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--retrain', action='store_true')
    p.add_argument('--show', action='store_true')
    args = p.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if not Path(args.source).is_file():
        raise SystemExit('Put your video in this folder as test.mp4 or specify --source "video.mov".')
    run([sys.executable, '-u', 'hand_tracking.py', '--self-test'], 'bbox_tracker_tests.txt')
    trained = False
    if Path('latest_training.json').is_file():
        metadata = json.loads(Path('latest_training.json').read_text())
        trained = Path(metadata['best_weights']).is_file()
    if args.retrain or not trained:
        run([sys.executable, '-u', 'train_hand.py', '--device', args.device, '--batch', str(args.batch)],
            'train_console.txt')
    command = [sys.executable, '-u', 'hand_tracking.py', '--source', args.source, '--device', args.device]
    if args.show:
        command.append('--show')
    run(command, 'tracking_console.txt')


if __name__ == '__main__':
    main()
