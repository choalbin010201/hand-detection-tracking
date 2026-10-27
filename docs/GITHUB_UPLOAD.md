# GitHub 업로드 안내

압축을 풀고 `hand-detection-tracking` 폴더 자체를 VS Code에서 엽니다. ZIP을 그대로 올리는 대신 폴더 안의 파일들을 Git으로 올립니다. 이 안내는 별도의 새 저장소로 업로드할 때 사용하는 절차입니다. 이미 이 저장소를 이용 중이라면 아래 명령으로 복제한 후 README의 실행 절차를 따르면 됩니다.

```bash
git clone https://github.com/choalbin010201/hand-detection-tracking.git
cd hand-detection-tracking
```

## 1. 공개될 내용 확인

README의 영상 GIF에는 촬영한 손과 배경이 표시됩니다. 이를 공개해도 되는지 확인하세요. 강의 자료에서 유래한 코드의 공개 허용 여부가 불명확하면 우선 Private 저장소로 시작할 수 있습니다. 학습 데이터·가중치·제출 보고서는 `.gitignore`로 제외됩니다.

## 2. 로컬 Git 저장소 만들기

VS Code 통합 터미널이 이 프로젝트 폴더를 가리키는 상태에서 실행합니다.

```bash
python3 hand_tracking.py --self-test
git init
git branch -M main
git add .
git status --short
git diff --cached --stat
```

추가 목록에서 `README.md`, Python 코드, `.gitignore`, `.github/workflows/tests.yml`, `docs/`, `results/`를 확인하세요. `dataset/`, `runs/`, `best.pt`, 원본 동영상은 목록에 나타나지 않아야 합니다. Git 커밋에는 로컬 Git에 설정된 작성자 이름·이메일이 기록됩니다.

```bash
git commit -m "Add YOLO hand detection and IoU tracking project"
```

## 3. GitHub 빈 저장소 생성 후 연결

GitHub에서 `hand-detection-tracking` 이름으로 새 저장소를 생성합니다. README·gitignore·라이선스 초기화 항목은 선택하지 않습니다. 소개 문구 예시는 다음과 같습니다.

> Custom YOLO hand detection with IoU-based tracking, video visualization, and unit tests.

아래 `YOUR_USERNAME`을 본인 GitHub 아이디로 변경하고 실행합니다.

```bash
git remote add origin https://github.com/YOUR_USERNAME/hand-detection-tracking.git
git push -u origin main
```

인증이 필요하면 Git/VS Code의 GitHub 로그인 안내에 따라 진행합니다. 업로드 후 README의 GIF와 Actions의 테스트 결과를 확인합니다. 프로젝트 변경 사항은 이후 `git add`, `git commit`, `git push`로 반영합니다.

## 기존 과제 결과로 재실행

기존 `week7_ready/dataset`을 프로젝트 루트로 복사하면 데이터 분할을 재사용할 수 있습니다. 이미 학습한 가중치는 복사하지 않고 기존 경로를 직접 지정해도 됩니다.

```bash
python hand_tracking.py --source "/실제/경로/test.mp4" --weights "/실제/경로/best.pt"
```

학습 코드의 결과 저장 루트만 절대경로로 정리하여 `runs/detect/runs/detect`처럼 경로가 중복되는 상황을 예방했습니다. `results/`에 포함한 값은 기존 과제 실행 기록이며 이 공개용 패키지로 모델을 새로 학습한 결과가 아닙니다.
