# Audio/ML Research Repository Template

音響信号処理・機械学習研究のための公開リポジトリテンプレートです。論文・実験コード・結果・デモページを同じリポジトリで管理し、`docs/` を GitHub Pages として公開する構成を想定しています。

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/                 # 実験設定 YAML
├── data/                    # 原則として Git 管理しない
│   ├── raw/
│   └── processed/
├── src/audio_ml_project/    # Python パッケージ本体
├── scripts/                 # 実行用スクリプト
├── notebooks/               # 探索・可視化用 notebook
├── tests/                   # 最小テスト
├── results/                 # 図表・モデル出力
├── examples/                # 小さいサンプル
└── docs/                    # GitHub Pages 用デモページ
```

## Recommended Demo Policy

このテンプレートでは、デモページを **同じリポジトリの `docs/` に含める方式** を採用しています。

理由:

- 研究コード・README・図表・デモを一元管理できる
- GitHub Pages の設定が簡単
- 小〜中規模の研究プロジェクトでは最も一般的で運用しやすい
- 論文提出・ポートフォリオ公開・卒論発表ページに使いやすい

大規模な Web アプリ、バックエンド API、GPU 推論サーバーが必要な場合のみ、デモを別リポジトリに分けるのがおすすめです。

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

サンプル実験:

```bash
python scripts/run_experiment.py --config configs/baseline.yaml
```

デモページ確認:

```bash
python -m http.server 8000 --directory docs
```

ブラウザで `http://localhost:8000` を開きます。

## Research Overview

- Task: Audio classification / enhancement / event detection / representation learning
- Input: WAV, MP3, spectrogram, mel-spectrogram, MFCC, embeddings
- Model: CNN / CRNN / Transformer / classical ML baseline
- Metrics: Accuracy, F1, AUC, SNR, PESQ, STOI, inference time

## Reproducibility Checklist

- [ ] データセット名・取得方法を記載
- [ ] 前処理条件を記載
- [ ] 学習条件を `configs/` に保存
- [ ] 乱数 seed を固定
- [ ] 評価指標を明記
- [ ] 図表の生成コードを保存
- [ ] モデル重みの公開可否を明記
- [ ] ライセンスを明記

## Citation

このリポジトリを使う場合は `CITATION.cff` を編集してください。
