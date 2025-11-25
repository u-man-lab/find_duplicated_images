# find_duplicated_images

* NOTE: See [`README.md`](./README.md) for English README.

## 概要

このリポジトリには、複製された写真ファイルを削除するのに役立つ2つのPythonスクリプトが含まれています。  
1. 1つ目のスクリプト、[`group_file_paths_list_by_its_name.py`](#1-group_file_paths_list_by_its_namepy)は、ファイル名の類似性に基づいて写真ファイルのパスをグループ化します。
1. 2つ目のスクリプト、[`find_duplicated_images.py`](#2-find_duplicated_imagespy)は、事前に定義されたグループ内の写真ファイルを分析し、複製された写真ファイルを検出します（回転やリサイズも考慮します）。

### 免責事項
開発者は、これらのスクリプトによって生成された複製画像リストに基づいてファイルを削除した場合に、いかなる問題が生じたとしても責任を負いません。  
ファイル削除などの不可逆的な措置を講じる前に、生成されたCSV出力を手動で確認し、検出された複製画像のうちいくつかが確かに元の画像のコピーであることを視覚的に確認することを強く推奨します。

---

## ライセンス & 開発者

- **ライセンス**: このリポジトリ内の[`LICENSE`](./LICENSE)を参照してください。
- **開発者**: U-MAN Lab. ([https://u-man-lab.com/](https://u-man-lab.com/))

★このリポジトリのスクリプトについて、以下の記事で解説しています。★  
[[Python] 大量の写真ファイルから複製写真のみをピックアップする。 | U-MAN Lab.](https://u-man-lab.com/find-duplicated-images-by-python/?utm_source=github&utm_medium=social&utm_campaign=find_duplicated_images)

---

## 1. `group_file_paths_list_by_its_name.py`

### 1.1. 概要

[`group_file_paths_list_by_its_name.py`](./group_file_paths_list_by_its_name.py)は、ファイルパスを含むCSVファイルを読み込んで各ファイル名を取得し、ファイル名が一部共通しているファイルをグループ化します。

スクリプトはCSVに以下の列を追加します。:

- 割り当てられたグループ名
- 取得したファイル名

---

### 1.2. インストールと使用方法

#### (1) Pythonをインストールする

[公式サイト](https://www.python.org/downloads/)を参照してPythonをインストールしてください。  
開発者が検証したバージョンより古い場合、スクリプトが正常に動作しない可能性があります。[`.python-version`](./.python-version)を参照してください。

#### (2) リポジトリをクローンする

```bash
git clone https://github.com/u-man-lab/find_duplicated_images.git
# gitコマンドを利用できない場合は、別の方法でスクリプトファイルとYAML設定ファイルを環境に配置してください。
cd ./find_duplicated_images
```

#### (3) Pythonライブラリをインストールする

開発者が検証したバージョンより古い場合、スクリプトが正常に動作しない可能性があります。
```bash
pip install --upgrade pip
pip install -r ./requirements.txt
```

#### (4) 入力用のCSVファイルを用意する

実行中のPC内に存在する写真・動画ファイルのパスが記載されたCSVを用意します。  
CSVがない場合は、以下のような方法で作成します。
```bash
TARGET_FOLDER='<対象ファイルが格納されているフォルダ>'
find "$TARGET_FOLDER" -type f > ./data/file_paths_list.csv
sed -i '1s/^/file_paths\n/' ./data/file_paths_list.csv  # 列ヘッダー追加
```

#### (5) 設定ファイルを編集する

設定ファイル[`configs/group_file_paths_list_by_its_name.yaml`](./configs/group_file_paths_list_by_its_name.yaml)を開き、ファイル内のコメントに従って値を編集します。

#### (6) スクリプトを実行する

```bash
python ./group_file_paths_list_by_its_name.py ./configs/group_file_paths_list_by_its_name.yaml
```

---

### 1.3. 期待される出力

成功した場合、標準エラー出力(stderr)に次のようなログが出力されます。:

```
2025-08-14 13:56:47,965 [INFO] __main__: "group_file_paths_list_by_its_name.py" start!
2025-08-14 13:56:48,007 [INFO] __main__: Reading CSV file "data/file_paths_list.csv"...
2025-08-14 13:56:48,635 [INFO] __main__: Now grouping...
2025-08-14 14:00:22,192 [INFO] __main__: Filtering to only grouped files...
2025-08-14 14:00:22,504 [INFO] __main__: Writing CSV file "results/file_paths_list_grouped_by_its_name.csv"...
2025-08-14 14:00:23,134 [INFO] __main__: "group_file_paths_list_by_its_name.py" done!
```
参考までに、33,753ファイルの処理に約3分40秒かかりました。Synology DS218（Realtek RTD1296、4コア 1.4 GHz、2GB DDR4）に、2×4TB WD Red（SMR）をRAID1で構成したNAS上で実行しました。

生成されるCSVは次のような形式になります。:

```
file_paths,group,file_name
/path1/animal928-img600x380-1320574493ftqrpo80714.jpg,animal928-img600x380-1320574493ftqrpo80714,animal928-img600x380-1320574493ftqrpo80714.jpg
/path2/animal928-img600x380-1320574493ftqrpo80714.jpg,animal928-img600x380-1320574493ftqrpo80714,animal928-img600x380-1320574493ftqrpo80714.jpg
:
```

---

### 1.4. よくあるエラー

詳細については、スクリプトのソースコードを参照してください。よくあるエラーには以下のものが含まれます。:

- **スクリプトに引数を渡していない**
  ```
  2025-08-13 09:46:05,471 [ERROR] __main__: This script needs a config file path as an arg.
  ```
- **設定ファイルの値がおかしい**
  ```
  2025-08-13 09:47:40,930 [ERROR] __main__: Failed to parse the config file.: "configs\group_file_paths_list_by_its_name.yaml"
  Traceback (most recent call last):
  :
  ```

---

## 2. `find_duplicated_images.py`

### 2.1. 概要

[`find_duplicated_images.py`](./find_duplicated_images.py)は、グループ化された写真ファイルのパスを含むCSVファイルを読み込み、各グループ内で複製写真を検出します。  
「perceptual hash」を使用して、写真の回転・リサイズにも対応しています。

出力には以下の内容が含まれます。:
- **CSVファイル**: 入力されたCSVファイルに以下の列が追加されます。  
   - 重複ID: 各グループ内の重複写真のグループID。  
   - 「not-oldest」マーク: 各重複グループ内でファイルが最古でないかどうか。（最古のファイルは重複写真の中のオリジナル写真であると考えられます。）重複グループ内で最古のファイルが複数ある場合は、入力されたCSVファイルにおいて最も上にあるファイルが最古であるとみなされます。
- **TXTファイル**: オリジナル写真とみなされない複製写真のファイルパス一覧。CSVファイルの「not-oldest」マークが付いているファイルパス一覧と全く同一です。

「not-oldest」マークを付けるために、写真ファイルの撮影日時（文字列ではなくUNIXなどの連続値）が必要です。  
撮影日時情報がなければ、このスクリプトを実行する前に、以下のリポジトリにある`extract_image_taken_datetime.py`を実行することで取得できます。  
https://github.com/u-man-lab/extract_image_taken_datetime/tree/main

また、本スクリプトにより取得したTXTファイル（複製写真のファイルパス一覧）を元に、複製写真を一括削除する場合は、以下のリポジトリにある`move_target_files_into_a_folder.py`をぜひ活用ください。複製写真をまとめて一時フォルダに移動でき、最終確認しながら削除することができます。  
https://github.com/u-man-lab/move_target_files_into_a_folder/tree/main

---

### 2.2. 実行方法

実行するために、以下の準備が完了していることを確認してください。:

- Pythonをインストールする
- リポジトリをクローンする
- Pythonライブラリをインストールする

（詳細は[1章](#1-group_file_paths_list_by_its_namepy)を参照してください。）

#### (1) 入力用のCSVファイルを用意する

実行中のPC内に存在する写真・動画ファイルのパスが記載されたCSVを用意します。  
また、ファイルのグループと連続値の撮影日時が格納された列も必要です。

#### (2) 設定ファイルを編集する

設定ファイル[`configs/find_duplicated_images.yaml`](./configs/find_duplicated_images.yaml)を開き、ファイル内のコメントに従って値を編集してください。

#### (3) スクリプトを実行する

```bash
python ./find_duplicated_images.py ./configs/find_duplicated_images.yaml
```

---

### 2.3. 期待される出力

成功した場合、標準エラー出力(stderr)に次のようなログが出力されます。:

```
2025-08-13 23:45:57,760 [INFO] __main__: "find_duplicated_images.py" start!
2025-08-13 23:45:57,773 [INFO] __main__: Reading file "results/file_paths_list_grouped_by_its_name.csv"...
2025-08-13 23:46:02,195 [INFO] __main__: 6910 groups found.
2025-08-13 23:46:02,201 [INFO] __main__: Processing [1/6910]: group "20100823071358"...
2025-08-13 23:46:05,715 [INFO] __main__: Processing [2/6910]: group "20100823071411"...
:
2025-08-14 02:56:25,575 [INFO] __main__: Processing [6910/6910]: group "_DSC0257"...
2025-08-14 02:57:11,720 [INFO] __main__: Writing CSV file "results/file_paths_list_grouped_by_its_name_with_duplicated_id.csv"...
2025-08-14 02:57:13,670 [INFO] __main__: Writing TXT file "results/not_oldest_in_duplicated_file_paths_list.txt"...
2025-08-14 02:57:13,676 [INFO] __main__: "find_duplicated_images.py" done!
```
参考までに、6,910グループに分割された15,956ファイルの処理に約3時間11分かかりました。Synology DS218（Realtek RTD1296、4コア 1.4 GHz、2GB DDR4）に、2×4TB WD Red（SMR）をRAID1で構成したNAS上で実行しました。

生成されるCSVファイルは次のような形式になります。:

```
file_paths,datetime_local_unix,group,file_name,duplicated_id,not_oldest
/path1/animal928-img600x380-1320574493ftqrpo80714.jpg,1330865697.000000,animal928-img600x380-1320574493ftqrpo80714,animal928-img600x380-1320574493ftqrpo80714.jpg,1,X
/path2/animal928-img600x380-1320574493ftqrpo80714.jpg,1330865688.000000,animal928-img600x380-1320574493ftqrpo80714,animal928-img600x380-1320574493ftqrpo80714.jpg,1,
:
```

生成されるTXTファイルは次のような形式になります。:

```
/path1/animal928-img600x380-1320574493ftqrpo80714.jpg
/path1/penticton_river_channel_sc0292.jpg
:
```

---

### 2.4. よくあるエラー

詳細については、スクリプトのソースコードを参照してください。よくあるエラーは、[`group_file_paths_list_by_its_name.py`](#1-group_file_paths_list_by_its_namepy)と同様です。
