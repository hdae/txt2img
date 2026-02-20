# モデルと設定

## 対応モデル

| タイプ           | 説明                                                     | プリセット例       | LoRA |
| :--------------- | :------------------------------------------------------- | :----------------- | :--- |
| **SDXL**         | SDXL とその派生(Illustrious等)。タグベースのプロンプト。 | `sdxl/illustrious` | ✅   |
| **Chroma**       | Flux派生軽量モデル(8.9B)。自然言語のプロンプト。         | `chroma/flash`     | ❌   |
| **Flux Dev**     | Flux.1 [dev]。高品質、約50ステップ。                     | `flux/dev`         | ❌   |
| **Flux Schnell** | Flux.1 [schnell]。高速、4ステップ。                      | `flux/schnell`     | ❌   |
| **Z-Image**      | Z-Image Turbo。高速6Bモデル、8ステップ。                 | `zimage/turbo`     | ❌   |
| **Anima**        | Anima Diffusionモデル。タグベースのプロンプト。          | `anima/preview`    | ❌   |

### 固定パラメータ

品質とパフォーマンスを最適化するため、以下のパラメータはモデルごとに固定されており、APIからは変更できません。

| モデル           | Steps | CFG Scale | Sampler |
| :--------------- | :---- | :-------- | :------ |
| **SDXL**         | 20    | 7.0       | euler_a |
| **Chroma**       | 4     | 4.0       | -       |
| **Flux Dev**     | 50    | 3.5       | -       |
| **Flux Schnell** | 4     | 0.0       | -       |
| **Z-Image**      | 8     | 0.0       | -       |
| **Anima**        | 32    | 4.0       | -       |

## VRAM プロファイル

`.env` 内の `VRAM_PROFILE` 環境変数で設定します。

| プロファイル   | VRAM目安 | 最適化戦略                                                        |
| :------------- | :------- | :---------------------------------------------------------------- |
| **`full`**     | 24GB+    | オフロードなし。最高速。モデルをVRAMに常駐。                      |
| **`balanced`** | 12-16GB  | CPUオフロード + VAEタイリング。バランス重視。                     |
| **`lowvram`**  | 8GB      | 順次CPUオフロード (streaming) + VAEタイリング。低速だが省メモリ。 |

## LoRA 設定 (SDXL)

プリセットJSONまたはリクエストボディでLoRAを設定します。

- **`ref`**: Civitai AIR URN (`urn:air:sdxl:lora:civitai:ID@VERSION`) または
  URL。
- **`triggers`**: トリガーワードのリスト。省略時はCivitaiから自動取得されます。
- **`weight`**: 全体的な適用強度 (0.0 - 2.0)。デフォルト: 1.0。
- **`trigger_weight`**: トリガーワードの埋め込み強度。デフォルト: 0.5。
