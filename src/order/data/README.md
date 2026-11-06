# 本地题库与词库来源

本目录不含在线识别服务。模型沿用项目已有本机 ONNX 权重；此次没有新增模型权重或调用模型 API。

- `idioms.txt`：从 [pwxcoo/chinese-xinhua/data/idiom.json](https://github.com/pwxcoo/chinese-xinhua/blob/master/data/idiom.json) 提取纯汉字、长度 4–8 的词头并去重，30,212 条。源文件 Git blob `c285f951f9aeba6303cb23821e96e9b62ed343f9`，下载后按 Git blob 算法校验；上游 MIT 许可保存在 `LICENSE.idioms`。
- `poetry.txt`：从 [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) 的 `蒙学/tangshisanbaishou.json`、`蒙学/qianjiashi.json` 提取句内 4–8 字的诗行，保留繁体并用系统 OpenCC `t2s` 加入简体，4,239 条原文、合计 8,075 条去重条目。两文件 blob 分别为 `d2f2e5b01722669b7280b4e016691b083dc94468`、`2e03d7afe8ccc40d13c8c983e1a1b84447edc0e0`。上游 MIT 许可保存在 `LICENSE.poetry`。提取脚本及原始数据保留在 `work/second-batch-20261023/`，运行识别器不依赖 OpenCC 或网络。
- [`../../rotate_single/data/`](../../rotate_single/data/)：预先选取 scikit-image v0.25.2 的 `astronaut.png`、`camera.png`、`chelsea.png`、`rocket.jpg` 四种正常朝向场景。来自其官方测试数据，原始下载链接和 SHA256 见 [`../../rotate_single/data/manifest.json`](../../rotate_single/data/manifest.json)。下载均与其 [`_registry.py`](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_registry.py) SHA256 一致。按 [`_fetchers.py`](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_fetchers.py) 说明：astronaut 为 NASA 公共领域图，camera 为 Lav Varshney 的 CC0 图，chelsea 为 Stefan van der Walt 的 CC0 图，rocket 为 SpaceX 发布的公共领域图。模型评估前按人物/猫/发射塔正常朝向选择，未依据模型预测筛选。

自建图像旋转题在服务端中心裁方、缩放至 300×300，以 Pillow 逆时针随机旋转并施加圆形遮罩；浏览器仅得到最终题图，不得到原图、来源名或角度。四张图的随机旋转不能视为大量独立自然场景，也不能确认这些公开测试照片从未出现在原模型训练集中。

自建文字题独立用 Pillow 和系统 Noto CJK 黑体/宋体生成。默认字体目录 `/usr/share/fonts/noto-cjk`，可以用 `CAPTCHA_LAB_FONT_DIR` 指定。字体题覆盖空心与实心变体；语序题从代码中 40 个常见成语生成，识别器搜索上述完整词库，不接收当题选词。词库存在同字异序词、同字多位置，识别器会弃权；这些题仍保留在评测分母。
