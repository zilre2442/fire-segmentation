"""使用 PlotNeuralNet 生成模型结构示意图的工具脚本。

该脚本默认在脚本同级目录寻找 PlotNeuralNet 源码：models/visual/PlotNeuralNet。
若不存在，会提示自动克隆或使用 --auto-clone 进行自动克隆。

功能概览
--------
* 支持 baseline、RGS_Net_V1、RGS_Net_V2 三种模型结构的可视化；
* 输出 .tex 文件到每个模型独立的新目录，并可选编译为 .pdf / .png；
* 依赖 LaTeX (pdflatex) 编译，若未安装会自动跳过编译，仅保留 .tex。

示例:
    # 自动克隆 + 生成 baseline 的 TeX
    python models/visual/visualize_plotneuralnet.py --model baseline --auto-clone

    # 已存在仓库时（无需 --auto-clone）并编译为 PDF 和 PNG
    python models/visual/visualize_plotneuralnet.py --model RGS_Net_V1 --compile pdf png

    # 一次性为三个模型全部生成与编译
                python models/visual/visualize_plotneuralnet.py --model all --compile pdf png
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List
import importlib


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PNN_DIR = os.path.join(SCRIPT_DIR, "PlotNeuralNet")


def ensure_plotneuralnet(auto_clone: bool = False) -> None:
    """确保 PlotNeuralNet 可用；支持自动克隆。"""
    if os.path.isdir(PNN_DIR):
        return
    if not auto_clone:
        raise RuntimeError(
            f"未找到 PlotNeuralNet 目录: {PNN_DIR}\n"
            "请手动放置或使用 --auto-clone 自动克隆。"
        )

    print("[info] 正在克隆 PlotNeuralNet...")
    import urllib.request
    import zipfile
    import io

    # 直接下载 v1.0 的镜像 zip（避免 git 依赖）；若失败请手动 git clone。
    url = "https://codeload.github.com/HarisIqbal88/PlotNeuralNet/zip/refs/heads/master"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            tmp_root = os.path.join(SCRIPT_DIR, "_pnn_tmp")
            if os.path.exists(tmp_root):
                shutil.rmtree(tmp_root)
            os.makedirs(tmp_root, exist_ok=True)
            zf.extractall(tmp_root)
            # 顶层目录名类似 PlotNeuralNet-master
            [extracted_dir] = [
                os.path.join(tmp_root, d)
                for d in os.listdir(tmp_root)
                if os.path.isdir(os.path.join(tmp_root, d))
            ]
            shutil.move(extracted_dir, PNN_DIR)
            shutil.rmtree(tmp_root, ignore_errors=True)
        print("[ok] PlotNeuralNet 克隆完成:", PNN_DIR)
    except Exception as e:
        raise RuntimeError(
            "自动下载 PlotNeuralNet 失败，请手动 git clone 到 models/visual/PlotNeuralNet：\n"
            "  git clone https://github.com/HarisIqbal88/PlotNeuralNet.git"
        ) from e


def have_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def setup_import_path() -> None:
    # 使得 `from pycore.* import *` 可用
    if PNN_DIR not in sys.path:
        sys.path.append(PNN_DIR)


def _tikzeng():
    setup_import_path()
    return importlib.import_module("pycore.tikzeng")


def _blocks():
    setup_import_path()
    return importlib.import_module("pycore.blocks")


def _common_preamble(project_path: str) -> List[str]:
    """构建文档头, project_path 应指向包含 layers/ 的 PlotNeuralNet 目录相对路径。"""
    tikz = _tikzeng()
    return [tikz.to_head(project_path), tikz.to_cor(), tikz.to_begin()]


def _common_finish() -> List[str]:
    tikz = _tikzeng()
    return [tikz.to_end()]


def build_arch_baseline(project_path: str) -> List[str]:
    """构建 baseline U-Net 的 tikz 结构。"""
    tikz = _tikzeng()
    blocks = _blocks()

    arch: List[str] = []
    arch += _common_preamble(project_path)

    # Encoder
    # 输入节点（示意），放在首层左侧
    arch += [
        tikz.to_Conv(name="input", s_filer=512, n_filer=3, offset="(-3,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption="Input"),
    ]
    arch += [
        tikz.to_ConvConvRelu(name="ccr_b1", s_filer=512, n_filer=(16, 16), offset="(0,0,0)", to="(0,0,0)", width=(2, 2), height=40, depth=40),
        tikz.to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=1, height=32, depth=32, opacity=0.5),
    ]
    arch += [tikz.to_connection("input", "ccr_b1")]
    arch += [*blocks.block_2ConvPool(name="b2", botton="pool_b1", top="pool_b2", s_filer=256, n_filer=32, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5)]
    arch += [*blocks.block_2ConvPool(name="b3", botton="pool_b2", top="pool_b3", s_filer=128, n_filer=64, offset="(1,0,0)", size=(25, 25, 4.5), opacity=0.5)]
    arch += [*blocks.block_2ConvPool(name="b4", botton="pool_b3", top="pool_b4", s_filer=64, n_filer=128, offset="(1,0,0)", size=(16, 16, 5.5), opacity=0.5)]

    # Bottleneck
    arch += [
        tikz.to_ConvConvRelu(name="ccr_b5", s_filer=32, n_filer=(256, 256), offset="(2,0,0)", to="(pool_b4-east)", width=(8, 8), height=8, depth=8, caption="Bottleneck"),
        tikz.to_connection("pool_b4", "ccr_b5"),
    ]

    # Decoder (seg)
    arch += [*blocks.block_Unconv(name="b6", botton="ccr_b5", top="end_b6", s_filer=64, n_filer=128, offset="(2.1,0,0)", size=(16, 16, 5.0), opacity=0.5)]
    arch += [*blocks.block_Unconv(name="b7", botton="end_b6", top="end_b7", s_filer=128, n_filer=64, offset="(2.1,0,0)", size=(25, 25, 4.5), opacity=0.5)]
    arch += [*blocks.block_Unconv(name="b8", botton="end_b7", top="end_b8", s_filer=256, n_filer=32, offset="(2.1,0,0)", size=(32, 32, 3.5), opacity=0.5)]
    arch += [*blocks.block_Unconv(name="b9", botton="end_b8", top="end_b9", s_filer=512, n_filer=16, offset="(2.1,0,0)", size=(40, 40, 2.5), opacity=0.5)]
    # 显式绘制跳跃连接连线标记（encoder -> decoder）
    arch += [
        tikz.to_skip(of="ccr_b4", to="ccr_res_b6", pos=1.25),
        tikz.to_skip(of="ccr_b3", to="ccr_res_b7", pos=1.25),
        tikz.to_skip(of="ccr_b2", to="ccr_res_b8", pos=1.25),
        tikz.to_skip(of="ccr_b1", to="ccr_res_b9", pos=1.25),
    ]
    # 输出
    arch += [tikz.to_ConvSoftMax(name="Seg", s_filer=512, offset="(0.75,0,0)", to="(end_b9-east)", width=1, height=40, depth=40, caption="Sigmoid")]
    arch += [tikz.to_connection("end_b9", "Seg")]

    arch += _common_finish()
    return arch


def build_arch_dual_decoder(caption_left: str, caption_right: str, project_path: str, show_err_fusion: bool = False) -> List[str]:
    """构建共享编码器 + 双解码器结构（用于 RGS_Net_V1 及变体）。

    参数:
    - caption_left: 分割分支输出标题
    - caption_right: 重建分支输出标题
    - project_path: PlotNeuralNet 的相对路径（用于 header 包含）
    - show_err_fusion: 若为 True，则在各解码阶段可视化“重建误差权重与分割特征的融合（乘性 gating）”，用于 V2。
    """
    tikz = _tikzeng()
    blocks = _blocks()

    arch: List[str] = []
    arch += _common_preamble(project_path)
    # 为重建分支定义单独的连线样式与箭头（绿色），并将跨越连接从“下方”改为“上方”路由。
    arch += [
        "% custom style for reconstruction skips (green, routed from above)\n"
        "\\tikzstyle{recconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:green,6;blue,1;black,3},opacity=0.7]\n"
        "\\newcommand{\\recopymidarrow}{\\tikz \\draw[-Stealth,line width=0.8mm,draw={rgb:green,6;blue,1;black,3}] (-0.3,0) -- ++(0.3,0);} \n"
    ]
    if show_err_fusion:
        # 误差权重融合（V2）可视化的虚线与箭头样式（橙色）
        arch += [
            "% error-weight fusion (orange dashed) for V2\n"
            "\\tikzstyle{errconnection}=[ultra thick,dashed,every node/.style={sloped,allow upside down},draw={rgb:red,5;yellow,5;black,2},opacity=0.8]\n"
            "\\newcommand{\\errmidarrow}{\\tikz \\draw[-Stealth,line width=0.6mm,draw={rgb:red,5;yellow,5;black,2}] (-0.25,0) -- ++(0.25,0);} \n"
        ]

    # 输入节点（示意）
    arch += [
        tikz.to_Conv(name="input", s_filer=512, n_filer=3, offset="(-3,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption="Input"),
    ]
    # Encoder 共用
    arch += [
        tikz.to_ConvConvRelu(name="ccr_b1", s_filer=512, n_filer=(32, 32), offset="(0,0,0)", to="(0,0,0)", width=(2, 2), height=40, depth=40),
        tikz.to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=1, height=32, depth=32, opacity=0.5),
    ]
    arch += [tikz.to_connection("input", "ccr_b1")]
    arch += [*blocks.block_2ConvPool(name="b2", botton="pool_b1", top="pool_b2", s_filer=256, n_filer=64, offset="(1,0,0)", size=(32, 32, 3.5), opacity=0.5)]
    arch += [*blocks.block_2ConvPool(name="b3", botton="pool_b2", top="pool_b3", s_filer=128, n_filer=128, offset="(1,0,0)", size=(25, 25, 4.5), opacity=0.5)]
    arch += [*blocks.block_2ConvPool(name="b4", botton="pool_b3", top="pool_b4", s_filer=64, n_filer=256, offset="(1,0,0)", size=(16, 16, 5.5), opacity=0.5)]
    arch += [
        tikz.to_ConvConvRelu(name="ccr_b5", s_filer=32, n_filer=(512, 512), offset="(2,0,0)", to="(pool_b4-east)", width=(8, 8), height=8, depth=8, caption="Bottleneck"),
        tikz.to_connection("pool_b4", "ccr_b5"),
    ]

    # Decoder A（上：分割）
    # 使用 blocks.Unconv 需要的 Unpool/ConvRes/Conv 序列在 blocks.py 里，但为了分叉布局，这里手搓几个模块并连 skip。
    # 为简单与稳健，沿用单分支的 Unet 示例布局，通过偏移 y 控制上下两个分支的相对位置。

    # 分支1: Seg (y 偏移 0)
    arch += [*blocks.block_Unconv(name="seg_b6", botton="ccr_b5", top="seg_end_b6", s_filer=64, n_filer=256, offset="(2.1,0,0)", size=(16, 16, 5.0), opacity=0.5)]
    arch += [*blocks.block_Unconv(name="seg_b7", botton="seg_end_b6", top="seg_end_b7", s_filer=128, n_filer=128, offset="(2.1,0,0)", size=(25, 25, 4.5), opacity=0.5)]
    arch += [*blocks.block_Unconv(name="seg_b8", botton="seg_end_b7", top="seg_end_b8", s_filer=256, n_filer=64, offset="(2.1,0,0)", size=(32, 32, 3.5), opacity=0.5)]
    arch += [*blocks.block_Unconv(name="seg_b9", botton="seg_end_b8", top="seg_end_b9", s_filer=512, n_filer=32, offset="(2.1,0,0)", size=(40, 40, 2.5), opacity=0.5)]
    arch += [tikz.to_ConvSoftMax(name="Seg", s_filer=512, offset="(0.75,0,0)", to="(seg_end_b9-east)", width=1, height=40, depth=40, caption=caption_left)]
    arch += [tikz.to_connection("seg_end_b9", "Seg")]

    # 添加 skip 到分割分支
    arch += [
        tikz.to_skip(of="ccr_b4", to="ccr_res_seg_b6", pos=1.25),
        tikz.to_skip(of="ccr_b3", to="ccr_res_seg_b7", pos=1.25),
        tikz.to_skip(of="ccr_b2", to="ccr_res_seg_b8", pos=1.25),
        tikz.to_skip(of="ccr_b1", to="ccr_res_seg_b9", pos=1.25),
    ]

    # 分支2: Recon (y 方向向下位移)
    # 通过相对较大的 x 偏移, 将 Recon 分支布局在 Seg 输出之后，避免遮挡；同时在 y 轴向下轻微偏移。
    # 注意：PlotNeuralNet 的 API 以绝对锚点为主，我们复用 seg_end_b9 的 east 作为 recon 开始参考点。
    # 在 TikZ3D 里暂不直接支持 y 偏移连接 skip，因此我们仅做平行的第二条解码路径可视化，不额外再画第二组 skip（避免线缠绕）。

    # 简化 Recon 解码器：4 个上采样块 + 最终 1x1 输出 (3 通道)
    # 重新用 block_Unconv 铺设一条路径
    arch += [*blocks.block_Unconv(name="rec_b6", botton="ccr_b5", top="rec_end_b6", s_filer=64, n_filer=256, offset="(2.1,-8,0)", size=(16, 16, 5.0), opacity=0.35)]
    arch += [*blocks.block_Unconv(name="rec_b7", botton="rec_end_b6", top="rec_end_b7", s_filer=128, n_filer=128, offset="(2.1,0,0)", size=(25, 25, 4.5), opacity=0.35)]
    arch += [*blocks.block_Unconv(name="rec_b8", botton="rec_end_b7", top="rec_end_b8", s_filer=256, n_filer=64, offset="(2.1,0,0)", size=(32, 32, 3.5), opacity=0.35)]
    arch += [*blocks.block_Unconv(name="rec_b9", botton="rec_end_b8", top="rec_end_b9", s_filer=512, n_filer=32, offset="(2.1,0,0)", size=(40, 40, 2.5), opacity=0.35)]
    arch += [tikz.to_Conv(name="Recon", s_filer=512, n_filer=3, offset="(0.75,0,0)", to="(rec_end_b9-east)", width=1, height=40, depth=40, caption=caption_right)]
    arch += [tikz.to_connection("rec_end_b9", "Recon")]

    # 为重建分支添加与编码器对应的跳跃连接（encoder -> reconstruction decoder），
    # 使用“横平竖直”的风格且从上方连接（绿色样式与分割分支区分）
    # e4->rec_b6, e3->rec_b7, e2->rec_b8, e1->rec_b9
    arch += [
    r"""
\path (ccr_b4-northwest) -- (ccr_b4-northeast) coordinate[pos=1.25] (ccr_b4-top) ;
\path (ccr_res_rec_b6-north)  -- (ccr_res_rec_b6-south)  coordinate[pos=-0.25] (ccr_res_rec_b6-top) ;
\draw [recconnection]  (ccr_b4-northeast)
-- node {\recopymidarrow}(ccr_b4-top)
-- node {\recopymidarrow}(ccr_res_rec_b6-top)
-- node {\recopymidarrow} (ccr_res_rec_b6-north);
""",
    r"""
\path (ccr_b3-northwest) -- (ccr_b3-northeast) coordinate[pos=1.25] (ccr_b3-top) ;
\path (ccr_res_rec_b7-north)  -- (ccr_res_rec_b7-south)  coordinate[pos=-0.25] (ccr_res_rec_b7-top) ;
\draw [recconnection]  (ccr_b3-northeast)
-- node {\recopymidarrow}(ccr_b3-top)
-- node {\recopymidarrow}(ccr_res_rec_b7-top)
-- node {\recopymidarrow} (ccr_res_rec_b7-north);
""",
    r"""
\path (ccr_b2-northwest) -- (ccr_b2-northeast) coordinate[pos=1.25] (ccr_b2-top) ;
\path (ccr_res_rec_b8-north)  -- (ccr_res_rec_b8-south)  coordinate[pos=-0.25] (ccr_res_rec_b8-top) ;
\draw [recconnection]  (ccr_b2-northeast)
-- node {\recopymidarrow}(ccr_b2-top)
-- node {\recopymidarrow}(ccr_res_rec_b8-top)
-- node {\recopymidarrow} (ccr_res_rec_b8-north);
""",
    r"""
\path (ccr_b1-northwest) -- (ccr_b1-northeast) coordinate[pos=1.25] (ccr_b1-top) ;
\path (ccr_res_rec_b9-north)  -- (ccr_res_rec_b9-south)  coordinate[pos=-0.25] (ccr_res_rec_b9-top) ;
\draw [recconnection]  (ccr_b1-northeast)
-- node {\recopymidarrow}(ccr_b1-top)
-- node {\recopymidarrow}(ccr_res_rec_b9-top)
-- node {\recopymidarrow} (ccr_res_rec_b9-north);
""",
    ]

    # 若需要，添加 V2 的“误差权重融合”可视化（以橙色虚线从重建分支阶段输出到分割分支对应阶段的融合处）。
    if show_err_fusion:
        arch += [
        r"""
% stage b6: rec_end_b6 -> ccr_res_seg_b6 (err weight fusion)
\draw [errconnection] (rec_end_b6-north)
to[out=90,in=90,looseness=0.8] node {\errmidarrow} (ccr_res_seg_b6-north);
""",
        r"""
% stage b7
\draw [errconnection] (rec_end_b7-north)
to[out=90,in=90,looseness=0.8] node {\errmidarrow} (ccr_res_seg_b7-north);
""",
        r"""
% stage b8
\draw [errconnection] (rec_end_b8-north)
to[out=90,in=90,looseness=0.8] node {\errmidarrow} (ccr_res_seg_b8-north);
""",
        r"""
% stage b9
\draw [errconnection] (rec_end_b9-north)
to[out=90,in=90,looseness=0.8] node {\errmidarrow} (ccr_res_seg_b9-north);
""",
        ]

    arch += _common_finish()
    return arch


def write_tex(arch: List[str], out_tex: str) -> None:
    tikz = _tikzeng()
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    tikz.to_generate(arch, out_tex)


def compile_tex(tex_path: str, open_file: bool = False) -> str | None:
    """使用 pdflatex 编译 tex -> pdf；若失败返回 None。"""
    if not have_cmd("pdflatex"):
        print("[warn] 未检测到 pdflatex，跳过编译：", tex_path)
        return None
    work_dir = os.path.dirname(tex_path)
    base = os.path.splitext(os.path.basename(tex_path))[0]
    pdf_path = os.path.join(work_dir, f"{base}.pdf")
    cmd = ["pdflatex", "-interaction=nonstopmode", os.path.basename(tex_path)]
    try:
        print("[info] 编译 LaTeX -> PDF:", tex_path)
        subprocess.run(cmd, cwd=work_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # 清理中间文件
        for ext in (".aux", ".log", ".vscodeLog"):
            f = os.path.join(work_dir, base + ext)
            if os.path.exists(f):
                os.remove(f)
        if open_file and have_cmd("xdg-open"):
            subprocess.Popen(["xdg-open", pdf_path])
        return pdf_path
    except subprocess.CalledProcessError as e:
        print("[error] pdflatex 编译失败，保留 .tex 供手动处理。\n", e)
        return None


def maybe_convert_png(pdf_path: str) -> str | None:
    """尝试将 PDF 转为 PNG（可选）。需要 `convert` 或 `pdftocairo`。"""
    if pdf_path is None:
        return None
    out_png = os.path.splitext(pdf_path)[0] + ".png"
    if have_cmd("pdftocairo"):
        cmd = ["pdftocairo", "-png", "-singlefile", pdf_path, os.path.splitext(out_png)[0]]
    elif have_cmd("convert"):
        cmd = ["convert", "-density", "200", pdf_path, "-quality", "90", out_png]
    else:
        print("[warn] 未找到 pdftocairo/convert，跳过 PNG 转换。")
        return None
    try:
        print("[info] 导出 PNG:", out_png)
        subprocess.run(cmd, check=True)
        return out_png
    except subprocess.CalledProcessError:
        print("[warn] PNG 转换失败。")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="PlotNeuralNet 可视化生成器")
    parser.add_argument("--model", choices=["baseline", "RGS_Net_V1", "RGS_Net_V2", "all"], default="baseline")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "diagrams"), help="输出根目录。会在其下为每个模型创建独立目录")
    parser.add_argument("--compile", nargs="*", choices=["pdf", "png"], default=[], help="是否编译导出 pdf / png")
    parser.add_argument("--auto-clone", action="store_true", help="未找到 PlotNeuralNet 时自动克隆")
    parser.add_argument("--open", action="store_true", help="完成后尝试打开 PDF")
    args = parser.parse_args()

    ensure_plotneuralnet(auto_clone=args.auto_clone)
    setup_import_path()

    targets = [args.model] if args.model != "all" else ["baseline", "RGS_Net_V1", "RGS_Net_V2"]

    results = []
    for name in targets:
        subdir = os.path.join(args.output_dir, name)
        os.makedirs(subdir, exist_ok=True)

        # 固定文件名，覆盖旧文件
        tex_file = os.path.join(subdir, f"{name}.tex")

        # 计算 tex 输出目录到 PlotNeuralNet 目录的相对路径, 供 to_head 使用
        project_path = os.path.relpath(PNN_DIR, start=subdir)

        if name == "baseline":
            arch = build_arch_baseline(project_path)
        elif name == "RGS_Net_V2":
            # V2: 双解码器 + 分割分支受“重建误差权重”引导（在各阶段进行融合）
            arch = build_arch_dual_decoder(
                caption_left="Seg (guided)",
                caption_right="Recon",
                project_path=project_path,
                show_err_fusion=True,
            )
        else:
            # RGS_Net_V1：双解码器结构
            arch = build_arch_dual_decoder(
                caption_left="Seg",
                caption_right="Recon",
                project_path=project_path,
                show_err_fusion=False,
            )

        write_tex(arch, tex_file)
        print(f"[ok] 写出 TeX: {tex_file}")

        pdf_path = None
        png_path = None
        if "pdf" in args.compile:
            pdf_path = compile_tex(tex_file, open_file=args.open)
        if "png" in args.compile:
            # 若没编译过 PDF，则先尝试编译
            if pdf_path is None:
                pdf_path = compile_tex(tex_file, open_file=False)
            png_path = maybe_convert_png(pdf_path)

        results.append((name, tex_file, pdf_path, png_path))

    print("\n[summary] 可视化完成：")
    for (name, tex_file, pdf_path, png_path) in results:
        print(f"- {name} -> tex: {tex_file}")
        if pdf_path:
            print(f"           pdf: {pdf_path}")
        if png_path:
            print(f"           png: {png_path}")


if __name__ == "__main__":
    main()


