"""按 XML object/name 将完整图片和原始标注复制到各类别目录，并生成 CSV。"""

import argparse
import hashlib
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

from count_labels import BASE, Progress, add_arguments, check_paths, load_data, print_summary, write_reports


def folder_name(label):
    """兼容 Windows 路径；不安全名称使用摘要，避免替换后撞名。"""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", label).rstrip(" .")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)}
    if safe != label or not safe or len(safe) > 80 or safe.split(".")[0].upper() in reserved:
        return "label_" + hashlib.sha256(label.encode("utf-8")).hexdigest()
    return safe


def split_samples(source, output, samples, overwrite=False):
    check_paths(source.resolve(), output.resolve())
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not output.is_dir() or not overwrite:
            raise ValueError(f"分类输出目录非空，请指定新的 --output 或使用 --overwrite 重建分类结果：{output}")
    if overwrite:
        category_root = output / "by_label"
        # 删除前检查解析后的绝对路径，不能通过链接删除输出目录之外的内容。
        if category_root.resolve() != output.resolve() / "by_label":
            raise ValueError(f"分类目录指向其他位置，拒绝清理：{category_root}")
        if category_root.exists():
            print(f"[清理旧结果] 正在删除：{category_root}", flush=True)
            shutil.rmtree(category_root)
    output.mkdir(parents=True, exist_ok=True)
    labels = sorted({label for s in samples for label in s.counts})
    print(f"[分类目录] 正在创建 {len(labels)} 个类别目录", flush=True)
    folders, used, copied = {}, set(), Counter()
    for label in labels:
        name = folder_name(label)
        # Windows 不区分大小写，确保不同标签不会合并。
        if name.casefold() in used:
            name += "_" + hashlib.sha256(label.encode("utf-8")).hexdigest()
        while name.casefold() in used:
            name += "_"
        used.add(name.casefold())
        folders[label] = "by_label/" + name
        (output / folders[label]).mkdir(parents=True)
    used_stems = {label: set() for label in labels}
    copy_total = sum(len(sample.counts) for sample in samples if sample.image is not None)
    progress = Progress("复制图片/XML 对", copy_total)
    completed = 0
    for sample in samples:
        if sample.image is None:
            continue
        relative = sample.xml.relative_to(source)
        for label in sample.counts:
            target = output / folders[label]
            # 文件直接放在类别下；跨批次同名时为图片/XML 添加相同后缀。
            stem = sample.xml.stem
            if stem.casefold() in used_stems[label]:
                digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:10]
                stem = f"{sample.xml.stem}__{digest}"
                index = 2
                while stem.casefold() in used_stems[label]:
                    stem = f"{sample.xml.stem}__{digest}_{index}"
                    index += 1
            used_stems[label].add(stem.casefold())
            image_target = target / (stem + sample.image.suffix)
            xml_target = target / (stem + ".xml")
            if image_target.exists() or xml_target.exists():
                raise ValueError(f"目标文件名冲突：{xml_target}")
            shutil.copy2(sample.image, image_target)
            shutil.copy2(sample.xml, xml_target)
            copied[label] += 1
            completed += 1
            progress.update(completed)
    return folders, copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, BASE / "datasets" / "classified")
    parser.add_argument("--overwrite", action="store_true", help="删除输出中的旧 by_label 目录后重新分类，并更新统计 CSV")
    args = parser.parse_args()
    try:
        source, output, mapping, samples, issues, totals = load_data(args)
        folders, copied = split_samples(source, output, samples, overwrite=args.overwrite)
        write_reports(output, source, mapping, samples, issues, totals, folders, copied)
    except (OSError, ValueError, ET.ParseError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"错误：{exc}\n")
    print_summary(output, totals)
    print(f"分类复制完成：{sum(copied.values())} 组图片/XML，目录：{output / 'by_label'}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
