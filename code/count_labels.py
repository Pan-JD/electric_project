"""统计 Pascal VOC XML 标签；仅使用 Python 标准库，兼容 Python 3.9+。"""

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import monotonic


BASE = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


class Progress:
    """每隔约 2 秒打印一次，阶段开始和完成时立即打印。"""

    def __init__(self, title, total=None):
        self.title = title
        self.total = total
        self.started = self.last_print = monotonic()
        self.show(0)

    def show(self, done):
        elapsed = monotonic() - self.started
        if self.total is None:
            detail = f"已发现 {done} 个文件"
        elif self.total == 0:
            detail = "0/0（无待处理项）"
        else:
            detail = f"{done}/{self.total}（{done / self.total:.1%}）"
        print(f"[{self.title}] {detail}，耗时 {elapsed:.1f} 秒", flush=True)
        self.last_print = monotonic()

    def update(self, done):
        if done == self.total or monotonic() - self.last_print >= 2:
            self.show(done)


def track(items, title, total=None):
    progress = Progress(title, total)
    count = 0
    for count, item in enumerate(items, 1):
        yield item
        progress.update(count)
    if total is None:
        print(f"[{title}] 扫描完成，共 {count} 个文件，耗时 {monotonic() - progress.started:.1f} 秒", flush=True)


@dataclass
class Sample:
    xml: Path
    image: object  # Path 或 None
    counts: Counter


def read_label_table(path):
    """读取参考表的标注编码、标注标签、业务标签列，支持共享/内联字符串。"""
    labels = {}
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = [
                "".join(node.text or "" for node in item.findall(".//s:t", NS))
                for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)
            ]
        for name in sorted(archive.namelist()):
            if not re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name):
                continue
            columns = None
            for row in ET.fromstring(archive.read(name)).findall("s:sheetData/s:row", NS):
                cells = {}
                for cell in row.findall("s:c", NS):
                    value = cell.findtext("s:v", "", NS)
                    if cell.get("t") == "s":
                        value = strings[int(value)]
                    elif cell.get("t") == "inlineStr":
                        value = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
                    cells[re.sub(r"\d", "", cell.get("r", ""))] = value.strip()
                if "标注编码" in cells.values() and "标注标签" in cells.values():
                    columns = {value: col for col, value in cells.items()}
                    continue
                if columns is None:
                    continue
                code = cells.get(columns["标注编码"], "")
                if not code:
                    continue
                info = (cells.get(columns["标注标签"], ""), cells.get(columns.get("业务标签"), ""))
                if code in labels and labels[code] != info:
                    raise ValueError(f"参考表存在冲突的标注编码：{code}")
                labels[code] = info
    if not labels:
        raise ValueError("参考表中未找到含有“标注编码”和“标注标签”的数据")
    return labels


def scan_samples(source):
    files = sorted(track((p for p in source.rglob("*") if p.is_file()), "扫描文件"),
                   key=lambda p: str(p).casefold())
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    xmls = [p for p in files if p.suffix.lower() == ".xml"]
    print(f"[扫描结果] 图片 {len(images)} 个，XML {len(xmls)} 个", flush=True)
    by_stem, by_name = defaultdict(list), defaultdict(list)
    for image in track(images, "建立图片索引", len(images)):
        by_stem[(image.parent, image.stem.casefold())].append(image)
        by_name[(image.parent, image.name.casefold())].append(image)
    issues, samples, paired = [], [], set()
    totals = Counter(xml_files=len(xmls), image_files=len(images))

    def issue(path, kind, detail):
        issues.append([path.relative_to(source).as_posix(), kind, detail])

    for image in track(images, "检查空图片", len(images)):
        if image.stat().st_size == 0:
            issue(image, "空图片", "文件大小为 0，不作为可用图片")
    for xml in track(xmls, "解析 XML", len(xmls)):
        try:
            root = ET.parse(xml).getroot()
            if root.tag != "annotation":
                raise ValueError(f"预期根节点 annotation，实际为 {root.tag}")
        except (ET.ParseError, OSError, ValueError) as exc:
            totals["invalid_xml"] += 1
            issue(xml, "XML解析失败", str(exc))
            continue
        totals["valid_xml"] += 1
        counts = Counter()
        for obj in root.findall("object"):
            label = (obj.findtext("name") or "").strip()
            if label:
                counts[label] += 1
            else:
                issue(xml, "空标签", "object/name 为空，该目标不计入标签数量")
        if not counts:
            issue(xml, "无有效标签", "未找到非空 object/name")

        candidates = by_stem[(xml.parent, xml.stem.casefold())]
        if not candidates:
            # XML 常保留服务器上的旧路径；仅取文件名，在当前 XML 目录查找。
            filename = (root.findtext("filename") or "").replace("\\", "/").split("/")[-1]
            candidates = by_name[(xml.parent, filename.casefold())]
        image = None
        if len(candidates) == 1:
            paired.add(candidates[0])
            if candidates[0].stat().st_size:
                image = candidates[0]
            else:
                issue(xml, "图片不可用", "对应图片为空文件")
        elif len(candidates) > 1:
            issue(xml, "图片匹配歧义", "同名候选图片不唯一，不自动选择")
        else:
            issue(xml, "缺少图片", "同目录下未找到同名图片或 filename 指定的图片")
        if image is not None:
            totals["paired_xml"] += 1
        samples.append(Sample(xml, image, counts))
    for image in track(images, "检查未配对图片", len(images)):
        if image not in paired:
            issue(image, "无可解析配对标注", "未与成功解析的 XML 配对")
    totals["objects"] = sum(sum(s.counts.values()) for s in samples)
    totals["observed_labels"] = len({label for s in samples for label in s.counts})
    totals["issues"] = len(issues)
    return samples, issues, totals


def write_csv(path, header, rows):
    # UTF-8 BOM 便于 Windows Excel 正确显示中文。
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_reports(output, source, mapping, samples, issues, totals, folders=None, copied=None):
    print(f"[统计报告] 正在生成 CSV：{output}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    objects, annotations, missing = Counter(), Counter(), Counter()
    image_sets = defaultdict(set)
    for sample in samples:
        objects.update(sample.counts)
        annotations.update(sample.counts.keys())
        for label in sample.counts:
            if sample.image is None:
                missing[label] += 1
            else:
                image_sets[label].add(sample.image)
    rows = []
    for label in sorted(set(mapping) | set(objects)):
        name, business = mapping.get(label, ("", ""))
        rows.append([
            label, name, business, "是" if label in mapping else "否",
            objects[label], annotations[label], len(image_sets[label]), missing[label],
            (folders or {}).get(label, ""), (copied or {}).get(label, 0),
        ])
    write_csv(output / "label_counts.csv", [
        "标注编码", "标注标签", "业务标签", "参考表内标签", "目标数量", "XML数量",
        "图片数量", "缺少可用图片的XML数量", "分类目录", "已复制样本对数",
    ], rows)
    write_csv(output / "issues.csv", ["相对路径", "问题类型", "详情"], issues)
    names = {
        "xml_files": "扫描XML文件数", "image_files": "扫描图片文件数",
        "valid_xml": "成功解析XML数", "invalid_xml": "解析失败XML数",
        "paired_xml": "有可用图片的XML数", "objects": "有效标签目标总数",
        "observed_labels": "实际出现类别数", "issues": "异常记录数",
    }
    write_csv(output / "summary.csv", ["项目", "值"],
              [("数据来源", str(source))] + [(name, totals[key]) for key, name in names.items()])
    print("[统计报告] 3 个 CSV 已生成", flush=True)


def check_paths(source, output):
    if not source.is_dir():
        raise ValueError(f"样本目录不存在：{source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输出目录与样本目录不能相同，也不能相互包含；请使用并列目录")


def add_arguments(parser, default_output):
    parser.add_argument("--source", type=Path, default=BASE / "sample", help="样本根目录，递归扫描所有子目录")
    parser.add_argument("--labels", type=Path, default=BASE / "Docs" / "变电领域标注标签.xlsx", help="标签参考表")
    parser.add_argument("--no-label-table", action="store_true", help="不读取参考表，仅统计实际出现标签")
    parser.add_argument("--output", type=Path, default=default_output, help="输出目录")


def load_data(args):
    source, output = args.source.resolve(), args.output.resolve()
    check_paths(source, output)
    print(f"输入目录：{source}\n输出目录：{output}", flush=True)
    print("[标签参考表] 不读取" if args.no_label_table else f"[标签参考表] 正在读取：{args.labels}", flush=True)
    mapping = {} if args.no_label_table else read_label_table(args.labels)
    samples, issues, totals = scan_samples(source)
    return source, output, mapping, samples, issues, totals


def print_summary(output, totals):
    print(f"扫描 XML {totals['xml_files']} 个，成功解析 {totals['valid_xml']} 个，解析失败 {totals['invalid_xml']} 个")
    print(f"实际出现 {totals['observed_labels']} 类，共 {totals['objects']} 个目标；异常记录 {totals['issues']} 条")
    print(f"统计结果：{output / 'label_counts.csv'}")
    print(f"异常明细：{output / 'issues.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, BASE / "run_results" / "label_statistics")
    args = parser.parse_args()
    try:
        source, output, mapping, samples, issues, totals = load_data(args)
        write_reports(output, source, mapping, samples, issues, totals)
    except (OSError, ValueError, ET.ParseError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"错误：{exc}\n")
    print_summary(output, totals)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
