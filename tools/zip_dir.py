#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把目录压缩成 zip（UTF-8 文件名，避免 GBK 乱码）。

用法：python tools/zip_dir.py <SRC_DIR> <OUT_ZIP>
"""
import os
import sys
import zipfile


def main():
    if len(sys.argv) < 3:
        print('usage: zip_dir.py <SRC_DIR> <OUT_ZIP>')
        return 2
    pkg, out = sys.argv[1], sys.argv[2]
    if not os.path.isdir(pkg):
        print('!! 源目录不存在: %s' % pkg)
        return 3

    base = os.path.dirname(os.path.normpath(pkg))
    n = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, dirs, files in os.walk(pkg):
            dirs.sort()
            for fn in sorted(files):
                p = os.path.join(root, fn)
                arc = os.path.relpath(p, base).replace('\\', '/')
                z.write(p, arc)
                n += 1

    size = os.path.getsize(out)
    print('文件数 = %d' % n)
    print('大小   = %.2f MB' % (size / 1048576.0))
    print('输出   = %s' % out)

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        bad = [x for x in names if '\ufffd' in x]
        print('乱码条目 = %d' % len(bad))
        for x in names[:5]:
            print('   ', x)
    return 0 if (n > 0 and size > 200 * 1024) else 4


if __name__ == '__main__':
    sys.exit(main())
