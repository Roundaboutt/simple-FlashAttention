# 文件结构

cpp_file目录里是kernel的实现，直接运行bench.py即可观察运行结果。

online-softmax目录里是softmax与V点积的优化实现，用cmake先build再运行即可。

可以尝试根据自己GPU的SRAM大小更改Br与Bc的值。