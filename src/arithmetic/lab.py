from functools import partial
from lab.printed import generate_printed

generate = partial(generate_printed, "arithmetic", "计算图中算式，输入整数结果。")
