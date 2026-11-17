from functools import partial
from lab.printed import generate_printed

generate = partial(generate_printed, "normal", "照图输入字符。")
