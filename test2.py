from docx import Document

# 初始化 Word 文档
doc = Document()

n = 9335  # 总数
per_line = 20
line = []

for i in range(1, n + 1):
    line.append(f"'GOT-10k_Train_{i:06d}'")
    if i % per_line == 0 or i == n:
        # 每20个写一行，并在末尾加逗号
        doc.add_paragraph(", ".join(line) + ",")
        line = []

# 保存文档到当前文件夹
doc.save("123.docx")
print("已生成 123.docx")
