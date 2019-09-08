import re
# multiline string


def zz(original):
	print(original, ' -----------------------')
	process = re.findall(r'[a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ\s\d]+', original)
	process = list(map(lambda x: x.strip(), process))
	return process


split_re = re.compile("train_\\d{2,}")
string = """train_016075
"Ấm thi bị hư k vô điên gửi đổi thì bất mình chiệu phí khăn thì chắc liệu loại 2 giao hang chậm"
1"""
# matches all whitespace characters
split_re = re.compile("train_\\d{2,}")

file_split = split_re.split(string)
file_split = list(filter(lambda x: x != '', file_split))
for label_data in file_split:
	data = label_data.split("\n")
	data = list(filter(lambda x: x != '', data))
	label = data[-1]
	text = ' '.join(data[:-1])
	review = zz(text)
	print(label, review)


